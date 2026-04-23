"""S3 / R2 ingestion with content-hash caching.

`source` is either `s3://bucket/prefix` or a local directory path. We pull
into `cache_root`, keyed by the object's content hash (from the ETag when
available, recomputed locally otherwise). Re-runs are no-ops if the content
hasn't changed.
"""

from __future__ import annotations

import hashlib
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import urlparse

if TYPE_CHECKING:
    from collections.abc import Iterable

    from botocore.client import BaseClient

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class IngestResult:
    fetched: int
    cached: int
    skipped_quarantined: int


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _cache_path(cache_root: Path, content_hash: str, original_name: str) -> Path:
    # 2-char prefix directories keep any single dir from blowing up.
    return cache_root / content_hash[:2] / content_hash[2:4] / f"{content_hash}-{original_name}"


def _s3_client(endpoint: str | None, region: str) -> BaseClient:
    import boto3

    return boto3.client("s3", endpoint_url=endpoint, region_name=region)


def _iter_s3(client: BaseClient, bucket: str, prefix: str) -> Iterable[tuple[str, str]]:
    paginator = client.get_paginator("list_objects_v2")
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        for obj in page.get("Contents", []) or []:
            key = obj["Key"]
            etag = obj.get("ETag", "").strip('"')
            yield key, etag


def ingest(
    source: str,
    cache_root: Path,
    *,
    s3_endpoint: str | None = None,
    s3_region: str = "auto",
    suffixes: tuple[str, ...] = (".jpg", ".jpeg", ".png", ".mp4", ".json"),
) -> IngestResult:
    cache_root.mkdir(parents=True, exist_ok=True)

    if source.startswith("s3://"):
        parsed = urlparse(source)
        bucket, prefix = parsed.netloc, parsed.path.lstrip("/")
        client = _s3_client(s3_endpoint, s3_region)
        fetched = cached = 0
        for key, etag in _iter_s3(client, bucket, prefix):
            if not key.lower().endswith(suffixes):
                continue
            # S3 ETag is an MD5 for single-part uploads. Safe as a cache key.
            content_hash = etag or hashlib.sha256(key.encode()).hexdigest()
            name = Path(key).name
            dest = _cache_path(cache_root, content_hash, name)
            if dest.exists():
                cached += 1
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            logger.info("fetch s3://%s/%s -> %s", bucket, key, dest)
            client.download_file(bucket, key, str(dest))
            fetched += 1
        return IngestResult(fetched=fetched, cached=cached, skipped_quarantined=0)

    src = Path(source)
    if not src.is_dir():
        raise FileNotFoundError(f"local source not found: {src}")
    fetched = cached = 0
    for path in src.rglob("*"):
        if not path.is_file() or not path.suffix.lower() in suffixes:
            continue
        content_hash = _hash_file(path)
        dest = _cache_path(cache_root, content_hash, path.name)
        if dest.exists():
            cached += 1
            continue
        dest.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, dest)
        fetched += 1
    return IngestResult(fetched=fetched, cached=cached, skipped_quarantined=0)
