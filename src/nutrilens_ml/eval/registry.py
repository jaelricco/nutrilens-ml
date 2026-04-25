"""Minimal S3 / R2 model registry.

Layout:
    <prefix>/<task>/<semver>/
        manifest.json   — metrics, dataset hash, training git sha, notes
        plate.onnx | pour.onnx
        plate.mlpackage | pour.mlpackage (optional, uploaded only from macOS)

Promotion is a config change on the Rust backend, not a deploy: set
`LATEST_PLATE_MODEL=<semver>` and it picks up the new artifact from the
same prefix on the next request.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from botocore.client import BaseClient


@dataclass(frozen=True)
class ModelManifest:
    task: str
    version: str  # semver
    metrics: dict[str, float]
    dataset_hash: str
    training_git_sha: str
    notes: str = ""

    def to_dict(self) -> dict[str, object]:
        return {
            "task": self.task,
            "version": self.version,
            "metrics": self.metrics,
            "dataset_hash": self.dataset_hash,
            "training_git_sha": self.training_git_sha,
            "notes": self.notes,
        }


def _client(endpoint: str | None, region: str) -> BaseClient:
    """Build an S3 / R2 client.

    Reads `S3_ACCESS_KEY_ID` / `S3_SECRET_ACCESS_KEY` from the environment
    (project convention from `Settings`), falling back to boto3's default
    credential chain (`AWS_*` env, instance metadata, etc.) when those are
    unset.
    """
    import os

    import boto3

    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        region_name=region,
        aws_access_key_id=os.environ.get("S3_ACCESS_KEY_ID"),
        aws_secret_access_key=os.environ.get("S3_SECRET_ACCESS_KEY"),
    )


def upload_model(
    *,
    bucket: str,
    prefix: str,
    manifest: ModelManifest,
    artifacts: list[Path],
    endpoint: str | None = None,
    region: str = "auto",
) -> str:
    """Upload artifacts + manifest to `<bucket>/<prefix>/<task>/<version>/`."""
    key_prefix = f"{prefix.rstrip('/')}/{manifest.task}/{manifest.version}"
    s3 = _client(endpoint, region)
    for artifact in artifacts:
        if not artifact.is_file():
            raise FileNotFoundError(artifact)
        s3.upload_file(str(artifact), bucket, f"{key_prefix}/{artifact.name}")
    s3.put_object(
        Bucket=bucket,
        Key=f"{key_prefix}/manifest.json",
        Body=json.dumps(manifest.to_dict(), indent=2).encode(),
        ContentType="application/json",
    )
    return key_prefix


def load_manifest(
    *,
    bucket: str,
    prefix: str,
    task: str,
    version: str,
    endpoint: str | None = None,
    region: str = "auto",
) -> ModelManifest:
    s3 = _client(endpoint, region)
    key = f"{prefix.rstrip('/')}/{task}/{version}/manifest.json"
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = json.loads(obj["Body"].read())
    return ModelManifest(**data)
