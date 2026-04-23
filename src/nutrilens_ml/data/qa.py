"""Per-sample QA checks.

Failing samples are moved to `quarantine_root/<sample_id>/` alongside a
`reason.txt`. We never silently drop; every rejection is auditable.
"""

from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

from nutrilens_ml.data.schemas import PlateSample, PourSample

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class QaOutcome:
    sample_id: str
    ok: bool
    reason: str | None = None


def _quarantine(
    quarantine_root: Path, sample_id: str, files: list[Path], reason: str
) -> None:
    target = quarantine_root / sample_id
    target.mkdir(parents=True, exist_ok=True)
    for f in files:
        if f.is_file():
            shutil.copy2(f, target / f.name)
    (target / "reason.txt").write_text(reason + "\n")


def check_plate_sample(sample: PlateSample, *, dataset_root: Path) -> QaOutcome:
    image = (dataset_root / sample.image_path).resolve()
    if not image.is_file():
        return QaOutcome(sample.sample_id, False, f"image missing: {image}")
    if not sample.items:
        return QaOutcome(sample.sample_id, False, "no items labeled")
    for item in sample.items:
        if item.mask_path is None:
            continue
        mask = (dataset_root / item.mask_path).resolve()
        if not mask.is_file():
            return QaOutcome(sample.sample_id, False, f"mask missing: {mask}")
    return QaOutcome(sample.sample_id, True)


def check_pour_sample(sample: PourSample, *, dataset_root: Path) -> QaOutcome:
    video = (dataset_root / sample.video_path).resolve()
    if not video.is_file():
        return QaOutcome(sample.sample_id, False, f"video missing: {video}")
    if sample.total_ml <= 0:
        return QaOutcome(sample.sample_id, False, f"non-positive total_ml: {sample.total_ml}")
    if sample.per_frame_cumulative_ml is not None:
        xs = sample.per_frame_cumulative_ml
        if any(b < a for a, b in zip(xs, xs[1:], strict=False)):
            return QaOutcome(sample.sample_id, False, "per-frame cumulative ml is not monotonic")
    return QaOutcome(sample.sample_id, True)


def run_qa(
    plate: list[PlateSample],
    pour: list[PourSample],
    *,
    dataset_root: Path,
    quarantine_root: Path,
) -> list[QaOutcome]:
    outcomes: list[QaOutcome] = []
    for s in plate:
        outcome = check_plate_sample(s, dataset_root=dataset_root)
        outcomes.append(outcome)
        if not outcome.ok:
            _quarantine(
                quarantine_root,
                s.sample_id,
                [dataset_root / s.image_path],
                outcome.reason or "unspecified",
            )
    for s in pour:
        outcome = check_pour_sample(s, dataset_root=dataset_root)
        outcomes.append(outcome)
        if not outcome.ok:
            _quarantine(
                quarantine_root,
                s.sample_id,
                [dataset_root / s.video_path],
                outcome.reason or "unspecified",
            )
    return outcomes
