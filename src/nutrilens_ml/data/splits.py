"""Deterministic hash-based dataset splits.

Why hash, not shuffle+index: adding samples shouldn't reshuffle existing ones
into different splits. A stable hash of `sample_id` pins each sample to a
split for the lifetime of that sample.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Literal

SplitName = Literal["train", "val", "test"]


@dataclass(frozen=True)
class SplitRatios:
    train: float = 0.8
    val: float = 0.1
    test: float = 0.1

    def __post_init__(self) -> None:
        total = self.train + self.val + self.test
        if abs(total - 1.0) > 1e-6:
            raise ValueError(f"split ratios must sum to 1.0, got {total}")


def assign_split(sample_id: str, ratios: SplitRatios = SplitRatios()) -> SplitName:
    bucket = int(hashlib.sha256(sample_id.encode()).hexdigest(), 16) / 2**256
    if bucket < ratios.train:
        return "train"
    if bucket < ratios.train + ratios.val:
        return "val"
    return "test"
