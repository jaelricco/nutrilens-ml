from collections import Counter

import pytest

from nutrilens_ml.data.splits import SplitRatios, assign_split


def test_split_is_stable() -> None:
    assert assign_split("abc") == assign_split("abc")


def test_split_respects_ratios_roughly() -> None:
    counts = Counter(assign_split(f"sample-{i}") for i in range(20_000))
    total = sum(counts.values())
    # Ratios are hash-driven so we allow a loose tolerance.
    assert abs(counts["train"] / total - 0.8) < 0.02
    assert abs(counts["val"] / total - 0.1) < 0.02
    assert abs(counts["test"] / total - 0.1) < 0.02


def test_invalid_ratios() -> None:
    with pytest.raises(ValueError, match="must sum to 1.0"):
        SplitRatios(train=0.5, val=0.3, test=0.3)
