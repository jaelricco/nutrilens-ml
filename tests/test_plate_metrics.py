import numpy as np
import pytest

from nutrilens_ml.eval.plate import iou, mae_grams, mean_iou, release_bar_v0, top_k_accuracy


def test_iou_overlap() -> None:
    a = np.zeros((4, 4), dtype=bool)
    a[:2, :2] = True
    b = np.zeros((4, 4), dtype=bool)
    b[1:3, 1:3] = True
    assert iou(a, b) == pytest.approx(1 / 7)


def test_iou_empty_union() -> None:
    z = np.zeros((4, 4), dtype=bool)
    assert iou(z, z) == 0.0


def test_mean_iou() -> None:
    a = np.array([[1, 0], [0, 0]], dtype=bool)
    b = np.array([[1, 0], [0, 0]], dtype=bool)
    assert mean_iou([(a, b), (a, b)]) == 1.0


def test_top_k_accuracy() -> None:
    logits = np.array([[0.1, 0.2, 0.7], [0.9, 0.05, 0.05]])
    target = np.array([2, 0])
    assert top_k_accuracy(logits, target, k=1) == 1.0
    assert top_k_accuracy(logits, target, k=2) == 1.0


def test_mae_grams() -> None:
    assert mae_grams([100, 200], [90, 210]) == pytest.approx(10.0)


def test_release_bar_passes() -> None:
    ok, failures = release_bar_v0({"top5": 0.7, "mIoU": 0.6, "mae_grams_pct": 0.25})
    assert ok and failures == []


def test_release_bar_fails() -> None:
    ok, failures = release_bar_v0({"top5": 0.4, "mIoU": 0.4, "mae_grams_pct": 0.5})
    assert not ok and len(failures) == 3
