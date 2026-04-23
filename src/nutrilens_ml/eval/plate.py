"""Plate-scanner metrics."""

from __future__ import annotations

from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import torch
    from torch import nn
    from torch.utils.data import DataLoader


def iou(pred_mask: Any, gt_mask: Any) -> float:
    import numpy as np

    p = np.asarray(pred_mask).astype(bool)
    g = np.asarray(gt_mask).astype(bool)
    inter = np.logical_and(p, g).sum()
    union = np.logical_or(p, g).sum()
    return 0.0 if union == 0 else float(inter) / float(union)


def mean_iou(pairs: Iterable[tuple[Any, Any]]) -> float:
    vals = [iou(p, g) for p, g in pairs]
    return 0.0 if not vals else sum(vals) / len(vals)


def top_k_accuracy(logits: Any, target: Any, k: int) -> float:
    import numpy as np

    arr = np.asarray(logits)
    t = np.asarray(target)
    topk = np.argsort(arr, axis=-1)[..., -k:]
    hits = (topk == t[..., None]).any(axis=-1)
    return float(hits.mean())


def mae_grams(pred_grams: Iterable[float], true_grams: Iterable[float]) -> float:
    preds = list(pred_grams)
    trues = list(true_grams)
    if not preds:
        return 0.0
    return sum(abs(p - t) for p, t in zip(preds, trues, strict=False)) / len(preds)


def macro_f1_over_loader(
    model: nn.Module, loader: DataLoader[Any], device: str
) -> float:
    """Macro-F1 of top-1 detected label per image vs the first gt label.

    Intentionally simple — it's used only as an early-stopping signal, not
    the release-bar metric (that's computed offline in the eval CLI).
    """
    import torch
    from sklearn.metrics import f1_score

    preds: list[int] = []
    trues: list[int] = []
    for images, targets in loader:
        images = [img.to(device) for img in images]
        outs = model(images)
        for out, tgt in zip(outs, targets, strict=False):
            labels = out["labels"].detach().cpu().tolist()
            scores = out["scores"].detach().cpu().tolist()
            pred = labels[scores.index(max(scores))] if scores else -1
            true_labels = tgt["labels"].tolist()
            trues.append(int(true_labels[0]) if true_labels else -1)
            preds.append(int(pred))
    if not preds:
        return 0.0
    return float(f1_score(trues, preds, average="macro", zero_division=0))


def release_bar_v0(metrics: dict[str, float]) -> tuple[bool, list[str]]:
    """Check the Phase 3 release bars. Returns (passed, failures)."""
    failures: list[str] = []
    if metrics.get("top5", 0) < 0.60:
        failures.append(f"top5 {metrics.get('top5', 0):.3f} < 0.60")
    if metrics.get("mIoU", 0) < 0.55:
        failures.append(f"mIoU {metrics.get('mIoU', 0):.3f} < 0.55")
    if metrics.get("mae_grams_pct", 1.0) > 0.30:
        failures.append(f"mae_grams_pct {metrics.get('mae_grams_pct', 1.0):.3f} > 0.30")
    return len(failures) == 0, failures
