"""ONNX Runtime inference wrappers for the server path.

Sessions are cached per process — loading an ONNX model is measured in
hundreds of milliseconds, so we avoid paying that on every request.

`plate_infer` and `pour_infer` are intentionally thin — preprocessing,
postprocessing, and calibration live in their own modules so the server
file stays obvious to read.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import numpy as np
    import onnxruntime as ort


@lru_cache(maxsize=8)
def _session(model_path: str) -> "ort.InferenceSession":
    import onnxruntime as ort

    providers = ort.get_available_providers()
    # Prefer CUDA when present; fall back to CPU.
    preferred = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in providers]
    return ort.InferenceSession(model_path, providers=preferred)


def plate_infer(model_path: Path, image_chw: "np.ndarray") -> dict[str, object]:
    sess = _session(str(model_path))
    boxes, labels, scores, masks = sess.run(None, {"image": image_chw})
    return {
        "boxes": boxes.tolist(),
        "labels": labels.tolist(),
        "scores": scores.tolist(),
        "masks_summary": [
            {"area": float((m > 0.5).sum()), "shape": list(m.shape)} for m in masks
        ],
    }


def pour_infer(
    model_path: Path, clip_cthw: "np.ndarray", liquid_idx: int
) -> dict[str, float]:
    import numpy as np

    sess = _session(str(model_path))
    out = sess.run(
        None,
        {
            "clip": clip_cthw.astype(np.float32),
            "liquid_idx": np.array([liquid_idx], dtype=np.int64),
        },
    )
    total_ml = float(out[0].reshape(-1)[0])
    return {"total_ml": max(0.0, total_ml)}
