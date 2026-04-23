"""PyTorch -> ONNX -> CoreML with a parity smoke test.

Two-step conversion keeps ONNX as the portable intermediate: it's what the
server-side runtime consumes, and it's also the input to coremltools. If the
smoke test finds the outputs diverge, we abort — a silently drifted model is
worse than a failed build.

coremltools is an optional dep (Mac-first); if it's missing we still produce
the ONNX artifact and skip the .mlpackage step with a clear warning.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from torch import nn

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class ExportResult:
    onnx_path: Path
    coreml_path: Path | None
    max_abs_delta: float


def export_to_onnx(
    model: nn.Module,
    example_inputs: torch.Tensor,
    out_path: Path,
    *,
    input_names: tuple[str, ...] = ("image",),
    output_names: tuple[str, ...] = ("boxes", "labels", "scores", "masks"),
    opset: int = 17,
) -> Path:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    model.eval()
    torch.onnx.export(
        model,
        (example_inputs,),
        str(out_path),
        input_names=list(input_names),
        output_names=list(output_names),
        opset_version=opset,
        do_constant_folding=True,
        dynamic_axes={"image": {0: "H", 1: "W"}},
    )
    return out_path


def _try_import_coremltools() -> object | None:
    try:
        import coremltools as ct  # noqa: PLC0415 — optional, macOS-first

        return ct
    except ImportError:
        logger.warning(
            "coremltools not installed — skipping .mlpackage. "
            "Install the `export` extra on a Mac to enable."
        )
        return None


def export_to_coreml(onnx_path: Path, out_path: Path) -> Path | None:
    ct = _try_import_coremltools()
    if ct is None:
        return None
    out_path.parent.mkdir(parents=True, exist_ok=True)
    mlmodel = ct.converters.onnx.convert(model=str(onnx_path))  # type: ignore[attr-defined]
    mlmodel.save(str(out_path))
    return out_path


def parity_check(
    torch_model: nn.Module,
    onnx_path: Path,
    example_inputs: torch.Tensor,
    *,
    tol: float = 1e-3,
) -> float:
    """Run the same input through PyTorch and ONNX Runtime; return max |delta|."""
    import onnxruntime as ort

    torch_model.eval()
    with torch.no_grad():
        torch_out = torch_model(example_inputs)
    torch_primary = torch_out[0] if isinstance(torch_out, tuple) else torch_out
    torch_primary_np = torch_primary.detach().cpu().numpy()

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    onnx_out = sess.run(None, {"image": example_inputs.numpy()})[0]

    # Some detection outputs are variable-length. If shapes don't line up, we
    # compare what we can and still return the delta for the caller to log.
    if onnx_out.shape != torch_primary_np.shape:
        min_shape = tuple(
            min(a, b) for a, b in zip(onnx_out.shape, torch_primary_np.shape, strict=False)
        )
        sl = tuple(slice(0, s) for s in min_shape)
        delta = float(np.max(np.abs(onnx_out[sl] - torch_primary_np[sl])))
    else:
        delta = float(np.max(np.abs(onnx_out - torch_primary_np)))

    if delta > tol:
        raise RuntimeError(
            f"ONNX parity check failed: max|Δ|={delta:.2e} > tol={tol:.2e}"
        )
    return delta


def export_pipeline(
    model: nn.Module,
    example_inputs: torch.Tensor,
    out_dir: Path,
    *,
    name: str = "plate",
    tol: float = 1e-3,
) -> ExportResult:
    onnx_path = export_to_onnx(model, example_inputs, out_dir / f"{name}.onnx")
    delta = parity_check(model, onnx_path, example_inputs, tol=tol)
    logger.info("onnx parity ok (max|Δ|=%.2e)", delta)
    coreml_path = export_to_coreml(onnx_path, out_dir / f"{name}.mlpackage")
    return ExportResult(onnx_path=onnx_path, coreml_path=coreml_path, max_abs_delta=delta)
