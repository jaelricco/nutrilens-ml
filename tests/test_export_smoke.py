"""Export smoke test on a trivial model.

We don't run the Mask R-CNN path in CI — that would need a GPU and minutes
of wallclock. This test proves the export/parity pipeline itself is healthy
against a 2-layer MLP, which is enough to catch regressions in the torch ->
ONNX -> onnxruntime toolchain.
"""

from __future__ import annotations

from pathlib import Path

import pytest

pytest.importorskip("torch")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

import torch  # noqa: E402
from torch import nn  # noqa: E402

from nutrilens_ml.export.convert import export_to_onnx, parity_check  # noqa: E402


class _Tiny(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Sequential(nn.Linear(8, 16), nn.ReLU(), nn.Linear(16, 4))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


def test_onnx_export_and_parity(tmp_path: Path) -> None:
    model = _Tiny().eval()
    example = torch.randn(1, 8)
    onnx_path = tmp_path / "tiny.onnx"
    export_to_onnx(
        model,
        example,
        onnx_path,
        input_names=("image",),
        output_names=("out",),
    )
    assert onnx_path.is_file()
    delta = parity_check(model, onnx_path, example, tol=1e-4)
    assert delta < 1e-4
