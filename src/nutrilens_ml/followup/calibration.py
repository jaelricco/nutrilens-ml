"""Temperature scaling + expected calibration error (ECE).

Temperature scaling is the simplest calibration that consistently works: one
learned scalar `T` that divides logits before softmax. We fit `T` on the
validation set by minimising NLL. If calibration doesn't improve ECE on the
val set, we don't ship it — a miscalibrated confidence is worse than an
uncalibrated one because downstream follow-up thresholds depend on it.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn


@dataclass(frozen=True)
class CalibrationResult:
    temperature: float
    ece_before: float
    ece_after: float

    @property
    def improves(self) -> bool:
        return self.ece_after < self.ece_before


def expected_calibration_error(
    probs: torch.Tensor, targets: torch.Tensor, n_bins: int = 15
) -> float:
    """Standard ECE: binned gap between top-1 confidence and accuracy."""
    conf, preds = probs.max(dim=-1)
    correct = (preds == targets).float()
    bin_edges = torch.linspace(0.0, 1.0, n_bins + 1)
    total = probs.size(0)
    ece = torch.tensor(0.0)
    for i in range(n_bins):
        in_bin = (conf > bin_edges[i]) & (conf <= bin_edges[i + 1])
        if not in_bin.any():
            continue
        bin_conf = conf[in_bin].mean()
        bin_acc = correct[in_bin].mean()
        ece = ece + (in_bin.sum().float() / total) * (bin_conf - bin_acc).abs()
    return float(ece)


class TemperatureScaler(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.log_t = nn.Parameter(torch.zeros(()))

    @property
    def temperature(self) -> float:
        return float(self.log_t.exp())

    def forward(self, logits: torch.Tensor) -> torch.Tensor:
        return logits / self.log_t.exp()


def fit_temperature(
    logits: torch.Tensor,
    targets: torch.Tensor,
    *,
    lr: float = 0.01,
    max_iter: int = 50,
) -> CalibrationResult:
    """Fit a single temperature on the validation set.

    Returns the result even when calibration fails to improve ECE — the
    caller decides whether to ship it based on `.improves`.
    """
    probs_before = torch.softmax(logits, dim=-1)
    ece_before = expected_calibration_error(probs_before, targets)

    scaler = TemperatureScaler()
    optimizer = torch.optim.LBFGS([scaler.log_t], lr=lr, max_iter=max_iter)
    nll = nn.CrossEntropyLoss()

    def _closure() -> torch.Tensor:
        optimizer.zero_grad()
        loss = nll(scaler(logits), targets)
        loss.backward()
        return loss

    optimizer.step(_closure)

    probs_after = torch.softmax(scaler(logits), dim=-1)
    ece_after = expected_calibration_error(probs_after, targets)
    return CalibrationResult(
        temperature=scaler.temperature,
        ece_before=ece_before,
        ece_after=ece_after,
    )
