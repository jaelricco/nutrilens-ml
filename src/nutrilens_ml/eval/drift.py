"""Output-distribution drift detection.

We log production predictions (hashed inputs + label, confidence) and
compare the weekly histogram against the frozen benchmark baseline using
symmetric KL divergence. Symmetric because one-sided KL explodes when a
new class appears in production.

Alert fires when the divergence exceeds `threshold` — the default is
conservative; tune once we have 4+ weeks of baseline.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass


@dataclass(frozen=True)
class DriftReport:
    kl_forward: float
    kl_reverse: float

    @property
    def symmetric_kl(self) -> float:
        return 0.5 * (self.kl_forward + self.kl_reverse)

    def alerts(self, threshold: float = 0.10) -> bool:
        return self.symmetric_kl > threshold


def _smoothed_distribution(counts: Counter[str], alpha: float = 1e-6) -> dict[str, float]:
    total = sum(counts.values()) + alpha * max(len(counts), 1)
    return {k: (v + alpha) / total for k, v in counts.items()}


def _kl(p: dict[str, float], q: dict[str, float]) -> float:
    import math

    keys = set(p) | set(q)
    eps = 1e-12
    return sum(
        p.get(k, eps) * math.log(p.get(k, eps) / max(q.get(k, eps), eps)) for k in keys
    )


def compute_drift(baseline_labels: list[str], observed_labels: list[str]) -> DriftReport:
    bc = Counter(baseline_labels)
    oc = Counter(observed_labels)
    # Ensure both distributions have the same support to keep KL bounded.
    for k in bc.keys() | oc.keys():
        bc.setdefault(k, 0)
        oc.setdefault(k, 0)
    p = _smoothed_distribution(bc)
    q = _smoothed_distribution(oc)
    return DriftReport(kl_forward=_kl(p, q), kl_reverse=_kl(q, p))
