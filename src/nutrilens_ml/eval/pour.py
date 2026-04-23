"""Pour-detector metrics.

Per-liquid breakdown is load-bearing — the release bar only ships liquids
that individually pass, rather than hiding minority-class failures under a
global average.
"""

from __future__ import annotations

from dataclasses import dataclass
from statistics import median


@dataclass(frozen=True)
class PerLiquidReport:
    liquid: str
    n_samples: int
    median_abs_err_pct: float
    p90_abs_err_pct: float

    def passes_v0(self) -> bool:
        return self.median_abs_err_pct <= 0.15 and self.p90_abs_err_pct <= 0.35


def _percentile(xs: list[float], pct: float) -> float:
    if not xs:
        return 0.0
    s = sorted(xs)
    idx = min(len(s) - 1, int(round(pct * (len(s) - 1))))
    return s[idx]


def per_liquid_report(
    samples: list[tuple[str, float, float]],
) -> list[PerLiquidReport]:
    """Group by liquid, return one report per liquid.

    `samples`: list of (liquid_name, pred_ml, true_ml).
    """
    buckets: dict[str, list[tuple[float, float]]] = {}
    for liquid, pred, true in samples:
        if true <= 0:
            continue
        buckets.setdefault(liquid, []).append((pred, true))

    reports: list[PerLiquidReport] = []
    for liquid, pairs in sorted(buckets.items()):
        errs_pct = [abs(p - t) / t for p, t in pairs]
        reports.append(
            PerLiquidReport(
                liquid=liquid,
                n_samples=len(pairs),
                median_abs_err_pct=float(median(errs_pct)),
                p90_abs_err_pct=_percentile(errs_pct, 0.9),
            )
        )
    return reports


def shippable_liquids(reports: list[PerLiquidReport]) -> list[str]:
    return [r.liquid for r in reports if r.passes_v0()]
