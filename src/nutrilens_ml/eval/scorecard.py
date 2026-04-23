"""Render release scorecards to Markdown.

The scorecard is the human-readable record of a release: what the model
scored on the frozen benchmark, which liquids pass the pour bar, how
calibration moved, and what the git SHA was. Committed under
`docs/scorecards/<task>-<semver>.md` so every release is auditable.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from nutrilens_ml.eval.pour import PerLiquidReport


@dataclass(frozen=True)
class Scorecard:
    task: str
    model_version: str
    bench_version: str
    git_sha: str
    metrics: dict[str, float]
    per_liquid: list[PerLiquidReport] | None = None
    notes: str = ""

    def to_markdown(self) -> str:
        lines = [
            f"# Scorecard: {self.task} {self.model_version}",
            "",
            f"- bench: `{self.bench_version}`",
            f"- commit: `{self.git_sha}`",
            f"- generated: {datetime.now(UTC).isoformat()}",
            "",
            "## Metrics",
            "",
            "| metric | value |",
            "|--------|-------|",
        ]
        for k, v in sorted(self.metrics.items()):
            lines.append(f"| {k} | {v:.4f} |")
        if self.per_liquid:
            lines += [
                "",
                "## Per-liquid pour error",
                "",
                "| liquid | n | median abs err % | p90 abs err % | v0 bar |",
                "|--------|---|------------------|---------------|--------|",
            ]
            for r in self.per_liquid:
                status = "PASS" if r.passes_v0() else "FAIL"
                lines.append(
                    f"| {r.liquid} | {r.n_samples} | "
                    f"{r.median_abs_err_pct:.3f} | {r.p90_abs_err_pct:.3f} | {status} |"
                )
        if self.notes:
            lines += ["", "## Notes", "", self.notes]
        return "\n".join(lines) + "\n"
