from nutrilens_ml.eval.drift import compute_drift
from nutrilens_ml.eval.pour import PerLiquidReport
from nutrilens_ml.eval.scorecard import Scorecard


def test_drift_matches_are_near_zero() -> None:
    a = ["pasta", "pasta", "salad", "salad", "rice"]
    report = compute_drift(a, a)
    assert report.symmetric_kl < 1e-6
    assert not report.alerts()


def test_drift_alerts_on_new_class() -> None:
    baseline = ["pasta"] * 90 + ["salad"] * 10
    observed = ["burger"] * 90 + ["fries"] * 10
    report = compute_drift(baseline, observed)
    assert report.alerts(threshold=0.1)


def test_scorecard_contains_failing_liquid() -> None:
    per_liquid = [
        PerLiquidReport(liquid="olive_oil", n_samples=50, median_abs_err_pct=0.08, p90_abs_err_pct=0.22),
        PerLiquidReport(liquid="cream", n_samples=40, median_abs_err_pct=0.40, p90_abs_err_pct=0.70),
    ]
    card = Scorecard(
        task="pour",
        model_version="0.1.0",
        bench_version="v0",
        git_sha="deadbeef",
        metrics={"median_abs_err_pct": 0.20, "p90_abs_err_pct": 0.50},
        per_liquid=per_liquid,
    )
    md = card.to_markdown()
    assert "olive_oil" in md and "PASS" in md
    assert "cream" in md and "FAIL" in md
