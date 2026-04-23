from nutrilens_ml.eval.pour import per_liquid_report, shippable_liquids


def test_per_liquid_passes_and_fails() -> None:
    samples = [
        # olive_oil: tight errors (~2-5%) — passes.
        ("olive_oil", 10.0, 10.0),
        ("olive_oil", 19.8, 20.0),
        ("olive_oil", 30.9, 30.0),
        ("olive_oil", 40.5, 40.0),
        ("olive_oil", 5.1, 5.0),
        # cream: wide errors (40%+) — fails p90.
        ("cream", 5.0, 10.0),
        ("cream", 30.0, 20.0),
        ("cream", 50.0, 100.0),
        ("cream", 12.0, 15.0),
    ]
    reports = per_liquid_report(samples)
    by_name = {r.liquid: r for r in reports}
    assert by_name["olive_oil"].passes_v0()
    assert not by_name["cream"].passes_v0()
    assert shippable_liquids(reports) == ["olive_oil"]


def test_zero_truth_is_dropped() -> None:
    reports = per_liquid_report([("vinegar", 1.0, 0.0)])
    assert reports == []
