from nutrilens_ml.followup.rules import generate_questions


def test_low_confidence_triggers_generic_question() -> None:
    pred = {"items": [{"label": "apple", "confidence": 0.9}], "overall_confidence": 0.6}
    qs = generate_questions(pred)
    assert any("main ingredients" in q for q in qs)


def test_salad_without_dressing() -> None:
    pred = {
        "items": [{"label": "salad", "confidence": 0.9}],
        "overall_confidence": 0.9,
    }
    qs = generate_questions(pred)
    assert any("dressing" in q.lower() for q in qs)


def test_salad_with_dressing_does_not_prompt() -> None:
    pred = {
        "items": [
            {"label": "salad", "confidence": 0.9},
            {"label": "dressing", "confidence": 0.7},
        ],
        "overall_confidence": 0.9,
    }
    qs = generate_questions(pred)
    assert not any("dressing" in q.lower() for q in qs)


def test_stir_fry_triggers_fat_question() -> None:
    pred = {"items": [{"label": "stir-fry", "confidence": 0.9}], "overall_confidence": 0.9}
    qs = generate_questions(pred)
    assert any("butter" in q or "oil" in q for q in qs)


def test_no_duplicate_category_questions() -> None:
    pred = {
        "items": [
            {"label": "sauté", "confidence": 0.9},
            {"label": "stir-fry", "confidence": 0.9},
        ],
        "overall_confidence": 0.9,
    }
    qs = generate_questions(pred)
    # Only one fat-category question, even though two labels match.
    fat_qs = [q for q in qs if "butter" in q or "oil" in q]
    assert len(fat_qs) == 1
