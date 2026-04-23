"""Rule-based follow-up question engine.

Plain Python, trivially testable. A prediction comes in, a list of questions
comes out. Rule order is preserved; the first matching rule for a given
hidden-ingredient category wins so we don't ask three overlapping questions.

If rules start feeling insufficient we'll add a model-backed follow-up step,
but not before — rules are cheap, inspectable, and good enough for v0.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

Prediction = dict[str, object]  # { "items": [...], "overall_confidence": float }


@dataclass(frozen=True)
class Rule:
    category: str  # e.g. "fat", "dressing", "sauce"
    question: str
    matches: Callable[[Prediction], bool]


def _has_label(pred: Prediction, *labels: str) -> bool:
    items = pred.get("items", [])
    if not isinstance(items, list):
        return False
    return any(
        isinstance(item, dict)
        and isinstance(item.get("label"), str)
        and item["label"].lower() in {lbl.lower() for lbl in labels}
        for item in items
    )


def _confidence_below(pred: Prediction, threshold: float) -> bool:
    conf = pred.get("overall_confidence", 1.0)
    return isinstance(conf, (int, float)) and float(conf) < threshold


# Order matters: more specific rules come first.
DEFAULT_RULES: list[Rule] = [
    Rule(
        category="fat",
        question="Was this cooked with butter or oil?",
        matches=lambda p: _has_label(p, "sauté", "stir-fry", "fried_vegetables"),
    ),
    Rule(
        category="dressing",
        question="Which dressing, and roughly how much?",
        matches=lambda p: _has_label(p, "salad", "mixed_greens") and not _has_label(p, "dressing"),
    ),
    Rule(
        category="sauce",
        question="Is there a sauce on this dish we should include?",
        matches=lambda p: _has_label(p, "pasta", "rice_bowl", "noodles")
        and not _has_label(p, "sauce", "pesto", "tomato_sauce"),
    ),
    Rule(
        category="low_confidence",
        question="We weren't sure about this one — can you tell us the main ingredients?",
        matches=lambda p: _confidence_below(p, 0.80),
    ),
]


def generate_questions(pred: Prediction, rules: list[Rule] | None = None) -> list[str]:
    rules = rules or DEFAULT_RULES
    seen: set[str] = set()
    out: list[str] = []
    for rule in rules:
        if rule.category in seen:
            continue
        if rule.matches(pred):
            out.append(rule.question)
            seen.add(rule.category)
    return out
