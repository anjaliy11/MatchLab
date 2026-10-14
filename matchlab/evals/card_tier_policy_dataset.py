"""Build a policy-labelled synthetic benchmark for the card compiler stand-in.

The approved policy supplies expected tiers; the deterministic persona compiler
supplies predictions for categories it actually emits. This is not human
adjudication or evidence about a production compiler.
"""
from __future__ import annotations

from typing import TypedDict

from .card_tier_eval import Tier, tier_for_fact
from .personas import compile_card, make_personas
from ..schemas import Stage


class PolicyTierRow(TypedDict):
    item_id: str
    category: str
    expected_blind: str
    expected_mutual: str
    predicted_blind: str
    predicted_mutual: str
    confidence_blind: str
    confidence_mutual: str
    prediction_status: str
    label_basis: str
    prediction_source: str


_POLICY_TIERS: dict[str, tuple[Tier, Tier]] = {
    "employer": (Tier.COARSE, Tier.EXACT),
    "city": (Tier.COARSE, Tier.EXACT),
    "school": (Tier.NEVER, Tier.EXACT),
    "religion": (Tier.NEVER, Tier.NEVER),
    "health": (Tier.NEVER, Tier.NEVER),
    "sexual_orientation": (Tier.NEVER, Tier.NEVER),
    "caste": (Tier.NEVER, Tier.NEVER),
    "politics": (Tier.NEVER, Tier.NEVER),
    "immigration": (Tier.NEVER, Tier.NEVER),
    "finances": (Tier.NEVER, Tier.NEVER),
    "hobby": (Tier.EXACT, Tier.EXACT),
    "sector": (Tier.EXACT, Tier.EXACT),
    "region": (Tier.EXACT, Tier.EXACT),
}

_PREDICTION_SOURCE = "synthetic_rule_based_standin.compile_card"


def build_policy_dataset(seed: int) -> tuple[list[PolicyTierRow], dict[str, object]]:
    """Return policy labels, supported stand-in predictions, and provenance metrics."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    persona = make_personas(1, seed=seed)[0]
    card = compile_card(persona, error_rate=0.0, seed=seed)
    compiled_facts = {fact.category: fact for fact in card.facts}
    rows: list[PolicyTierRow] = []
    supported: list[tuple[str, Tier, Tier, Tier, Tier]] = []

    for index, (category, (expected_blind, expected_mutual)) in enumerate(_POLICY_TIERS.items(), start=1):
        fact = compiled_facts.get(category)
        if fact is None:
            predicted_blind = predicted_mutual = Tier.UNKNOWN
            prediction_status = "unsupported"
        else:
            predicted_blind = tier_for_fact(fact, Stage.BLIND)
            predicted_mutual = tier_for_fact(fact, Stage.MUTUAL)
            prediction_status = "supported_by_standin"
            supported.append(
                (category, expected_blind, expected_mutual, predicted_blind, predicted_mutual)
            )
        rows.append({
            "item_id": f"policy-{category}-{index:03d}",
            "category": category,
            "expected_blind": expected_blind.value,
            "expected_mutual": expected_mutual.value,
            "predicted_blind": predicted_blind.value,
            "predicted_mutual": predicted_mutual.value,
            "confidence_blind": "",
            "confidence_mutual": "",
            "prediction_status": prediction_status,
            "label_basis": "approved_policy_rubric",
            "prediction_source": _PREDICTION_SOURCE,
        })

    stages = {}
    for stage_index, stage in enumerate(("blind", "mutual")):
        expected_index = stage_index + 1
        predicted_index = stage_index + 3
        matches = sum(case[expected_index] is case[predicted_index] for case in supported)
        stages[stage] = {
            "scored_supported_categories": len(supported),
            "policy_matches": matches,
            "policy_mismatches": len(supported) - matches,
            "policy_agreement": matches / len(supported) if supported else None,
        }

    by_category = {
        row["category"]: {
            "prediction_status": row["prediction_status"],
            "expected_blind": row["expected_blind"],
            "expected_mutual": row["expected_mutual"],
            "predicted_blind": row["predicted_blind"],
            "predicted_mutual": row["predicted_mutual"],
        }
        for row in rows
    }
    report: dict[str, object] = {
        "status": "synthetic_policy_check_not_compiler_accuracy",
        "label_basis": "approved_policy_rubric",
        "human_adjudication": "not_performed",
        "prediction_source": _PREDICTION_SOURCE,
        "confidence_available": False,
        "confidence_note": "The stand-in compiler does not emit confidence scores.",
        "items": len(rows),
        "seed": seed,
        "injected_error_rate": 0.0,
        "supported_categories": len(supported),
        "unsupported_categories": len(rows) - len(supported),
        "stages": stages,
        "by_category": by_category,
    }
    return rows, report
