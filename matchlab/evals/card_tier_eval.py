"""Evaluate compiled disclosure tiers against adjudicated human labels.

The evaluator consumes aggregates-ready annotations and never emits item IDs or
fact text. It does not infer gold labels or manufacture compiler predictions.
"""
from __future__ import annotations

import csv
import math
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Iterable

from ..schemas import Fact, Stage, SUPPORTED_FACT_CATEGORIES


class Tier(str, Enum):
    NEVER = "never"
    COARSE = "coarse"
    EXACT = "exact"
    UNKNOWN = "unknown"


_RANK = {Tier.NEVER: 0, Tier.COARSE: 1, Tier.EXACT: 2}
_STAGES = ("blind", "mutual")
_CSV_COLUMNS = (
    "item_id",
    "category",
    "predicted_blind",
    "confidence_blind",
    "rater1_blind",
    "rater2_blind",
    "adjudicated_blind",
    "predicted_mutual",
    "confidence_mutual",
    "rater1_mutual",
    "rater2_mutual",
    "adjudicated_mutual",
)
_CATEGORIES = SUPPORTED_FACT_CATEGORIES


@dataclass(frozen=True)
class CardTierCase:
    item_id: str
    category: str
    predicted_blind: Tier
    confidence_blind: float
    rater1_blind: Tier
    rater2_blind: Tier
    adjudicated_blind: Tier
    predicted_mutual: Tier
    confidence_mutual: float
    rater1_mutual: Tier
    rater2_mutual: Tier
    adjudicated_mutual: Tier


def tier_for_fact(fact: Fact, stage: Stage) -> Tier:
    """Map a compiled Fact's effective stage form to its disclosure tier."""
    if stage is Stage.BLIND:
        form = fact.blind_form
    elif stage is Stage.MUTUAL:
        form = fact.mutual_form
    else:
        raise ValueError("unsupported disclosure stage")
    if not form:
        return Tier.NEVER
    return Tier.EXACT if form == fact.value else Tier.COARSE


def _validate_config(
    false_safe_cost: float,
    overrestriction_cost: float,
    minimum_confidence: float,
) -> None:
    costs = (false_safe_cost, overrestriction_cost)
    if not all(
        isinstance(cost, (int, float)) and not isinstance(cost, bool)
        and math.isfinite(cost) and cost > 0
        for cost in costs
    ):
        raise ValueError("cost weights must be finite positive numbers")
    if false_safe_cost <= overrestriction_cost:
        raise ValueError("false-safe cost must exceed over-restriction cost")
    if (not isinstance(minimum_confidence, (int, float)) or isinstance(minimum_confidence, bool)
            or not math.isfinite(minimum_confidence) or not 0.0 <= minimum_confidence <= 1.0):
        raise ValueError("minimum confidence must be between 0.0 and 1.0")


def _tier(value: str, *, allow_unknown: bool, line: int, column: str) -> Tier:
    try:
        tier = Tier(value.strip().casefold())
    except (ValueError, AttributeError):
        raise ValueError(f"row {line}: invalid tier in {column}") from None
    if tier is Tier.UNKNOWN and not allow_unknown:
        raise ValueError(f"row {line}: unknown tier is not valid in {column}")
    return tier


def _confidence(value: str, *, line: int, column: str) -> float:
    try:
        confidence = float(value)
    except (TypeError, ValueError):
        raise ValueError(f"row {line}: invalid confidence in {column}") from None
    if not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
        raise ValueError(f"row {line}: confidence out of range in {column}")
    return confidence


def load_tier_cases(path: str | Path) -> list[CardTierCase]:
    """Load the fixed two-rater CSV schema; errors identify rows/columns only."""
    cases: list[CardTierCase] = []
    seen_ids: set[str] = set()
    with Path(path).open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        if reader.fieldnames is None or len(reader.fieldnames) != len(set(reader.fieldnames)):
            raise ValueError("CSV header is missing or contains duplicate columns")
        if set(reader.fieldnames) != set(_CSV_COLUMNS):
            raise ValueError("CSV header does not match the required privacy-minimal schema")

        for line, row in enumerate(reader, start=2):
            if None in row:
                raise ValueError(f"row {line}: extra CSV fields")
            item_id = (row.get("item_id") or "").strip()
            category = (row.get("category") or "").strip()
            if not item_id or not category:
                raise ValueError(f"row {line}: item_id and category are required")
            if category not in _CATEGORIES:
                raise ValueError(f"row {line}: category is not in the supported category set")
            if item_id in seen_ids:
                raise ValueError(f"row {line}: duplicate item_id")
            seen_ids.add(item_id)

            stage_values: dict[str, tuple[Tier, float, Tier, Tier, Tier]] = {}
            for stage in _STAGES:
                pred_column = f"predicted_{stage}"
                confidence_column = f"confidence_{stage}"
                rater1_column = f"rater1_{stage}"
                rater2_column = f"rater2_{stage}"
                adjudicated_column = f"adjudicated_{stage}"
                stage_values[stage] = (
                    _tier(row.get(pred_column, ""), allow_unknown=True, line=line, column=pred_column),
                    _confidence(row.get(confidence_column, ""), line=line, column=confidence_column),
                    _tier(row.get(rater1_column, ""), allow_unknown=False, line=line, column=rater1_column),
                    _tier(row.get(rater2_column, ""), allow_unknown=False, line=line, column=rater2_column),
                    _tier(row.get(adjudicated_column, ""), allow_unknown=False, line=line,
                          column=adjudicated_column),
                )

            blind = stage_values["blind"]
            mutual = stage_values["mutual"]
            cases.append(CardTierCase(
                item_id=item_id,
                category=category,
                predicted_blind=blind[0],
                confidence_blind=blind[1],
                rater1_blind=blind[2],
                rater2_blind=blind[3],
                adjudicated_blind=blind[4],
                predicted_mutual=mutual[0],
                confidence_mutual=mutual[1],
                rater1_mutual=mutual[2],
                rater2_mutual=mutual[3],
                adjudicated_mutual=mutual[4],
            ))
    if not cases:
        raise ValueError("CSV contains no labelled rows")
    return cases


def _rater_metrics(cases: list[CardTierCase], stage: str) -> dict[str, float | int | None]:
    rater1 = [getattr(case, f"rater1_{stage}") for case in cases]
    rater2 = [getattr(case, f"rater2_{stage}") for case in cases]
    agreements = sum(first is second for first, second in zip(rater1, rater2))
    count = len(cases)
    first_counts = Counter(rater1)
    second_counts = Counter(rater2)
    expected_agreement = sum(
        first_counts[tier] * second_counts[tier] for tier in (Tier.NEVER, Tier.COARSE, Tier.EXACT)
    ) / (count * count)
    observed_agreement = agreements / count
    kappa = (
        None if expected_agreement == 1.0
        else (observed_agreement - expected_agreement) / (1.0 - expected_agreement)
    )
    return {
        "items": count,
        "exact_agreement": observed_agreement,
        "cohens_kappa": kappa,
    }


def _score(
    cases: list[CardTierCase],
    stage: str,
    *,
    default_deny: bool,
    false_safe_cost: float,
    overrestriction_cost: float,
    minimum_confidence: float,
) -> dict[str, int | float | None]:
    total = len(cases)
    correct = 0
    false_safe = 0
    over_disclosure = 0
    over_restriction = 0
    permitted_gold = 0
    never_gold = 0
    unscored_predictions = 0
    loss = 0.0

    for case in cases:
        predicted = getattr(case, f"predicted_{stage}")
        confidence = getattr(case, f"confidence_{stage}")
        gold = getattr(case, f"adjudicated_{stage}")
        if predicted is Tier.UNKNOWN and not default_deny:
            unscored_predictions += 1
            never_gold += gold is Tier.NEVER
            permitted_gold += gold is not Tier.NEVER
            continue
        predicted_rank = _RANK.get(predicted, _RANK[Tier.NEVER])
        if default_deny and (predicted is Tier.UNKNOWN or confidence < minimum_confidence):
            predicted_rank = _RANK[Tier.NEVER]
        gold_rank = _RANK[gold]

        correct += predicted_rank == gold_rank
        never_gold += gold is Tier.NEVER
        permitted_gold += gold is not Tier.NEVER
        false_safe += gold is Tier.NEVER and predicted_rank > _RANK[Tier.NEVER]
        if predicted_rank > gold_rank:
            over_disclosure += 1
            loss += false_safe_cost * (predicted_rank - gold_rank)
        elif predicted_rank < gold_rank:
            over_restriction += 1
            loss += overrestriction_cost * (gold_rank - predicted_rank)

    return {
        "items": total,
        "tier_accuracy": correct / total if total else None,
        "unscored_prediction_count": unscored_predictions,
        "false_safe_count": false_safe,
        "false_safe_rate": false_safe / never_gold if never_gold else None,
        "over_disclosure_count": over_disclosure,
        "over_disclosure_rate": over_disclosure / total if total else None,
        "over_restriction_count": over_restriction,
        "over_restriction_rate": over_restriction / permitted_gold if permitted_gold else None,
        "asymmetric_loss": loss,
    }


def _stage_report(
    cases: list[CardTierCase],
    stage: str,
    *,
    false_safe_cost: float,
    overrestriction_cost: float,
    minimum_confidence: float,
) -> dict[str, object]:
    low_confidence_downgrades = sum(
        getattr(case, f"confidence_{stage}") < minimum_confidence
        and getattr(case, f"predicted_{stage}") not in (Tier.NEVER, Tier.UNKNOWN)
        for case in cases
    )
    unknown_predictions = sum(getattr(case, f"predicted_{stage}") is Tier.UNKNOWN for case in cases)

    def group_report(group: list[CardTierCase]) -> dict[str, object]:
        return {
            "raters": _rater_metrics(group, stage),
            "raw": _score(
                group, stage, default_deny=False, false_safe_cost=false_safe_cost,
                overrestriction_cost=overrestriction_cost, minimum_confidence=minimum_confidence,
            ),
            "default_deny": _score(
                group, stage, default_deny=True, false_safe_cost=false_safe_cost,
                overrestriction_cost=overrestriction_cost, minimum_confidence=minimum_confidence,
            ),
        }

    by_category = {
        category: group_report([case for case in cases if case.category == category])
        for category in sorted({case.category for case in cases})
    }
    overall = group_report(cases)
    overall["by_category"] = by_category
    overall["low_confidence_downgrade_count"] = low_confidence_downgrades
    overall["unknown_prediction_count"] = unknown_predictions
    return overall


def evaluate_tiers(
    cases: Iterable[CardTierCase],
    false_safe_cost: float,
    overrestriction_cost: float,
    minimum_confidence: float,
) -> dict[str, object]:
    """Return raw and default-deny metrics; caller must provide policy costs."""
    _validate_config(false_safe_cost, overrestriction_cost, minimum_confidence)
    case_list = list(cases)
    if not case_list:
        raise ValueError("at least one labelled case is required")

    unique_ids = [case.item_id for case in case_list]
    if any(not item_id for item_id in unique_ids) or len(unique_ids) != len(set(unique_ids)):
        raise ValueError("item IDs must be non-empty and unique")
    if any(case.category not in _CATEGORIES for case in case_list):
        raise ValueError("all cases must use a supported category")
    for case in case_list:
        for stage in _STAGES:
            for suffix in ("predicted", "rater1", "rater2", "adjudicated"):
                tier = getattr(case, f"{suffix}_{stage}")
                if not isinstance(tier, Tier):
                    raise ValueError("all tier labels must use the Tier enum")
                if tier is Tier.UNKNOWN and suffix != "predicted":
                    raise ValueError("unknown is valid only for compiler predictions")
            confidence = getattr(case, f"confidence_{stage}")
            if (not isinstance(confidence, (int, float)) or isinstance(confidence, bool)
                    or not math.isfinite(confidence) or not 0.0 <= confidence <= 1.0):
                raise ValueError("all confidence values must be between 0.0 and 1.0")

    return {
        "label_basis": "caller_supplied_adjudicated_labels",
        "label_provenance_verified": False,
        "items": len(case_list),
        "configuration": {
            "false_safe_cost": false_safe_cost,
            "overrestriction_cost": overrestriction_cost,
            "minimum_confidence": minimum_confidence,
            "tier_order": [Tier.NEVER.value, Tier.COARSE.value, Tier.EXACT.value],
            "unknown_prediction_policy": Tier.NEVER.value,
        },
        "stages": {
            stage: _stage_report(
                case_list, stage, false_safe_cost=false_safe_cost,
                overrestriction_cost=overrestriction_cost, minimum_confidence=minimum_confidence,
            )
            for stage in _STAGES
        },
    }
