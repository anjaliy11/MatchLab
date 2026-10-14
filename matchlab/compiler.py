"""Provider-neutral, fail-closed boundary for disclosure-card compilers.

The compiler receives the full profile and is part of the trusted computing
base. Its output is treated as untrusted until this module validates it.
"""
from __future__ import annotations

import logging
import math
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Protocol

from .evals.card_tier_eval import Tier
from .schemas import DisclosureCard, Fact, SUPPORTED_FACT_CATEGORIES

_LOG = logging.getLogger(__name__)
_FACT_KEYS = frozenset({
    "category",
    "value",
    "blind_tier",
    "blind_confidence",
    "blind_form",
    "mutual_tier",
    "mutual_confidence",
    "mutual_form",
    "blind_attr",
    "deny_descriptors",
    "value_specificity",
})
_TOP_LEVEL_KEYS = frozenset({"facts"})
_TIER_VALUES = [Tier.NEVER.value, Tier.COARSE.value, Tier.EXACT.value, Tier.UNKNOWN.value]

CARD_COMPILER_OUTPUT_SCHEMA: dict[str, object] = {
    "$schema": "https://json-schema.org/draft/2020-12/schema",
    "type": "object",
    "required": ["facts"],
    "additionalProperties": False,
    "properties": {
        "facts": {
            "type": "array",
            "items": {
                "type": "object",
                "required": sorted(_FACT_KEYS),
                "additionalProperties": False,
                "properties": {
                    "category": {
                        "type": "string",
                        "enum": sorted(SUPPORTED_FACT_CATEGORIES),
                    },
                    "value": {"type": "string", "minLength": 1},
                    "blind_tier": {"type": "string", "enum": _TIER_VALUES},
                    "blind_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "blind_form": {"type": ["string", "null"]},
                    "mutual_tier": {"type": "string", "enum": _TIER_VALUES},
                    "mutual_confidence": {"type": "number", "minimum": 0, "maximum": 1},
                    "mutual_form": {"type": ["string", "null"]},
                    "blind_attr": {
                        "type": ["array", "null"],
                        "minItems": 2,
                        "maxItems": 2,
                        "items": {"type": "string", "minLength": 1},
                    },
                    "deny_descriptors": {
                        "type": "array",
                        "items": {"type": "string", "minLength": 1},
                    },
                    "value_specificity": {"type": "number", "minimum": 0, "maximum": 1},
                },
            },
        },
    },
}


class CardCompiler(Protocol):
    """Adapter contract for a quarantined compiler backend."""

    def compile(self, profile: Mapping[str, object]) -> Mapping[str, object]: ...


@dataclass(frozen=True)
class CardCompilationResult:
    card: DisclosureCard
    accepted: bool
    rejected_facts: int
    reason_codes: tuple[str, ...]


def _empty_result(
    subject_id: str,
    display_name: str,
    reason_code: str,
    rejected_facts: int = 0,
) -> CardCompilationResult:
    return CardCompilationResult(
        DisclosureCard(subject_id, display_name),
        False,
        rejected_facts,
        (reason_code,),
    )


def _validate_call(
    subject_id: str,
    display_name: str,
    minimum_confidence: float,
) -> None:
    if not isinstance(subject_id, str) or not subject_id.strip():
        raise ValueError("subject_id must be a non-empty string")
    if not isinstance(display_name, str) or not display_name.strip():
        raise ValueError("display_name must be a non-empty string")
    if (
        not isinstance(minimum_confidence, (int, float))
        or isinstance(minimum_confidence, bool)
        or not math.isfinite(minimum_confidence)
        or not 0.0 <= minimum_confidence <= 1.0
    ):
        raise ValueError("minimum_confidence must be between 0.0 and 1.0")


def _valid_string(value: object) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _same_value(left: str, right: str) -> bool:
    return " ".join(left.casefold().split()) == " ".join(right.casefold().split())


def _blind_attributes(value: object) -> tuple[str, str] | None:
    if value is None:
        return None
    if (
        isinstance(value, (list, tuple))
        and len(value) == 2
        and all(_valid_string(item) for item in value)
    ):
        return value[0].strip(), value[1].strip()
    return None


def _tier_form(
    tier_value: object,
    confidence_value: object,
    form_value: object,
    exact_value: str,
    minimum_confidence: float,
    reason_codes: set[str],
) -> str | None:
    confidence_valid = (
        isinstance(confidence_value, (int, float))
        and not isinstance(confidence_value, bool)
        and math.isfinite(confidence_value)
        and 0.0 <= confidence_value <= 1.0
    )
    if not confidence_valid:
        reason_codes.add("invalid_confidence")
    if form_value is not None and not isinstance(form_value, str):
        reason_codes.add("invalid_tier_form")

    try:
        tier = Tier(tier_value)
    except (ValueError, TypeError):
        reason_codes.add("unknown_tier")
        if form_value is not None:
            reason_codes.add("invalid_tier_form")
        return None

    if tier is Tier.UNKNOWN:
        reason_codes.add("unknown_tier")
        if form_value is not None:
            reason_codes.add("invalid_tier_form")
        return None
    if tier is Tier.NEVER:
        if form_value is not None:
            reason_codes.add("invalid_tier_form")
        return None
    if not confidence_valid:
        return None
    if confidence_value < minimum_confidence:
        reason_codes.add("low_confidence")
        return None

    if tier is Tier.EXACT:
        if form_value is not None and (
            not isinstance(form_value, str) or not _same_value(form_value, exact_value)
        ):
            reason_codes.add("invalid_tier_form")
            return None
        return exact_value

    if (
        not isinstance(form_value, str)
        or not form_value.strip()
        or _same_value(form_value, exact_value)
    ):
        reason_codes.add("invalid_tier_form")
        return None
    return form_value.strip()


def validate_compiler_output(
    output: object,
    *,
    subject_id: str,
    display_name: str,
    minimum_confidence: float,
) -> CardCompilationResult:
    """Validate strict compiler output and default unsafe tiers to never."""
    _validate_call(subject_id, display_name, minimum_confidence)
    if (
        not isinstance(output, Mapping)
        or set(output) != _TOP_LEVEL_KEYS
        or not isinstance(output.get("facts"), list)
    ):
        return _empty_result(subject_id, display_name, "invalid_output_schema")

    raw_facts: list[object] = output["facts"]
    for raw_fact in raw_facts:
        if not isinstance(raw_fact, Mapping) or set(raw_fact) != _FACT_KEYS:
            return _empty_result(
                subject_id, display_name, "invalid_output_schema", len(raw_facts)
            )

    compiled_facts: list[Fact] = []
    reason_codes: set[str] = set()
    rejected_facts = 0
    for index, raw_fact in enumerate(raw_facts):
        category = raw_fact["category"]
        value = raw_fact["value"]
        specificity = raw_fact["value_specificity"]
        descriptors = raw_fact["deny_descriptors"]

        if not isinstance(category, str) or category not in SUPPORTED_FACT_CATEGORIES:
            rejected_facts += 1
            reason_codes.add("unsupported_category")
            continue
        if not _valid_string(value):
            rejected_facts += 1
            reason_codes.add("invalid_fact")
            continue
        if (
            not isinstance(specificity, (int, float))
            or isinstance(specificity, bool)
            or not math.isfinite(specificity)
            or not 0.0 <= specificity <= 1.0
        ):
            rejected_facts += 1
            reason_codes.add("invalid_specificity")
            continue
        if (
            not isinstance(descriptors, list)
            or any(not _valid_string(descriptor) for descriptor in descriptors)
        ):
            rejected_facts += 1
            reason_codes.add("invalid_deny_descriptors")
            continue

        fact_reasons: set[str] = set()
        blind_form = _tier_form(
            raw_fact["blind_tier"],
            raw_fact["blind_confidence"],
            raw_fact["blind_form"],
            value,
            minimum_confidence,
            fact_reasons,
        )
        raw_blind_attr = raw_fact["blind_attr"]
        blind_attr = _blind_attributes(raw_blind_attr)
        if raw_blind_attr is not None and blind_attr is None:
            fact_reasons.add("invalid_blind_attributes")
        if blind_form is not None and blind_attr is None:
            blind_form = None
            fact_reasons.add("invalid_blind_attributes")
        mutual_form = _tier_form(
            raw_fact["mutual_tier"],
            raw_fact["mutual_confidence"],
            raw_fact["mutual_form"],
            value,
            minimum_confidence,
            fact_reasons,
        )
        reason_codes.update(fact_reasons)
        compiled_facts.append(Fact(
            fact_id=f"compiled-{index + 1}",
            category=category,
            value=value.strip(),
            blind_form=blind_form,
            mutual_form=mutual_form,
            blind_attr=blind_attr,
            deny_descriptors=tuple(descriptor.strip() for descriptor in descriptors),
            source="card_compiler",
            value_specificity=float(specificity),
        ))

    return CardCompilationResult(
        DisclosureCard(subject_id, display_name, compiled_facts),
        not reason_codes,
        rejected_facts,
        tuple(sorted(reason_codes)),
    )


def compile_card_safely(
    compiler: CardCompiler,
    profile: Mapping[str, object],
    *,
    subject_id: str,
    display_name: str,
    minimum_confidence: float,
) -> CardCompilationResult:
    """Call a compiler adapter and fail closed without exposing exception text."""
    _validate_call(subject_id, display_name, minimum_confidence)
    try:
        output = compiler.compile(profile)
    except Exception as exc:
        _LOG.error("card compiler failed closed (exception_type=%s)", type(exc).__name__)
        return _empty_result(subject_id, display_name, "compiler_error")
    return validate_compiler_output(
        output,
        subject_id=subject_id,
        display_name=display_name,
        minimum_confidence=minimum_confidence,
    )
