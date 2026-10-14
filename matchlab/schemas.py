"""Shared types.

A DisclosureCard is the ONLY view of a person that a live pitch agent is allowed to
see. It is compiled offline (by a quarantined step) from the full profile. Each Fact
says what may be said at each stage of the pitch flow:

    BLIND   the first pitch, before anyone has said yes
    MUTUAL  after both people said yes

`value` is the exact private value. `blind_form` / `mutual_form` are the only strings
the agent may say at those stages (None = nothing / never). MUTUAL does not
inherit the BLIND form.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum


class Stage(IntEnum):
    BLIND = 0
    MUTUAL = 1


SUPPORTED_FACT_CATEGORIES = frozenset({
    "employer", "city", "school", "religion", "health", "sexual_orientation",
    "caste", "politics", "immigration", "finances", "hobby", "sector", "region",
})


@dataclass(frozen=True)
class Fact:
    fact_id: str
    category: str                                # value from SUPPORTED_FACT_CATEGORIES
    value: str                                   # exact private value
    blind_form: str | None = None                # coarse form allowed at BLIND; None = nothing
    mutual_form: str | None = None               # form allowed after mutual yes; None = never
    blind_attr: tuple[str, str] | None = None    # attribute revealed by blind_form (anonymity accounting)
    deny_descriptors: tuple[str, ...] = ()       # paraphrases the compile step predicted would identify `value`
    source: str = "self_report"                  # provenance
    value_specificity: float = 1.0               # 0 = ambiguous lexical value; 1 = distinctive value

    def __post_init__(self) -> None:
        if not 0.0 <= self.value_specificity <= 1.0:
            raise ValueError("value_specificity must be between 0.0 and 1.0")


@dataclass
class DisclosureCard:
    user_id: str
    display_name: str
    facts: list[Fact] = field(default_factory=list)

    def speakable(self, stage: Stage) -> list[str]:
        out = []
        for f in self.facts:
            form = f.blind_form if stage == Stage.BLIND else f.mutual_form
            if form:
                out.append(form)
        return out

    def forbidden_terms(self, stage: Stage) -> list[tuple[str, str]]:
        """(category, term) pairs that must not appear in speech at this stage:
        the exact value plus the compile step's predicted identifying paraphrases."""
        terms: list[tuple[str, str]] = []
        for f in self.facts:
            allowed = {f.blind_form} if stage == Stage.BLIND else {f.mutual_form}
            candidates = (f.value, f.blind_form, f.mutual_form, *f.deny_descriptors)
            terms.extend(
                (f.category, term)
                for term in dict.fromkeys(candidates)
                if term and term not in allowed
            )
        return terms


@dataclass
class LogEvent:
    """One logged recommendation. `propensity` = probability the logging policy showed
    this target to this viewer; without it, offline evaluation of new rankers is biased."""
    viewer: int
    target: int
    propensity: float
    policy: str
    yes_a: bool
    yes_b: bool | None
    good: bool | None
