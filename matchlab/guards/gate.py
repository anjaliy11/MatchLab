"""Inline sentence gate: runs on each complete sentence before it reaches text-to-speech.

Deliberately deterministic and cheap (no model call) so it fits an inline latency
budget; semantic judging belongs in an async audit, not here. Checks:

  leak          a restricted value or predicted paraphrase appears (fuzzy, transliteration-tolerant)
  confirmation  the viewer's question is about a restricted category or names a
                restricted value, and the reply is a bare yes/no. Both answers leak.
  number        a number appears that is not in any allowed phrase (cheap hallucination check)
  anonymity     the reply would shrink the viewer's anonymity set for this target below k.
                Attributes come from the agent's structured `declared_attrs` when it provides
                them (reliable), else are inferred by string match (brittle: paraphrase evades it)
  entailment    optional pluggable semantic check (an NLI model); off by default
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Callable

from ..schemas import DisclosureCard, Stage
from ..text import fuzzy_in
from .budget import DisclosureBudget

AFFIRM = {"yes", "yeah", "yep", "correct", "right", "exactly", "haan", "sahi", "bilkul"}
NEGATE = {"no", "nope", "nahi", "nahin", "galat"}
BINARY_ANSWER_RE = re.compile(
    r"\s*(?:yes|yeah|yep|correct|right|exactly|haan|sahi|bilkul|no|nope|nahi|nahin|galat|"
    r"that's\s+right|that\s+is\s+right)"
    r"(?:\s*[,;:]?\s*(?:(?:he|she|they)\s+(?:does(?:n't| not)|is(?:n't| not)|has(?:n't| not)|"
    r"do(?:n't| not)|are(?:n't| not))|(?:woh|vah)\s+(?:nahi\s+)?(?:hai|hain|karta|karti|karte)|"
    r"(?:that's|that\s+is)\s+(?:right|correct)|definitely|indeed|of course|galat|"
    r"bilkul(?:\s+sahi)?|sahi(?:\s+bilkul)?))?\s*[.!?]*\s*", re.I)
RESTRICTED_CATEGORY_CUES = {
    "employer": re.compile(r"\b(?:work|works|working|job|company|employer|office|kaam|naukri)\b", re.I),
    "city": re.compile(r"\b(?:city|town|hometown|location|live|lives|stay|stays|shehar|shahar|rehta|rehti)\b", re.I),
    "school": re.compile(r"\b(?:school|college|university|studied|study|education|padhai)\b", re.I),
    "religion": re.compile(r"\b(?:religion|religious|dharm|faith|mazhab)\b", re.I),
    "health": re.compile(r"\b(?:health|healthy|medical|condition|illness|disease|diagnosis|sehat|bimari)\b", re.I),
    "sexual_orientation": re.compile(
        r"\b(?:sexual[\s-]+orientation|orientation|sexuality|gay|lesbian|bisexual|queer|pansexual|"
        r"asexual|heterosexual|(?:is|was|are)\s+(?:he|she|they)\s+straight)\b", re.I),
    "caste": re.compile(r"\b(?:caste|jati|jaati|varna)\b", re.I),
    "politics": re.compile(
        r"\b(?:politics|political|party affiliation|political party|vote|raajneeti|rajniti)\b", re.I),
    "immigration": re.compile(
        r"\b(?:immigration|immigrant|visa|citizenship|citizen|passport|nagrikta)\b", re.I),
    "finances": re.compile(
        r"\b(?:finances|financial|income|salary|debt|loan|wealth|earnings|paisa|kamai)\b", re.I),
}
DEFLECTIONS = (
    "That's something they'll share themselves once you both match.",
    "I'll leave that one for the two of you to talk about.",
    "I'd rather not say; it's theirs to share.",
)
LOW_SPECIFICITY_THRESHOLD = 0.5


@dataclass
class GateResult:
    action: str                       # "pass" | "rewrite"
    reasons: list[str] = field(default_factory=list)
    text: str = ""


class SentenceGate:
    def __init__(self, card: DisclosureCard, stage: Stage = Stage.BLIND,
                 budget: DisclosureBudget | None = None, viewer_id: str = "viewer",
                 entailer: Callable[[str, list[str]], bool] | None = None):
        self.card, self.stage, self.budget, self.viewer_id = card, stage, budget, viewer_id
        self.entailer = entailer
        self._forbidden = card.forbidden_terms(stage)
        self._low_specificity_terms = {
            (fact.category, fact.value)
            for fact in card.facts
            if fact.value_specificity < LOW_SPECIFICITY_THRESHOLD
        }
        self._allowed_text = " ".join(card.speakable(stage))

    def _blind_attrs_in(self, sentence: str) -> dict[str, str]:
        attrs = {}
        for f in self.card.facts:
            if f.blind_form and f.blind_attr and fuzzy_in(sentence, f.blind_form):
                attrs[f.blind_attr[0]] = f.blind_attr[1]
        return attrs

    def _matches_restricted_value(self, text: str, category: str, term: str) -> bool:
        if not fuzzy_in(text, term):
            return False
        if (category, term) not in self._low_specificity_terms:
            return True
        category_cue = RESTRICTED_CATEGORY_CUES.get(category)
        return category_cue is None or bool(category_cue.search(text))

    def check(self, sentence: str, question: str | None = None,
              declared_attrs: dict[str, str] | None = None) -> GateResult:
        reasons: list[str] = []
        for cat, term in self._forbidden:
            if self._matches_restricted_value(sentence, cat, term):
                reasons.append(f"leak:{cat}")
        if question:
            value_asked = {cat for cat, term in self._forbidden
                           if self._matches_restricted_value(question, cat, term)}
            restricted_categories = {cat for cat, _ in self._forbidden}
            category_asked = {
                cat for cat in restricted_categories
                if cat in RESTRICTED_CATEGORY_CUES
                and RESTRICTED_CATEGORY_CUES[cat].search(question)
            }
            category_binary_answer = bool(BINARY_ANSWER_RE.fullmatch(sentence))
            words = set(re.findall(r"[a-z]+", sentence.casefold()))
            if value_asked and words & (AFFIRM | NEGATE):
                reasons.extend(f"confirmation:{c}" for c in sorted(value_asked))
            if category_asked and category_binary_answer:
                reasons.extend(f"confirmation:{c}" for c in sorted(category_asked))
        allowed_nums = set(re.findall(r"\d+", self._allowed_text))
        if set(re.findall(r"\d+", sentence)) - allowed_nums:
            reasons.append("unsupported_number")
        attrs = {**self._blind_attrs_in(sentence), **(declared_attrs or {})} if self.budget else {}
        if self.budget and attrs:
            ok, why = self.budget.check(self.viewer_id, self.card.user_id, attrs)
            if not ok:
                reasons.append(f"budget:{why}")
        if self.entailer and not reasons and not self.entailer(sentence, self.card.speakable(self.stage)):
            reasons.append("unsupported_claim")
        if reasons:
            return GateResult("rewrite", sorted(set(reasons)), DEFLECTIONS[len(sentence) % len(DEFLECTIONS)])
        if self.budget and attrs:
            self.budget.commit(self.viewer_id, self.card.user_id, attrs)
        return GateResult("pass", [], sentence)
