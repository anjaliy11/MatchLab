"""Text matching used by the guards.

Design choice: leak detection matches against the user's OWN private values (and
predicted paraphrases), not against an English NER model. NER degrades on
code-mixed Hinglish; matching known private strings after light normalisation does
not depend on the language of the surrounding sentence.

Known limits (tracked as capability evals, not hidden):
  * indirect descriptions that never use the value or a predicted paraphrase
  * non-Latin renderings of a Latin-script value (e.g. Devanagari)
"""
from __future__ import annotations

import re
import unicodedata
from functools import lru_cache
from difflib import SequenceMatcher

_MAP = (("ee", "i"), ("oo", "u"), ("ph", "f"), ("ck", "k"))


def normalize(s: str) -> str:
    s = unicodedata.normalize("NFKD", s).casefold()
    s = "".join(c for c in s if not unicodedata.combining(c))
    for a, b in _MAP:
        s = s.replace(a, b)
    s = s.replace("w", "v")
    return re.sub(r"(.)\1+", r"\1", s)          # collapse repeated letters


@lru_cache(maxsize=65536)
def _tok(s: str) -> tuple[str, ...]:
    return tuple(re.findall(r"[a-z0-9]+", normalize(s)))


def tokens(s: str) -> list[str]:
    return list(_tok(s))


def fuzzy_in(text: str, term: str, thresh: float = 0.84) -> bool:
    """True if `term` (possibly misspelled / re-spelled / split across words) appears in `text`.
    Short terms (<4 chars normalised) require an exact token match to limit false positives."""
    tt, xt = tokens(term), tokens(text)
    if not tt or not xt:
        return False
    term_s = "".join(tt)
    n = len(tt)
    if len(term_s) < 4:
        return any(xt[i:i + n] == tt for i in range(len(xt) - n + 1))
    for w in {max(1, n - 1), n, n + 1}:
        for i in range(len(xt) - w + 1):
            cand = "".join(xt[i:i + w])
            if abs(len(cand) - len(term_s)) > max(2, 0.25 * len(term_s)):
                continue
            if SequenceMatcher(None, cand, term_s).ratio() >= thresh:
                return True
    return False


def split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p for p in parts if p]
