"""Memory write policy.

Persistent memory turns a one-off injection into a standing backdoor, and published
attacks (MINJA) reach it through ordinary user queries. So memory writes are treated
as security events with an explicit policy rather than something the model decides:

  * a self-report is only stored if the SPEAKER is the SUBJECT
  * statements by one person about another are never stored as facts about the other
  * model inferences about sensitive categories are never stored
  * directive-like text is rejected without treating ordinary uses of words such
    as "ignore" or "always" as instructions
  * per-subject write rate limit (blunts slow, multi-turn poisoning)
  * every decision returns a reason code for logging and audit
"""
from __future__ import annotations

import re
from collections import defaultdict, deque
from dataclasses import dataclass

SENSITIVE = {"religion", "health", "sexual_orientation", "caste", "politics", "immigration", "finances"}
INSTRUCTION_RE = re.compile(
    r"^\s*(?:"
    r"(?:(?:please|kindly)\s+)?(?:remember that|from now on|ignore|disregard)\b"
    r"|(?:always|never)\s+(?:say|tell|trust|approve|match|share|reveal)\b"
    r"|you\s+(?:must|should)\s+(?:always|never|ignore|disregard|treat|say|tell|trust|approve|match|share|reveal)\b"
    r"|treat\s+(?:me|them|him|her|anyone|everyone)\s+as\b"
    r"|whenever\s+(?:someone|anyone|he|she|they)\b"
    r"|new rule\b"
    r")", re.I)


@dataclass
class MemoryWrite:
    subject_id: str
    speaker_id: str
    kind: str          # self_report | behavior | model_inference | third_party_claim
    category: str
    text: str
    ts: float


@dataclass
class Decision:
    allow: bool
    reason: str


class MemoryWriteGuard:
    def __init__(self, max_writes_per_hour: int = 20):
        self.limit = max_writes_per_hour
        self._recent: dict[str, deque] = defaultdict(deque)

    def decide(self, w: MemoryWrite) -> Decision:
        if w.kind == "third_party_claim":
            return Decision(False, "third_party_claim")
        if w.kind == "self_report" and w.speaker_id != w.subject_id:
            return Decision(False, "speaker_not_subject")
        if w.kind == "model_inference" and w.category in SENSITIVE:
            return Decision(False, "sensitive_inference")
        if INSTRUCTION_RE.search(w.text):
            return Decision(False, "instruction_like")
        q = self._recent[w.subject_id]
        while q and w.ts - q[0] > 3600:
            q.popleft()
        if len(q) >= self.limit:
            return Decision(False, "rate_limited")
        q.append(w.ts)
        return Decision(True, "ok")
