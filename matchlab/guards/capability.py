"""Design-time capability audit (Meta's "Agents Rule of Two") plus a heuristic scanner.

The audit is the real defence. An agent session should combine at most two of:
  A  processes untrustworthy input
  B  has access to sensitive / private data
  C  can change state or communicate externally
If all three hold, one successful injection can read private data and send it out.

The scanner is NOT a defence. Attackers can paraphrase around any pattern list, so it
exists only to flag and log suspicious text for review and to measure how much a
pattern list would catch. Never rely on it to make a session safe.
"""
from __future__ import annotations

import re
from dataclasses import dataclass


@dataclass(frozen=True)
class SessionCapabilities:
    name: str
    untrusted_input: bool
    private_data: bool
    external_effect: bool


def rule_of_two(s: SessionCapabilities) -> dict:
    held = [n for n, v in (("untrusted_input", s.untrusted_input),
                           ("private_data", s.private_data),
                           ("external_effect", s.external_effect)) if v]
    return {"session": s.name, "properties_held": held, "passes": len(held) <= 2}


INJECTION_PATTERNS = [
    r"ignore (all |your |the |any )?(previous |prior |above )?(instructions|rules)",
    r"disregard (all |your |the )?(previous |prior )?(instructions|rules)",
    r"you are (now )?in (debug|developer|admin) mode",
    r"(print|reveal|show|dump|output) (the |your )?(full |entire |complete )?(profile|system prompt|instructions)",
    r"^\s*(system|assistant)\s*:",
    r"\bwhen asked about\b.*\breveal\b",
    r"(pichle|purane) nirdesh (ignore|bhool)",
]
_RX = [re.compile(p, re.I | re.M) for p in INJECTION_PATTERNS]


def scan_untrusted(text: str) -> list[str]:
    return [p for p, rx in zip(INJECTION_PATTERNS, _RX) if rx.search(text)]
