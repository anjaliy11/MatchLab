"""Cost per good call.

No vendor prices are built in: they change, and an unverified number in a design doc
reads as a fact. Pass your own `unit_costs`. The structure is the useful part:

  cost per pitch   = pitch minutes x (speech + model cost per minute) [+ gate overhead]
  audit overhead   = judge_sample_rate x judge_cost_per_call          (async LLM judge)
  cost per good call = (pitches x cost per pitch + audits) / good calls
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnitCosts:
    per_pitch_minute: float            # speech-to-text + model + text-to-speech, per minute
    pitch_minutes: float = 1.0
    gate_overhead_per_pitch: float = 0.0
    judge_cost_per_call: float = 0.0   # async LLM judge, per audited pitch
    judge_sample_rate: float = 0.0


def cost_per_good_call(pitches: float, good_calls: float, u: UnitCosts) -> float:
    per_pitch = u.pitch_minutes * u.per_pitch_minute + u.gate_overhead_per_pitch
    audits = pitches * u.judge_sample_rate * u.judge_cost_per_call
    return (pitches * per_pitch + audits) / max(good_calls, 1e-9)


def sensitivity(pitches: float, good_calls: float, base: UnitCosts, sample_rates=(0.0, 0.05, 0.25, 1.0)) -> dict:
    out = {}
    for r in sample_rates:
        u = UnitCosts(base.per_pitch_minute, base.pitch_minutes, base.gate_overhead_per_pitch,
                      base.judge_cost_per_call, r)
        out[r] = cost_per_good_call(pitches, good_calls, u)
    return out
