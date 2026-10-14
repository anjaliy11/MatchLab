"""Disclosure budget: stop a blind pitch from narrowing a person to a handful of people.

Each blind-safe fact reveals an attribute (e.g. sector=tech, region=south). Individually
harmless; combined they can identify someone. We track, per (viewer, target), the size
of the anonymity set: how many people in the pool match everything revealed so far.
A reply that would push it below k_min is blocked.

A second, per-viewer cap on total bits released across ALL targets limits the rate at
which one account can harvest identifying detail (the dating-app analogue of an
oracle attack: many cheap queries, each leaking a little).
"""
from __future__ import annotations

import math
from collections import Counter, defaultdict


class AnonymityPool:
    """Population used for anonymity accounting. Records are pre-aggregated into distinct
    attribute combinations, so a size query costs O(#combinations), not O(#people);
    the first version scanned every record and took ~20 ms per check, far too slow
    for an inline guard."""

    def __init__(self, records: list[dict[str, str]]):
        self.records = records
        self.n = len(records)
        self._combos = [(dict(k), c) for k, c in Counter(tuple(sorted(r.items())) for r in records).items()]

    def size(self, constraints: dict[str, str]) -> int:
        if not constraints:
            return self.n
        items = constraints.items()
        return sum(c for d, c in self._combos if all(d.get(k) == v for k, v in items))

    def bits(self, constraints: dict[str, str]) -> float:
        return math.log2(self.n / max(1, self.size(constraints))) if constraints else 0.0


class DisclosureBudget:
    def __init__(self, pool: AnonymityPool, k_min: int = 20, viewer_bits_cap: float = 40.0):
        self.pool, self.k_min, self.cap = pool, k_min, viewer_bits_cap
        self.revealed: dict[tuple[str, str], dict[str, str]] = defaultdict(dict)
        self.viewer_bits: dict[str, float] = defaultdict(float)

    def check(self, viewer: str, target: str, new_attrs: dict[str, str]) -> tuple[bool, str]:
        merged = {**self.revealed[(viewer, target)], **new_attrs}
        if self.pool.size(merged) < self.k_min:
            return False, f"anonymity_set<{self.k_min}"
        gain = self.pool.bits(merged) - self.pool.bits(self.revealed[(viewer, target)])
        if self.viewer_bits[viewer] + gain > self.cap:
            return False, "viewer_bits_cap"
        return True, "ok"

    def commit(self, viewer: str, target: str, new_attrs: dict[str, str]) -> None:
        before = self.pool.bits(self.revealed[(viewer, target)])
        self.revealed[(viewer, target)].update(new_attrs)
        self.viewer_bits[viewer] += self.pool.bits(self.revealed[(viewer, target)]) - before

    def anonymity_size(self, viewer: str, target: str) -> int:
        return self.pool.size(self.revealed[(viewer, target)])
