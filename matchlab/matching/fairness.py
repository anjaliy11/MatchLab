"""Fairness slices.

Voice-derived features are only as reliable as the speech recognition under them, and
recognition error differs by language and accent (code-mixed Hinglish is a documented
hard case). If feature reliability differs by group, model accuracy differs by group,
and so does who gets seen. Slice every metric by language group before shipping.
"""
from __future__ import annotations

import numpy as np


def exposure_gini(counts) -> float:
    x = np.sort(np.asarray(counts, float))
    n = len(x)
    if n == 0 or x.sum() == 0:
        return 0.0
    return float((2 * np.arange(1, n + 1) - n - 1).dot(x) / (n * x.sum()))


def group_gap(values, groups, a, b, n_boot: int = 400, seed: int = 0) -> dict:
    """mean(values | group a) - mean(values | group b) with a bootstrap 95% CI."""
    v, g = np.asarray(values, float), np.asarray(groups)
    va, vb = v[g == a], v[g == b]
    rng = np.random.default_rng(seed)
    diffs = [rng.choice(va, len(va)).mean() - rng.choice(vb, len(vb)).mean() for _ in range(n_boot)]
    return {"gap": float(va.mean() - vb.mean()), "ci95": (float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))),
            "n": (len(va), len(vb))}


def auc(scores, labels) -> float:
    s, y = np.asarray(scores, float), np.asarray(labels, bool)
    order = np.argsort(s)
    ranks = np.empty(len(s)); ranks[order] = np.arange(1, len(s) + 1)
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def feature_parity(wer_by_group: dict[str, float], max_ratio: float = 1.5, max_abs: float = 0.05) -> dict:
    """Flag when transcript error differs enough between groups that downstream features
    (style matching, grounding checks, leak detection) cannot be assumed equally reliable."""
    lo, hi = min(wer_by_group.values()), max(wer_by_group.values())
    ratio = hi / lo if lo > 0 else float("inf")
    return {"ratio": ratio, "abs_gap": hi - lo, "flag": bool(ratio > max_ratio or (hi - lo) > max_abs)}
