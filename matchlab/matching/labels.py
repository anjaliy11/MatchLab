"""Reliability of the 'good conversation' label.

Both people rate every call, so inter-rater agreement is available for free. A model
cannot be validated beyond the reliability of its label: if the two raters barely
agree, a low model score may just reflect label noise.
"""
from __future__ import annotations

import numpy as np


def agreement(a, b) -> float:
    return float(np.mean(np.asarray(a, bool) == np.asarray(b, bool)))


def cohens_kappa(a, b) -> float:
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    po = np.mean(a == b)
    pe = a.mean() * b.mean() + (1 - a.mean()) * (1 - b.mean())
    return float((po - pe) / (1 - pe)) if pe < 1 else float("nan")


def krippendorff_alpha_nominal(a, b) -> float:
    """Krippendorff's alpha for two raters, binary labels, no missing data."""
    a, b = np.asarray(a, int), np.asarray(b, int)
    vals = np.concatenate([a, b])
    n = len(vals)
    p1 = vals.mean()
    do = np.mean(a != b)                                 # observed disagreement
    de = 2 * p1 * (1 - p1) * n / (n - 1)                 # expected disagreement
    return float(1 - do / de) if de > 0 else float("nan")


def max_observable_corr(reliability: float) -> float:
    """Upper bound on the correlation any predictor can show against a noisy label
    (classical attenuation: sqrt of label reliability)."""
    return float(np.sqrt(max(0.0, reliability)))
