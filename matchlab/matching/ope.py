"""Off-policy evaluation: estimate how a NEW ranker would have done from logs of the OLD one.

Needs the logging propensity of every shown item. Without it, only replay-style
estimates are possible, and those inherit the old ranker's selection bias.

  w_i = pi_new(a_i | x_i) / pi_log(a_i | x_i)

  IPS        mean(w * r)                      unbiased, high variance
  SNIPS      sum(w r) / sum(w)                biased slightly, much lower variance
  clipped    SNIPS with w capped at c         trades bias for variance
  DR         E_pi_new[q_hat] + mean(w (r - q_hat(a_i)))   unbiased if EITHER the
             propensities OR the reward model q_hat is right
"""
from __future__ import annotations

import numpy as np


def ips(r, w) -> float:
    return float(np.mean(np.asarray(r) * np.asarray(w)))


def snips(r, w) -> float:
    r, w = np.asarray(r, float), np.asarray(w, float)
    return float((w * r).sum() / w.sum())


def clipped_snips(r, w, c: float = 10.0) -> float:
    return snips(r, np.minimum(np.asarray(w, float), c))


def dr(r, w, q_logged, q_target_expected) -> float:
    """q_logged[i] = q_hat(x_i, a_i); q_target_expected[i] = sum_a pi_new(a|x_i) q_hat(x_i, a)."""
    r, w = np.asarray(r, float), np.asarray(w, float)
    return float(np.mean(np.asarray(q_target_expected) + w * (r - np.asarray(q_logged))))


def ess_fraction(w) -> float:
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum() / len(w))


def bootstrap_ci(estimator, *arrays, n: int = 300, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    m = len(arrays[0])
    vals = []
    for _ in range(n):
        idx = rng.integers(0, m, m)
        vals.append(estimator(*[np.asarray(a)[idx] for a in arrays]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))
