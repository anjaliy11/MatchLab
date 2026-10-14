"""Calibration. A score that ranks well but is miscalibrated breaks any decision that
multiplies probabilities, such as P(A yes) x P(B yes) x P(good | match)."""
from __future__ import annotations

import numpy as np


def brier(p, y) -> float:
    return float(np.mean((np.asarray(p, float) - np.asarray(y, float)) ** 2))


def ece(p, y, bins: int = 10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges[1:-1]), 0, bins - 1)
    total = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            total += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(total)


def fit_temperature(p, y, grid=np.linspace(0.3, 3.0, 55)) -> float:
    """1-parameter recalibration: p' = sigmoid(logit(p) / T), chosen to minimise log loss."""
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    y = np.asarray(y, float)
    z = np.log(p / (1 - p))
    best, best_T = np.inf, 1.0
    for T in grid:
        q = 1 / (1 + np.exp(-z / T))
        q = np.clip(q, 1e-9, 1 - 1e-9)
        loss = -np.mean(y * np.log(q) + (1 - y) * np.log(1 - q))
        if loss < best:
            best, best_T = loss, T
    return float(best_T)


def apply_temperature(p, T: float):
    p = np.clip(np.asarray(p, float), 1e-6, 1 - 1e-6)
    return 1 / (1 + np.exp(-np.log(p / (1 - p)) / T))
