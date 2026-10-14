"""Synthetic two-sided matching market with hidden ground truth.

Everything here is synthetic. No real user data, no scraping.

Model (all hidden from the policies unless stated):
  - Each user has a latent "who I am" vector z and a latent "what I like" vector t.
  - P(A says yes to B) = sigmoid(a_A + p_B + kappa * fit_AB), where
        a_A  = how generous A is with yeses          (rater effect)
        p_B  = how well B's pitch lands / popularity (target effect, NOT fit)
        fit  = t_A . z_B / sqrt(d)                   (true taste fit)
  - Flow is sequential, like a pitch-first product: A is pitched B first; B is
    pitched A only if A said yes. A match needs both yes.
  - P(good call | match) = sigmoid(g0 + g1 * mean(fit_AB, fit_BA)).
    Pitchability p does NOT help call quality: persuasion gets you a yes,
    fit gets you a good conversation.
  - The system only sees PUBLIC info: noisy stated profile X ~ z + noise, and a
    stated taste S that is a blend of the true taste and a shared "everyone says
    they want the same type" stereotype (stated != revealed preference).
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -30, 30)))


class World:
    def __init__(self, seed: int, nA: int = 150, nB: int = 250, d: int = 8,
                 kappa: float = 1.5, rho: float = 0.4, sigma_s: float = 0.7,
                 g0: float = -0.3, g1: float = 1.5):
        rng = np.random.default_rng(seed)
        self.seed, self.nA, self.nB, self.d = seed, nA, nB, d
        sd = np.sqrt(d)

        # hidden ground truth
        self.ZA = rng.normal(size=(nA, d))
        self.ZB = rng.normal(size=(nB, d))
        self.TA = rng.normal(size=(nA, d))
        self.TB = rng.normal(size=(nB, d))
        self.aA = rng.normal(0.0, 0.5, nA)
        self.aB = rng.normal(-0.3, 0.5, nB)
        self.pA = rng.normal(0.0, 0.7, nA)
        self.pB = rng.normal(0.0, 0.7, nB)

        # public / stated information
        self.XA = self.ZA + rng.normal(0, sigma_s, (nA, d))
        self.XB = self.ZB + rng.normal(0, sigma_s, (nB, d))
        stereotype = rng.normal(size=d)
        stereotype = stereotype / np.linalg.norm(stereotype) * sd
        self.SA = rho * self.TA + (1 - rho) * stereotype + rng.normal(0, 0.3, (nA, d))
        self.SB = rho * self.TB + (1 - rho) * stereotype + rng.normal(0, 0.3, (nB, d))

        # ground-truth probability matrices, shape (nA, nB)
        self.fitAB = self.TA @ self.ZB.T / sd
        self.fitBA = self.ZA @ self.TB.T / sd
        self.PA = sigmoid(self.aA[:, None] + self.pB[None, :] + kappa * self.fitAB)
        self.PB = sigmoid(self.aB[None, :] + self.pA[:, None] + kappa * self.fitBA)
        self.Q = sigmoid(g0 + g1 * 0.5 * (self.fitAB + self.fitBA))

    def public(self) -> SimpleNamespace:
        """What a production system could legitimately see at cold start."""
        return SimpleNamespace(nA=self.nA, nB=self.nB, d=self.d,
                               XA=self.XA, XB=self.XB, SA=self.SA, SB=self.SB)

    def draws(self, t: int):
        """Common random numbers: same uniform draws for every policy at round t,
        so policy comparisons are paired and low-variance."""
        r = np.random.default_rng([self.seed, 7919, t])
        shape = (self.nA, self.nB)
        return r.random(shape), r.random(shape), r.random(shape)
