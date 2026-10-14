"""Matching policies.

Observation tuple passed to update(): (i, j, yA, yB, good)
  yA   : bool, did A say yes to the pitch about B
  yB   : bool | None, did B say yes (None if A said no, B was never pitched)
  good : bool | None, was the call good (None if no match, so no call happened)

Policies only touch PUBLIC info (world.public()) and observations, except the
oracles, which read ground truth and exist only as upper bounds.
"""
from __future__ import annotations

import numpy as np

from .world import sigmoid


class Policy:
    name = "base"

    def scores(self, t: int) -> np.ndarray:  # (nA, nB), higher = pitch first
        raise NotImplementedError

    def update(self, obs: list) -> None:
        pass

    def diagnostics(self, world) -> dict:
        return {}


class RandomPolicy(Policy):
    name = "random"

    def __init__(self, pub, seed=0):
        self.pub, self.rng = pub, np.random.default_rng([seed, 1])

    def scores(self, t):
        return self.rng.random((self.pub.nA, self.pub.nB))


class StatedCompat(Policy):
    """Static profile compatibility: stated taste . stated profile, both ways.
    Roughly what a one-shot 'profile similarity' matcher does."""
    name = "stated_compat"

    def __init__(self, pub, seed=0):
        self.pub = pub

    def scores(self, t):
        p = self.pub
        return (p.SA @ p.XB.T + p.XA @ p.SB.T) / np.sqrt(p.d)


class Popularity(Policy):
    """Reciprocal popularity: observed yes-rates of both sides (Elo-like)."""
    name = "popularity"

    def __init__(self, pub, seed=0):
        self.pub, self.rng = pub, np.random.default_rng([seed, 2])
        self.yesB = np.zeros(pub.nB); self.pitB = np.zeros(pub.nB)
        self.yesA = np.zeros(pub.nA); self.pitA = np.zeros(pub.nA)

    def scores(self, t):
        popB = (self.yesB + 1) / (self.pitB + 2)
        popA = (self.yesA + 1) / (self.pitA + 2)
        return popA[:, None] * popB[None, :] + 1e-6 * self.rng.random((self.pub.nA, self.pub.nB))

    def update(self, obs):
        for i, j, yA, yB, good in obs:
            self.pitB[j] += 1; self.yesB[j] += yA
            if yB is not None:
                self.pitA[i] += 1; self.yesA[i] += yB


class Oracle(Policy):
    """Reads ground truth. Upper bound only."""

    def __init__(self, world, objective):
        self.w, self.objective = world, objective
        self.name = f"oracle_{objective}"

    def scores(self, t):
        w = self.w
        if self.objective == "match":
            return w.PA * w.PB
        return w.PA * w.PB * w.Q / (1.0 + w.PA)


class LearnedReciprocal(Policy):
    """Online reciprocal model with explicit effect decomposition.

    For rater u looking at target v (X = public stated profile of v):
        logit P(yes_uv) = b_u + c_v + T_u . X_v / sqrt(d)
          b_u : rater generosity        (not fit)
          c_v : target pitchability     (not fit)
          T_u : taste vector, prior-mean = eta * stated taste, shrunk toward it
                (stated taste is the cold-start prior; revealed behaviour overrides
                it as evidence grows)
    The T_u . X_v term is the dyadic fit estimate. Ranking uses it, not b or c,
    when objective == 'gpc'.

    Calls are only observed for matches, so a second small model maps estimated
    fit to P(good | match).

    Exploration: Thompson-style noise on T_u with std ts_sigma / sqrt(1 + n_u).

    objective:
        'match' : maximise P(A yes) * P(B yes)              (what swipe apps optimise)
        'gpc'   : maximise P(A yes) * P(B yes) * P(good|match) / (1 + P(A yes))
                  i.e. expected good calls per expected pitch cost; B is only
                  pitched if A says yes, so expected cost is 1 + P(A yes).
    """

    def __init__(self, pub, objective="gpc", seed=0, eta=0.5, lr=0.2, l2=0.05,
                 ts_sigma=0.8, passes=3):
        self.pub, self.objective = pub, objective
        self.name = f"learned_{objective}"
        self.rng = np.random.default_rng([seed, 3])
        self.sd = np.sqrt(pub.d)
        self.eta, self.lr, self.l2, self.ts_sigma, self.passes = eta, lr, l2, ts_sigma, passes
        self.mA = eta * pub.SA; self.mB = eta * pub.SB          # prior means from onboarding
        self.TA, self.TB = self.mA.copy(), self.mB.copy()
        self.bA = np.zeros(pub.nA); self.bB = np.zeros(pub.nB)
        self.cB = np.zeros(pub.nB)   # how pitchable B targets are, as seen by A raters
        self.cA = np.zeros(pub.nA)   # how pitchable A targets are, as seen by B raters
        self.nA = np.zeros(pub.nA); self.nB = np.zeros(pub.nB)
        self.q0, self.q1 = 0.0, 0.5

    # ---- scoring -------------------------------------------------------
    def _fits(self, TA, TB):
        fitA = TA @ self.pub.XB.T / self.sd          # (nA, nB): A's fit with B
        fitB = self.pub.XA @ TB.T / self.sd          # (nA, nB): B's fit with A
        return fitA, fitB

    def predict(self, explore=False):
        TA, TB = self.TA, self.TB
        if explore:
            TA = TA + self.rng.normal(size=TA.shape) * (self.ts_sigma / np.sqrt(1 + self.nA))[:, None]
            TB = TB + self.rng.normal(size=TB.shape) * (self.ts_sigma / np.sqrt(1 + self.nB))[:, None]
        fitA, fitB = self._fits(TA, TB)
        PA = sigmoid(self.bA[:, None] + self.cB[None, :] + fitA)
        PB = sigmoid(self.bB[None, :] + self.cA[:, None] + fitB)
        Q = sigmoid(self.q0 + self.q1 * 0.5 * (fitA + fitB))
        return PA, PB, Q, fitA, fitB

    def scores(self, t):
        PA, PB, Q, _, _ = self.predict(explore=True)
        if self.objective == "match":
            return PA * PB
        return PA * PB * Q / (1.0 + PA)

    # ---- learning ------------------------------------------------------
    def _step(self, T, b, c, m, n, u, v, X, y):
        z = b[u] + c[v] + T[u] @ X[v] / self.sd
        g = sigmoid(z) - y
        lr = self.lr / (1.0 + 0.02 * n[u])
        T[u] -= lr * (g * X[v] / self.sd + self.l2 * (T[u] - m[u]))
        b[u] -= lr * (g + 0.01 * b[u])
        c[v] -= lr * 0.5 * (g + 0.05 * c[v])

    def update(self, obs):
        p = self.pub
        for _ in range(self.passes):
            for i, j, yA, yB, good in obs:
                self._step(self.TA, self.bA, self.cB, self.mA, self.nA, i, j, p.XB, float(yA))
                if yB is not None:
                    self._step(self.TB, self.bB, self.cA, self.mB, self.nB, j, i, p.XA, float(yB))
        for i, j, yA, yB, good in obs:
            self.nA[i] += 1
            if yB is not None:
                self.nB[j] += 1
            if good is not None:
                f = 0.5 * (self.TA[i] @ p.XB[j] + p.XA[i] @ self.TB[j]) / self.sd
                err = sigmoid(self.q0 + self.q1 * f) - float(good)
                self.q0 -= 0.05 * err
                self.q1 -= 0.05 * err * f

    # ---- evals -----------------------------------------------------------
    def state_bytes_per_user(self):
        d = self.pub.d
        return (d + 3) * 8  # taste vector + bias + pitchability + count, float64

    def diagnostics(self, world):
        fitA, _ = self._fits(self.TA, self.TB)
        corr_fit = float(np.corrcoef(fitA.ravel(), world.fitAB.ravel())[0, 1])
        corr_pitch = float(np.corrcoef(self.cB, world.pB)[0, 1])
        PA, _, _, _, _ = self.predict(explore=False)
        auc = _auc(PA.ravel(), world.PA.ravel() > np.random.default_rng(0).random(world.PA.size))
        return {"fit_recovery_corr": corr_fit, "pitchability_recovery_corr": corr_pitch,
                "yes_auc": auc, "state_bytes_per_user": self.state_bytes_per_user()}


def _auc(scores, labels):
    labels = np.asarray(labels, bool)
    order = np.argsort(scores)
    ranks = np.empty(len(scores)); ranks[order] = np.arange(1, len(scores) + 1)
    n1, n0 = labels.sum(), (~labels).sum()
    return float((ranks[labels].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def make_policy(name, world, seed=0):
    pub = world.public()
    if name == "random":
        return RandomPolicy(pub, seed)
    if name == "stated_compat":
        return StatedCompat(pub, seed)
    if name == "popularity":
        return Popularity(pub, seed)
    if name == "learned_match":
        return LearnedReciprocal(pub, "match", seed)
    if name == "learned_gpc":
        return LearnedReciprocal(pub, "gpc", seed)
    if name == "oracle_match":
        return Oracle(world, "match")
    if name == "oracle_gpc":
        return Oracle(world, "gpc")
    raise ValueError(name)
