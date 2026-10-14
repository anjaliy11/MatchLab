"""Matching-side evaluation tools, exercised on synthetic logs.

    python scripts/run_matching_evals.py --seeds 10
"""
import argparse
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matchlab.matching import calibration as cal
from matchlab.matching import fairness as fair
from matchlab.matching import labels as lab
from matchlab.matching import ope
from matchlab.matching.cost import UnitCosts, cost_per_good_call, sensitivity
from matchlab.sim.sim import simulate
from matchlab.sim.world import World, sigmoid


def softmax_rows(x):
    z = np.exp(x - x.max(axis=1, keepdims=True))
    return z / z.sum(axis=1, keepdims=True)


# ---------------------------------------------------------------- 1. OPE
def ope_trial(seed, n_logs=30000, eps=0.1, beta=1.5, gamma=0.5, tau=2.0):
    w = World(seed)
    sd = np.sqrt(w.d)
    stated = (w.SA @ w.XB.T + w.XA @ w.SB.T) / sd
    reward = w.PA * w.PB * w.Q                                   # true P(good call) per pitch
    pi0 = softmax_rows(beta * w.pB[None, :] + gamma * stated)    # old ranker: popularity + stated fit
    pi0 = (1 - eps) * pi0 + eps / w.nB                           # epsilon floor keeps weights bounded
    targets = {
        "uniform": np.full_like(pi0, 1.0 / w.nB),
        "stated_ranker": softmax_rows(tau * stated),
        "better_ranker": softmax_rows(tau * 0.5 * (w.fitAB + w.fitBA)),   # hypothetical stronger ranker
    }
    rng = np.random.default_rng([seed, 5])
    i = rng.integers(0, w.nA, n_logs)
    j = (np.cumsum(pi0, axis=1)[i] < rng.random(n_logs)[:, None]).sum(axis=1).clip(0, w.nB - 1)
    r = (rng.random(n_logs) < reward[i, j]).astype(float)
    prior = 5.0
    q_t = (np.bincount(j, r, w.nB) + prior * r.mean()) / (np.bincount(j, minlength=w.nB) + prior)
    out = {}
    for name, pi1 in targets.items():
        wt = pi1[i, j] / pi0[i, j]
        truth = float((pi1 * reward).sum(axis=1).mean())
        q_exp = (pi1 * q_t[None, :]).sum(axis=1)[i]
        out[name] = {"truth": truth, "replay": float(r.mean()), "ips": ope.ips(r, wt),
                     "snips": ope.snips(r, wt), "clipped": ope.clipped_snips(r, wt, 10.0),
                     "dr": ope.dr(r, wt, q_t[j], q_exp), "ess": ope.ess_fraction(wt)}
    return out


def section_ope(seeds):
    print("\n== 1. Off-policy evaluation: estimate a NEW ranker's good-calls-per-pitch from logs of the OLD one")
    print("   (relative error vs ground truth, mean over seeds; replay = naive average of logged outcomes)")
    trials = [ope_trial(s) for s in range(seeds)]
    print(f"   {'target':<15}{'truth':>8}{'replay':>9}{'IPS':>8}{'SNIPS':>8}{'clip10':>8}{'DR':>8}{'ESS%':>7}")
    for name in trials[0]:
        tr = np.mean([t[name]["truth"] for t in trials])
        row = f"   {name:<15}{tr:>8.4f}"
        for k in ("replay", "ips", "snips", "clipped", "dr"):
            err = np.mean([(t[name][k] - t[name]["truth"]) / t[name]["truth"] for t in trials])
            row += f"{100 * err:>+8.1f}%" if k == "replay" else f"{100 * err:>+7.1f}%"
        row += f"{100 * np.mean([t[name]['ess'] for t in trials]):>6.0f}%"
        print(row)
    print("   RMSE of relative error across seeds:")
    for k in ("replay", "ips", "snips", "clipped", "dr"):
        e = [np.sqrt(np.mean([((t[n][k] - t[n]["truth"]) / t[n]["truth"]) ** 2 for t in trials])) for n in trials[0]]
        print(f"     {k:<8}" + "".join(f"{100 * x:>9.1f}%" for x in e))


# ---------------------------------------------------------------- 2. label reliability
def section_labels():
    print("\n== 2. Reliability of the 'good conversation' label (two raters per call)")
    w = World(0)
    rng = np.random.default_rng(1)
    q = w.Q.ravel()
    true = rng.random(20000) < rng.choice(q, 20000)
    print(f"   {'rater flip prob':>16}{'agreement':>11}{'kappa':>8}{'alpha':>8}{'max corr a model can show':>28}")
    for e in (0.05, 0.15, 0.30):
        a = np.where(rng.random(len(true)) < e, ~true, true)
        b = np.where(rng.random(len(true)) < e, ~true, true)
        k = lab.cohens_kappa(a, b)
        print(f"   {e:>16.2f}{lab.agreement(a, b):>11.3f}{k:>8.3f}{lab.krippendorff_alpha_nominal(a, b):>8.3f}"
              f"{lab.max_observable_corr(k):>28.3f}")


# ---------------------------------------------------------------- 3. calibration
def section_calibration():
    print("\n== 3. Calibration: an over-confident yes-model, before and after temperature scaling")
    w = World(0)
    rng = np.random.default_rng(2)
    p_true = w.PA.ravel()
    y = rng.random(len(p_true)) < p_true
    z = np.log(p_true / (1 - p_true))
    p_pred = sigmoid(1.8 * z)                                    # injected over-confidence
    h = len(y) // 2
    T = cal.fit_temperature(p_pred[:h], y[:h])
    p_fix = cal.apply_temperature(p_pred[h:], T)
    print(f"   fitted temperature T = {T:.2f} (true miscalibration factor 1.8)")
    print(f"   held-out ECE   before {cal.ece(p_pred[h:], y[h:]):.4f}   after {cal.ece(p_fix, y[h:]):.4f}")
    print(f"   held-out Brier before {cal.brier(p_pred[h:], y[h:]):.4f}   after {cal.brier(p_fix, y[h:]):.4f}")


# ---------------------------------------------------------------- 4. fairness
def section_fairness(seeds):
    print("\n== 4. Fairness slices")
    print("   4a. exposure inequality of pitched B users (Gini; 0 = equal), same exposure cap for all")
    for pol in ("random", "popularity", "learned_gpc"):
        g = np.mean([simulate(World(s), pol, rounds=40, seed=s)["exposure_gini_B"] for s in range(min(seeds, 4))])
        print(f"      {pol:<14}{g:.3f}")
    print("   4b. voice-feature reliability by language group (SYNTHETIC noise levels, an assumption, not a measurement)")
    rng = np.random.default_rng(3)
    n = 20000
    good = rng.random(n) < 0.4
    signal = good * 1.0 + rng.normal(0, 1, n)
    group = np.where(rng.random(n) < 0.5, "english", "hinglish")
    noise = np.where(group == "english", 0.4, 1.4)
    feature = signal + rng.normal(0, 1, n) * noise
    res = {g: fair.auc(feature[group == g], good[group == g]) for g in ("english", "hinglish")}
    print(f"      predictive AUC by group: {res}  gap = {res['english'] - res['hinglish']:.3f}")
    print(f"      parity check on assumed WER {{'english': 0.08, 'hinglish': 0.30}}: {fair.feature_parity({'english': 0.08, 'hinglish': 0.30})}")


# ---------------------------------------------------------------- 5. cost
def section_cost(seeds):
    print("\n== 5. Cost per good call (abstract cost units; plug in real prices)")
    base = UnitCosts(per_pitch_minute=1.0, pitch_minutes=1.0, gate_overhead_per_pitch=0.0,
                     judge_cost_per_call=3.0)
    print(f"   {'policy':<15}{'pitch deliveries':>18}{'good calls':>12}{'cost/good call':>16}")
    for pol in ("random", "stated_compat", "learned_gpc"):
        rs = [simulate(World(s), pol, rounds=40, seed=s) for s in range(min(seeds, 4))]
        pit = np.mean([r["total_pitch_deliveries"] for r in rs])
        gd = np.mean([r["total_good_calls"] for r in rs])
        print(f"   {pol:<15}{pit:>18.0f}{gd:>12.0f}{cost_per_good_call(pit, gd, base):>16.2f}")
    print("   async LLM-judge overhead on top (judge costs 3 pitch-minutes per audited pitch), learned_gpc:")
    for r, c in sensitivity(pit, gd, base).items():
        print(f"      audit sample rate {r:>5.0%}: {c:.2f} per good call")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=10)
    a = ap.parse_args()
    section_ope(a.seeds)
    section_labels()
    section_calibration()
    section_fairness(a.seeds)
    section_cost(a.seeds)


if __name__ == "__main__":
    main()
