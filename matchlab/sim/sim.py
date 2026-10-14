from __future__ import annotations

import numpy as np

from ..matching.fairness import exposure_gini
from .policies import make_policy
from .world import World


def assign(S, K, cap, rng):
    """Give each A user up to K pitches; each B user can be pitched at most `cap`
    times per round (exposure cap). A users are served in random order."""
    nA, nB = S.shape
    capacity = np.full(nB, cap)
    top = np.argsort(-S, axis=1)[:, :40]
    pairs = []
    for i in rng.permutation(nA):
        k = 0
        for j in top[i]:
            if not np.isfinite(S[i, j]):
                break
            if capacity[j] > 0:
                capacity[j] -= 1
                pairs.append((int(i), int(j)))
                k += 1
                if k == K:
                    break
    return pairs


def simulate(world: World, policy_name: str, rounds=40, K=3, cap=3, seed=0):
    policy = make_policy(policy_name, world, seed)
    nA, nB = world.nA, world.nB
    seen = np.zeros((nA, nB), bool)
    rng_assign = np.random.default_rng([world.seed, 99])
    cost = pitches_pairs = matches = goods = 0
    goodB = np.zeros(nB); matchB = np.zeros(nB); pitchB = np.zeros(nB)
    sum_pB = 0.0
    sum_fit_matched = 0.0
    goods_by_round = []

    for t in range(rounds):
        S = np.where(seen, -np.inf, policy.scores(t))
        pairs = assign(S, K, cap, rng_assign)
        uA, uB, uQ = world.draws(t)
        obs = []
        g_round = 0
        for i, j in pairs:
            seen[i, j] = True
            pitches_pairs += 1
            cost += 1                      # pitch delivered to A
            pitchB[j] += 1
            sum_pB += world.pB[j]
            yA = bool(uA[i, j] < world.PA[i, j])
            yB = good = None
            if yA:
                cost += 1                  # pitch delivered to B
                yB = bool(uB[i, j] < world.PB[i, j])
                if yB:
                    matches += 1; matchB[j] += 1
                    sum_fit_matched += 0.5 * (world.fitAB[i, j] + world.fitBA[i, j])
                    good = bool(uQ[i, j] < world.Q[i, j])
                    if good:
                        goods += 1; goodB[j] += 1; g_round += 1
            obs.append((i, j, yA, yB, good))
        policy.update(obs)
        goods_by_round.append(g_round)

    last = max(1, rounds // 4)
    out = {
        "good_per_100_pitches": 100.0 * goods / max(cost, 1),
        "match_rate_per_pair": matches / max(pitches_pairs, 1),
        "good_given_match": goods / max(matches, 1),
        "pitches_per_good_call": cost / max(goods, 1),
        "B_users_with_good_call": float((goodB > 0).mean()),
        "B_users_with_match": float((matchB > 0).mean()),
        "mean_true_pitchability_of_pitched_B": sum_pB / max(pitches_pairs, 1),
        "mean_true_fit_of_matches": sum_fit_matched / max(matches, 1),
        "late_good_per_round": float(np.mean(goods_by_round[-last:])),
        "total_good_calls": goods,
        "total_pitch_deliveries": cost,
        "exposure_gini_B": exposure_gini(pitchB),
    }
    out.update(policy.diagnostics(world))
    return out
