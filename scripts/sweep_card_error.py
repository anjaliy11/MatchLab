"""How much does the card compile step matter? Sweep its mis-tier rate.

In this design the compiled card is the trusted computing base: both the live agent and the
gate believe it, so a private value mis-tiered as blind-safe is invisible to both.

    python scripts/sweep_card_error.py --personas 30 --trials 3
"""
import argparse
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matchlab.evals.personas import make_personas, make_pool
from matchlab.evals.runner import ARMS, leak_rate, pass_stats, run_suite
from matchlab.evals.tasks import build_tasks

ap = argparse.ArgumentParser()
ap.add_argument("--personas", type=int, default=30)
ap.add_argument("--trials", type=int, default=3)
a = ap.parse_args()
tasks, pool = build_tasks(make_personas(a.personas, seed=0)), make_pool(20000, seed=0)
print(f"{'card error':>10} " + "".join(f"{x[:19]:>22}" for x in ARMS) + "   (attack-task leak rate)")
for e in (0.0, 0.05, 0.2):
    R = run_suite(tasks, pool, trials=a.trials, card_error_rate=e)["results"]
    print(f"{e:>10} " + "".join(f"{leak_rate(R, arm):>22.3f}" for arm in ARMS))
