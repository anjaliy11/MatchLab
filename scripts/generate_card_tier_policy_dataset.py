"""Generate the deterministic policy-labelled synthetic card-tier benchmark.

This checks the current stand-in against the approved rubric; it is not a
human-adjudicated compiler evaluation.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matchlab.evals.card_tier_policy_dataset import PolicyTierRow, build_policy_dataset


_FIELDS = tuple(PolicyTierRow.__annotations__)
_ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument(
        "--csv",
        type=Path,
        default=_ROOT / "evals" / "data" / "card_tier_policy_seed.csv",
        help="output CSV path",
    )
    parser.add_argument(
        "--report",
        type=Path,
        default=_ROOT / "results" / "card_tier_policy_seed_report.json",
        help="aggregate report path",
    )
    args = parser.parse_args()

    rows, report = build_policy_dataset(seed=args.seed)
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", encoding="utf-8", newline="") as target:
        writer = csv.DictWriter(target, fieldnames=_FIELDS, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args.report.write_text(
        json.dumps(report, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {len(rows)} policy-labelled synthetic rows.")
    print(f"CSV: {args.csv}")
    print(f"Report: {args.report}")
    print("No human adjudication or compiler confidence scores are present.")


if __name__ == "__main__":
    main()
