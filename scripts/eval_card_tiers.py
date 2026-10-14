"""Score compiled disclosure tiers against a two-rater adjudicated CSV.

The CSV contains only IDs, categories, tier labels, confidences, and rater
annotations; do not put profile text or restricted values in it. The output is
aggregate-only and omits item IDs.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matchlab.evals.card_tier_eval import evaluate_tiers, load_tier_cases


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", required=True, help="two-rater/adjudicated CSV")
    parser.add_argument("--false-safe-cost", required=True, type=float)
    parser.add_argument("--overrestriction-cost", required=True, type=float)
    parser.add_argument("--minimum-confidence", required=True, type=float)
    parser.add_argument("--out", help="optional path for aggregate JSON output")
    args = parser.parse_args()

    try:
        cases = load_tier_cases(args.labels)
        report = evaluate_tiers(
            cases,
            false_safe_cost=args.false_safe_cost,
            overrestriction_cost=args.overrestriction_cost,
            minimum_confidence=args.minimum_confidence,
        )
        rendered = json.dumps(report, indent=2, sort_keys=True, allow_nan=False)
        if args.out:
            Path(args.out).write_text(rendered + "\n", encoding="utf-8")
        else:
            print(rendered)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
