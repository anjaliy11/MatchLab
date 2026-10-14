# Decision 0006: Generate a policy-labelled synthetic tier baseline

Context: The evaluator required two independent rater labels, adjudication,
compiler predictions, and confidence. The approved category-by-stage policy
matrix existed, but no human annotations or compiler confidence interface
existed. A missing human label set cannot be repaired by inventing raters or
confidence values.

Class: eval.

Threat addressed: `THREAT_MODEL.md`, attack 14 (card mis-tier).

Eval first: Added tests that require one row per approved category, policy
labels at both stages, predictions only for categories emitted by the
stand-in compiler, explicit unknowns elsewhere, deterministic seeded output,
no private values, and declared non-human provenance. Expected result: the
five categories currently emitted by `compile_card` are scored; the other
eight stay unsupported and unscored.

Options:
- Fill the human-rater columns from the approved policy table: rejected
  because that would falsely claim independent human judgments.
- Import a public PII dataset: rejected because PII span labels do not define
  matchlab's stage-specific disclosure tiers.
- Add a separate synthetic policy-alignment benchmark: chosen; it can be
  useful without being mislabeled as human compiler evidence.

Choice and why: Generate a privacy-minimal CSV with policy-derived expected
tiers, predictions extracted from the current deterministic stand-in, blank
confidence columns, and explicit provenance. Produce an aggregate report
showing supported and unsupported categories and policy agreement.

Evidence: `python tests/test_core.py` tests deterministic generation, policy
labels, support boundaries, and absence of the synthetic persona's private
values. `python scripts/generate_card_tier_policy_dataset.py --seed 0`
regenerates the CSV and report. All outputs are synthetic and by construction.

Costs accepted: No model, network call, new dependency, live guard behavior,
or runtime latency is added. There is no independent compiler accuracy,
human-rater agreement, confidence calibration, or broad generalization
measurement.

Rollout and rollback: Keep the generated baseline separate from the
human-adjudicated evaluator. Use it only as a deterministic policy regression
for the stand-in. Do not use it to promote production enforcement. Remove or
regenerate it if the approved policy changes.

What this does not solve: It does not create a production compiler or
production-ready deployment. Production evaluation still needs actual
compiler predictions and confidence, representative synthetic cases,
independent human ratings and adjudication, and a target deployment
architecture.
