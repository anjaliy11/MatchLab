# Decision 0005: Require adjudicated labels for card-tier evaluation

Context: The compiled disclosure card is the trusted computing base, but the
repository has no human-labelled tier set. The existing card-error sweep
injects selected mis-tier rates, so it measures downstream sensitivity rather
than the real compiler's tier accuracy.

Class: eval.

Threat addressed: `THREAT_MODEL.md`, attack 14 (card mis-tier).

Eval first: Added tests for tier extraction from compiled facts, per-stage
asymmetric loss, low-confidence and unknown prediction downgrades, CSV
validation, and aggregate-only output. Before implementation, the focused test
failed because the evaluator module did not exist. Expected behavior: unsafe
over-disclosure costs more than over-restriction; unknown or below-threshold
predictions are treated as `never`; no item ID or fact text appears in output.

Options:
- Generate synthetic gold tiers from the stand-in compiler: executable now,
  but circular and not evidence of compiler quality.
- Commit example labels as if they were human annotations: creates false
  evidence and was rejected.
- Implement a strict evaluator requiring supplied two-rater/adjudicated labels:
  produces no accuracy claim until labels exist, while making the protocol
  repeatable.

Choice and why: Implement the strict CSV evaluator. It reports raw and
default-deny metrics for BLIND and MUTUAL tiers, grouped by stage and category,
including tier accuracy, false-safe rate, over-disclosure,
over-restriction, rater agreement/Cohen's kappa, and ordinal asymmetric loss.
The loss weights and confidence threshold are mandatory invocation parameters,
not hidden defaults. Unknown or below-threshold predictions are downgraded to
`never`. The evaluator emits aggregate-only output and excludes item IDs.

Evidence: `python tests/test_core.py` passes the evaluator calculation, input
parsing, no-ID-output, and invalid-configuration tests. The CLI has no labelled
input dataset to run against, so actual compiler accuracy, per-category
false-safe rate, and asymmetric loss remain **not measured**. Existing
`sweep_card_error.py` outputs remain injected-error sensitivity analysis.

Costs accepted: No model, network, or runtime guard cost is added. Choosing
cost weights and a confidence threshold is deferred to the maintainer; the
evaluator refuses equal or reversed cost weights. Unit-test values verify
arithmetic only and are not policy recommendations. No claims of compiler
accuracy are made without adjudicated labels.

Rollout and rollback: Use the evaluator in offline review after a synthetic,
two-rater tier set is supplied. Keep compiler decisions and the card-only
rollout in shadow until false-safe rates and costs are reviewed and an
enforcement threshold is agreed. Remove the evaluator if it emits restricted
values or does not match the approved label protocol.

What this does not solve: It does not create a real compiler, assign tiers,
produce human labels, prove rater quality, establish the approved loss ratio,
or decide whether two independent compiler passes are needed.

Follow-up: The maintainer approved an explicit category-by-stage policy
matrix for an initial synthetic annotation packet. A local blank CSV template
and rater procedure were added in `evals/data/card_tier_labels_template.csv`
and `docs/CARD_TIER_ANNOTATION.md`. No human ratings have been supplied yet;
the template is not a compiler-evaluation dataset and public PII-span datasets
do not replace the required stage-tier adjudication.
