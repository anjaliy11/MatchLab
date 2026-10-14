# Decision 0003: Deflect binary answers to restricted-category questions

Context: The gate blocked answers confirming the subject's actual restricted
value, but a bare denial to a wrong guess could reveal the same category
information. Checking only values present in the persona missed wrong guesses.

Class: guard.

Threat addressed: `THREAT_MODEL.md`, attack 2 (confirmation oracle).

Eval first: Added regression tests for wrong guesses across all ten selected
sensitive categories, English and Hinglish category questions, and unrelated
benign yes/no questions. Expected result: binary answers to restricted-category
questions are deflected; unrelated benign yes/no questions remain answerable.
Focused tests initially exposed stand-in responses such as "No, he doesn't"
and "Nahi, galat"; the matcher was expanded and those cases are now covered.

Options:
- Match only the subject's known restricted values: low utility cost, but
  misses wrong guesses.
- Deflect binary-style answers when question text matches a restricted-category
  intent pattern: deterministic and category-independent of the actual value,
  but may miss unrecognized phrasings.
- Add semantic judging inline: broader potential coverage but violates the
  deterministic/no-network inline guard invariant.

Choice and why: Use deterministic category-intent patterns for English and
Hinglish, and deflect binary-style replies. Keep content-bearing answers to
unrelated benign questions available. This closes the reproduced wrong-guess
denial without an inline model or network call.

Evidence: `python tests/test_core.py` passes all 27 tests. The regenerated
synthetic safety run reports wrong-guess recall 1.00 (n=150), category-intent
deflection 20/20 across ten categories, benign unrelated yes/no answers 120/120,
and A1 wrong-guess deflection share 1.00 (0.00 before the change). The
category probe is hand-authored, synthetic, and not exhaustive. In the same
run, safe sentence FPR was 0/900 and near-miss FPR was 0/360. A1 gate p50/p95/p99
latency was 3.00/7.96/10.98 ms and A3 was 6.85/13.72/19.06 ms; these are
machine-specific synthetic-run timings, not production measurements.

Costs accepted: A1 wrong-guess deflection share rises by 1.00 on the synthetic
wrong-guess task; the probe's benign yes/no answer rate remains 1.00. No model
or network cost was added. Pattern coverage and utility have not been assessed
with human-reviewed or real-user prompts.

Rollout and rollback: Treat this as shadow-evaluated prototype behavior, not
production enforcement. Before enforcement, evaluate broader category
wordings and benign yes/no traffic, then review deflection share and false
positives. Roll back the category-level rule if those measurements exceed an
approved utility/false-positive bound; do not weaken the named-value
confirmation rule.

What this does not solve: The regex patterns do not recognize every
restricted-category question, and this does not address indirect descriptions,
other scripts, or multi-turn narrowing.
