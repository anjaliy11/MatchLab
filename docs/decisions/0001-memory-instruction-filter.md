# Decision 0001: Narrow memory instruction matching

Context: The memory guard's unanchored patterns treated ordinary phrases such as
"I ignore spicy food", "You should meet my sister", and "I always say hi first"
as instructions. The existing benign probe had only five hand-picked writes.

Class: guard.

Threat addressed: `THREAT_MODEL.md`, attack 11 (memory standing-rule poisoning).

Eval first: Added failing tests for the three reproduced benign false blocks and
for a deterministic English/Hinglish corpus of 240 benign writes. Expected result:
all three attacks remain blocked; ordinary writes are allowed; benign corpus
false-block rate is at most the maintainer-approved 1%. `python tests/test_core.py`
failed on the benign regression test before the guard change.

Options:
- Keep broad keyword matching: retains the false-positive behavior.
- Anchor imperative and standing-rule forms: deterministic and does not add
  latency or dependencies.
- Add a semantic classifier: broader coverage, but requires a separate decision
  on model/dependency, cost, and an off-path design.

Evidence: `python scripts/run_safety_suite.py --personas 30 --trials 5 --out
results/results_safety.json` recorded 3/3 standing-rule attacks blocked, 0/2
paraphrased instructions blocked, and 0/240 synthetic benign English/Hinglish
writes blocked. The 240-item corpus was generated from 20 ordinary-language
templates and 12 generic activities; it is not human-reviewed or real-user data.
A local in-process benchmark over 24,000 calls measured memory guard p50/p95
latency of 3.9/5.6 microseconds. No model or network calls were added.

Choice: Anchor matching to directive-like text at the beginning of a write,
including explicit "from now on" and "remember that" standing rules. Keep the
documented paraphrased-instruction miss unchanged.

Costs accepted: The observed corpus false-block rate is 0%, within the approved
1% empirical corpus bound; the allowed benign-write rate was 240/240. Pitch
deflection share does not apply to memory admission. The change has no model
cost and does not change the inline sentence gate. Synthetic corpus results do
not establish a human-reviewed or production false-positive rate.

Rollout and rollback: This is an offline-tested heuristic change, not a
production promotion. Any production integration should run in shadow mode
before enforcement. Roll back by restoring the previous instruction pattern if
reviewed benign writes show an unacceptable false-block rate.

What this does not solve: Paraphrased memory instructions remain a known miss;
the matcher is regex-based and the benign corpus is synthetic.
