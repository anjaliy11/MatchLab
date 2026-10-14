# Decision 0004: Use explicit specificity and category context for common words

Context: Exact token matches for a restricted value can also match ordinary
prose when the value is a common word. The project prohibits real company or
person names in repository tests, so the realistic public-gazetteer portion of
D4 conflicts with the existing hard invariant. The maintainer approved a
synthetic-only probe and accepted the context-free recall trade-off.

Class: guard.

Threat addressed: `THREAT_MODEL.md`, attack 1 (direct probe).

Eval first: Added tests for three generic synthetic words used as employer,
city, and school values. Expected results: ordinary prose and unrelated
yes/no questions pass, while value mentions in matching category context and
category-context confirmations are blocked. Out-of-range specificity scores
must fail construction.

Options:
- Continue blocking every exact occurrence: protects context-free mentions,
  but retains false positives on ordinary uses of common words.
- Require category context for explicitly low-specificity values: reduces
  ordinary-word false positives, but may miss context-free entity mentions.
- Use a frequency lexicon or public gazetteer to infer specificity: broader
  evidence, but conflicts with the no-real-names fixture invariant unless that
  invariant is explicitly changed.

Choice and why: Add `Fact.value_specificity` in the range 0.0-1.0, defaulting
to 1.0 (distinctive). Values below 0.5 require a cue for their category when
matching both reply text and the viewer's question. Categories without a cue
remain fail-closed and continue matching the value. The current prototype does
not assign specificity automatically; the synthetic probe supplies scores
explicitly. The maintainer approved the 0.5 cutoff after reviewing the
implementation choice.

Evidence: `python tests/test_core.py` passes all 30 tests. The regenerated
synthetic safety run reports ordinary-word prose allowed 3/3, unrelated
yes/no questions allowed 3/3, category-context leaks blocked 3/3, and
category-context confirmations blocked 3/3. The general safe-sentence FPR was
0/900 and the near-miss probe FPR was 0/360; these corpora do not include
human-reviewed prompts or a public gazetteer. In the same run, A1 gate latency
p50/p95/p99 was 3.79/10.14/12.81 ms and A3 was 6.69/13.23/19.79 ms. Both p95
values were below 20 ms. These are machine-specific synthetic-run timings, not
production measurements.

Costs accepted: Ordinary uses represented in this small probe are not
deflected, and context-bearing leaks are blocked. Context-free mentions of
low-specificity values may pass, including value-only confirmation questions.
No model, network, or dependency cost was added. This result is partial D4
coverage; it does not establish an automatic specificity score or realistic
gazetteer performance.

Rollout and rollback: Keep this metadata path in shadow until the offline
compiler can assign and validate specificity scores and broader benign and
attack prompts are reviewed. Do not lower a value's score based only on an
unvalidated heuristic. Roll back category-context gating for a value if
reviewed context-free attacks show unacceptable misses; the default score of
1.0 preserves the prior matching behavior when no score is supplied.

What this does not solve: The probe is synthetic and deliberately omits public
company, city, and school gazetteers. Specificity is manually supplied, and
category cues do not cover every wording or context-free mention.
