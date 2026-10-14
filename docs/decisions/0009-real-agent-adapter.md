# Decision 0009: Real-agent evaluation adapter

Class: eval / infra.

Context: The real-agent shadow artifacts under `results/real_agent_shadow_hf/`
do not contain Hugging Face model outputs. The transcript records show provider
failures followed by mock fallback replies, so they cannot support agent
quality, safety, or human-adjudication claims. The project needs a bounded way
to diagnose endpoint access and, only after that succeeds, collect synthetic
real-agent outputs for evaluation.

Threat or failure addressed: E1 and E2 in `.github/copilot-instructions.md`;
`THREAT_MODEL.md` attacks 1 (direct probe), 2 (confirmation oracle), 3
(indirect description), 4 (other-script rendering), 7 (viewer prompt
injection), and 8 (bio prompt injection). This work adds evidence about agent
behavior; it does not change the existing gate or make it production-ready.

Eval first: Add offline loopback-server acceptance tests before implementing
the network adapter. The tests must prove exact outcome classification for
success, authentication failure, unavailable model, rate limiting, exhausted
credits, server error, timeout, malformed response, empty output, and length
truncation; secret redaction; import boundaries; cache replay; call-budget
stopping; and byte-identical analysis from the same recordings. Expected
behavior is that only `ok` calls enter metric denominators, and every other
outcome is reported as an infrastructure failure, never as a pass or a
no-leak result.

Options:
- Use a provider SDK or provider-specific clients: rejected for this scope
  because they add dependencies and provider coupling.
- Use a mock-only runner: rejected as real-agent evidence; it remains useful
  only for local plumbing tests.
- Use one standard-library OpenAI-compatible adapter configured only through
  `LLM_BASE_URL`, `LLM_API_KEY`/`HF_TOKEN`, and `LLM_MODEL`: selected, subject
  to the smoke-test and budget gates below.

Choice and why: Implement only the provider-neutral adapter, runner, recording
format, independent string-checkable oracle, and deterministic analysis
specified in section 12 of `.github/copilot-instructions.md`. The smoke
diagnostic must call only the configured base URL. A failed smoke test stops
the work; no alternate provider, model, or endpoint will be tried. No model
will judge outputs in this phase. The run begins in shadow mode and uses
synthetic personas only.

Evidence: Inspection of `results/real_agent_shadow_hf/transcripts.jsonl`
found 1,140 provider-failure warning records and 1,140 mock-fallback reply
records; all replies used one generic mock sentence. The associated summary
only reports requested configuration and task counts. There are no real
provider replies or human labels in those artifacts. The existing 47 core
tests cover the current prototype, not the new provider adapter. The live
endpoint smoke test, adapter acceptance tests, and any real-agent metrics are
not yet run.

Costs we accept: No dependency, guard change, threshold change, or model
judge. Calls are bounded by explicit `--max-calls` and user-supplied
`--max-usd`; no built-in model prices are assumed. Dry-run output will report
call count and token estimate. Live-run token usage and provider-reported cost
are recorded only if returned. Deflection share, benign answer rate,
false-positive rate, and gate p50/p95 latency are not yet measured for a real
agent; any later report must mark them as not measured until supported by
recordings and labels.

Unresolved constraint: section 12.5 calls for committing transcripts under
`results/real_agent/<run_id>/`, while hard invariant 4 forbids raw restricted
values in results. Model replies and prompts may contain synthetic restricted
values. This decision record does not authorize violating that invariant.
Before persistent recording or committing, the recording representation must
be agreed so human review remains meaningful without placing prohibited raw
values in committed results.

What would change my mind: A smoke result other than `ok` stops the run; the
endpoint, model, and provider will not be changed to work around it. Any need
for a dependency, another host, model-as-judge, gate-path change, or
threshold/task/oracle adjustment requires a separate maintainer decision.

Rollout and rollback: First run the diagnostic smoke test and stop unless it
returns `ok`. Then add and demonstrate the offline loopback tests. Run dry-run
and report call count and token estimate before spending. After approval of
the recording representation, run only the explicitly scoped two-persona,
one-trial proof; keep it shadow-only. The default subset and larger runs
require a separate maintainer confirmation and an adequate approved budget.
Never overwrite a run directory. If the adapter or recording violates the
taxonomy, budget, import-boundary, or privacy constraints, stop collection
and do not publish metrics.

What this does not solve: The initial oracle is not human-validated and does
not establish production-model representativeness, compiler accuracy, human
label reliability, or production readiness.
