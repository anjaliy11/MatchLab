# Decision 0007: Add a loopback-only sentence-gate staging adapter

Context: The workspace contains an evaluation and guard library, not a running
voice application. A real STT/agent/TTS pipeline and vendor were not selected.
The maintainer selected local staging of the sentence gate only.

Class: infra.

Threat addressed: `THREAT_MODEL.md`, attacks 1, 2, and 14; this adapter does
not claim to prevent or measure attacks beyond the existing gate tests.

Eval first: Added HTTP integration tests for a restricted-value rewrite,
health response, malformed/oversized/invalid-Unicode requests, gate failure,
and rejection of a non-loopback bind address. Expected behavior: invalid
requests return a generic client error; gate failures return HTTP 503 with a
fixed deflection; neither error response includes the request text.

Options:
- Add a public or LAN-facing service: rejected because authentication,
  deployment, logging, and data-handling policies are not designed.
- Add an app framework dependency: rejected because a small local adapter
  needs no new package.
- Provide a standard-library loopback smoke-test around synthetic cards:
  chosen to verify HTTP integration without implying production readiness.

Choice and why: Added a single-threaded standard-library HTTP server bound
strictly to `127.0.0.1`, accepting only `/health` and `/v1/check`. Requests are
limited to 16 KiB and per-field text to 8 KiB. Access logs are disabled. The
service uses a seeded synthetic persona and the rule-based card compiler;
it makes no model or external network call and writes no request data to disk.
Gate exceptions are logged only by exception type and fail closed with a
fixed deflection.

Evidence: `python tests/test_core.py` passes the HTTP boundary tests,
including the fail-closed path. The CLI was launched with a seeded synthetic
persona; `/health` and a benign `/v1/check` request both returned expected
responses, and the process was stopped afterward. The local adapter's
performance and concurrency are not production measurements.

Costs accepted: No additional dependency, model cost, or external network
call. The single-threaded server is deliberately unsuitable for concurrent
production traffic. Any local process can call it; loopback binding is not
authentication.

Rollout and rollback: Run only on the developer machine with synthetic inputs.
Stop the local process to roll back. Do not expose it through a proxy, firewall
rule, LAN, or public host.

What this does not solve: It does not provide production identity/access
control, persistence, rate limiting, durable atomic anonymity budgets,
production logging/retention, real card compilation, STT/TTS, monitoring,
deployment automation, or production reliability evidence.
