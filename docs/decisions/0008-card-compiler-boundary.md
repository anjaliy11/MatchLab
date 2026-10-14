# Decision 0008: Add a fail-closed card-compiler boundary

Context: The card compiler is the trusted computing base, but the workspace
only had a deterministic stand-in. The project needed a provider-neutral
contract that rejects malformed output and withholds unknown or low-confidence
tiers without selecting a model vendor.

Class: guard / infra.

Threat addressed: `THREAT_MODEL.md`, attack 14 (card mis-tier), and the core
invariant that the blind-stage agent receives only allowed card forms.

Eval first: Added tests for valid tier mapping, invalid schema, unsupported
categories, unknown tiers, low confidence, missing blind anonymity
attributes, backend exceptions, and invalid thresholds. The test exposed an
existing inconsistency: `mutual_form=None` was documented as `never` but
inherited the BLIND form in the card, evaluator, and stand-in agent. The
maintainer approved strict `never` semantics; all paths now agree.

Options:
- Select a vendor/model and build its adapter now: deferred; no provider was
  approved.
- Trust backend-typed output directly: rejected because provider responses are
  untrusted at the security boundary.
- Add a strict provider-neutral protocol, JSON Schema, runtime validation, and
  default-deny mapping: chosen.

Choice and why: `CardCompiler` defines the backend protocol. The exported
`CARD_COMPILER_OUTPUT_SCHEMA` describes a closed JSON object. The validator
rejects malformed envelopes, drops unsupported or structurally invalid
facts, maps unknown tiers and low-confidence stages to `never`, and requires
blind anonymity attributes before allowing a BLIND form. Provider exceptions
produce an empty card and a sanitized `compiler_error` reason. The confidence
threshold is mandatory at the call site; no policy cutoff is silently
selected. `None` in `mutual_form` now means strictly never.

Evidence: `python tests/test_core.py` covers the failure and success paths.
Pylance reports no diagnostics in `matchlab/compiler.py` or the touched card
schema/evaluator modules. There is no production provider to evaluate, and no
human-adjudicated label set; the synthetic policy baseline remains by
construction.

Costs accepted: No network/model call or dependency is added. Unknown or
low-confidence output can reduce utility by withholding facts. The provider
protocol does not itself sandbox a model, enforce a provider timeout, secure
profile transport, or make an independent pass.

Rollout and rollback: The boundary is opt-in and not wired into the local gate
staging server or live product path. Use it only with synthetic fixtures
until a backend, threat review, timeouts, data-handling policy, and
human-adjudicated evaluation are approved. Roll back by removing the wrapper
from the call path; never bypass validation to recover utility.

What this does not solve: It does not provide a production compiler, isolate
a model or tools, measure compiler accuracy, establish confidence
calibration, or implement a production deployment.
