# Decision 0002: Require memory-write timestamps

Context: `MemoryWrite.ts` defaulted to `0.0`. Writes constructed without an
explicit timestamp all shared the same time, so the per-subject sliding window
could not expire those entries.

Class: guard.

Threat addressed: `THREAT_MODEL.md`, attack 13 (memory slow poisoning).

Eval first: Added a test requiring construction without `ts` to fail and a
deterministic fake-clock test that admits a write, rate-limits a second write
inside the window, and admits a later write after the window expires. Before the
change, the required-timestamp test failed.

Options:
- Use each write's explicit event timestamp for the existing window.
- Add an injected guard clock for processing/arrival time while retaining the
  event timestamp separately.

Choice and why: Require `MemoryWrite.ts` and use caller-supplied timestamps.
This is consistent with the existing in-repository call sites, which all supply
timestamps, and avoids introducing a separate clock abstraction.

Evidence: `python tests/test_core.py` passes the required-timestamp test and
fake-clock expiry test. The full 25-test suite passes. Existing memory eval
fixtures already provide explicit timestamps. No randomization or model calls
are involved.

Costs accepted: Callers outside this workspace must now provide an event
timestamp. Window expiry depends on callers providing monotonically advancing
timestamps; delayed or out-of-order events are not normalized to arrival time.
No model cost or pitch deflection applies.

Rollout and rollback: Keep any production integration in shadow mode until its
timestamp source and ordering are verified. Roll back by restoring the default
only if compatibility requires it; do not treat constant/default timestamps as
valid rate-limit time.

What this does not solve: The limiter remains process-local, and this change
does not provide atomic cross-worker rate limiting or handle out-of-order event
timestamps.
