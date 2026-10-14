# Local gate staging

This local-only adapter exercises the sentence gate over HTTP using one seeded,
synthetic persona and the deterministic stand-in card compiler. It does not run
speech recognition, text-to-speech, a pitch model, a production card compiler,
or persistent state.

Start it from the project root:

```powershell
python scripts/run_local_gate_staging.py --seed 0
```

The service binds only to `127.0.0.1`. It is single-threaded, unauthenticated,
non-persistent, and accepts request bodies up to 16 KiB. Do not expose it on a
LAN, reverse proxy, public host, or with real user/profile data. Request
logging is disabled; request bodies and gate text are not written to disk.
Gate failures return HTTP 503 with a fixed deflection rather than the input
sentence.

Check health:

```powershell
Invoke-RestMethod http://127.0.0.1:8765/health
```

Check one sentence:

```powershell
$body = @{ sentence = "He is thoughtful and curious."; question = "Tell me about him."; stage = "blind" } |
    ConvertTo-Json
Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8765/v1/check `
    -ContentType "application/json" -Body $body
```

The endpoint accepts `sentence` (required string), optional `question` (string
or null), and optional `stage` (`blind` or `mutual`). It returns only the gate
action, reason codes, and output text. Malformed requests fail with a generic
error response. The service has no authentication because it is restricted to
loopback and synthetic staging; that is not an acceptable production access
control design.

This endpoint is a local integration smoke-test, not a production deployment.
Production remains blocked on a real card compiler and independent tier
evaluation, durable atomic budget and memory stores, request authentication,
service-level concurrency and timeouts, privacy-reviewed logging and retention,
operational monitoring, and an approved rollout/rollback plan.
