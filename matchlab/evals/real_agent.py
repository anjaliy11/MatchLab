from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import os
import platform
import random
import subprocess
import time
import uuid
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, MutableMapping, Sequence
from urllib.parse import urlsplit

from ..adapters.openai_compat import (
    ABORT_OUTCOMES,
    RETRYABLE_OUTCOMES,
    CompletionError,
    OpenAICompatibleAdapter,
    resolve_api_key,
    validate_base_url,
)
from ..adapters.prompts import build_prompt_messages, prompt_template_text, to_chat_messages
from ..guards.budget import DisclosureBudget
from ..guards.gate import DEFLECTIONS, SentenceGate
from ..schemas import Stage
from .analysis import analyze_recordings
from .oracle import GroundTruthOracle
from .personas import compile_card, make_pool
from .tasks import EvalTask

_ARM_MODE = {
    "r0": ("R0_full_profile", "r0", False),
    "R0_full_profile": ("R0_full_profile", "r0", False),
    "r1": ("R1_card_only", "r1", False),
    "R1_card_only": ("R1_card_only", "r1", False),
    "r0n": ("R0n_no_instruction", "r0n", False),
    "R0n_no_instruction": ("R0n_no_instruction", "r0n", False),
    "r0_gate": ("R0_full_profile+gate", "r0", True),
    "R0_full_profile+gate": ("R0_full_profile+gate", "r0", True),
    "r1_gate": ("R1_card_only+gate", "r1", True),
    "R1_card_only+gate": ("R1_card_only+gate", "r1", True),
}


@dataclass(frozen=True)
class RunConfig:
    base_url: str
    model: str
    output_root: Path
    api_key: str | None = None
    arms: tuple[str, ...] = ("r0", "r1")
    trials: int = 1
    temperature: float = 0.7
    max_tokens: int = 150
    max_calls: int = 100
    max_minutes: float = 120.0
    max_usd: float | None = None
    input_price_per_million_tokens: float | None = None
    output_price_per_million_tokens: float | None = None
    seed: int = 0
    human_review_sample_per_arm: int = 50
    max_retries: int = 3


@dataclass(frozen=True)
class RunResult:
    run_dir: Path
    manifest: dict[str, Any]


class _BudgetExceeded(RuntimeError):
    pass


def load_dotenv(
    path: str | Path,
    env: MutableMapping[str, str] | None = None,
) -> None:
    """Load optional key/value configuration without replacing host environment values."""
    target = os.environ if env is None else env
    env_path = Path(path)
    if not env_path.exists():
        return
    text = env_path.read_text(encoding="utf-8-sig")
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
            value = value[1:-1]
        if key and key not in target:
            target[key] = value


def _hash_json(value: Any) -> str:
    raw = json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _host(base_url: str) -> tuple[str, str]:
    parts = urlsplit(base_url)
    hostname = parts.hostname or ""
    host = f"{hostname}:{parts.port}" if parts.port else hostname
    try:
        is_loopback = ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        is_loopback = hostname.lower() == "localhost"
    return host, "local" if is_loopback else "hosted"


def _git_sha() -> str:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=2,
        )
    except (OSError, subprocess.SubprocessError):
        return "unavailable"
    return result.stdout.strip()


def _hardware_summary(host_kind: str) -> dict[str, Any] | None:
    if host_kind != "local":
        return None
    return {
        "machine_architecture": platform.machine(),
        "logical_cpu_count": os.cpu_count(),
    }


def _safe_value(value: Any, secret: str | None) -> Any:
    if isinstance(value, str):
        return OpenAICompatibleAdapter.redact(value, secret)
    if isinstance(value, dict):
        return {str(key): _safe_value(item, secret) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_safe_value(item, secret) for item in value]
    return value


def _atomic_json(path: Path, data: dict[str, Any], secret: str | None) -> None:
    serialized = json.dumps(
        _safe_value(data, secret),
        ensure_ascii=False,
        sort_keys=True,
        indent=2,
    )
    temp_path = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    with temp_path.open("w", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temp_path.replace(path)


def _new_run_dir(output_root: Path) -> tuple[str, Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    for _ in range(5):
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_id = f"{timestamp}-{uuid.uuid4().hex[:12]}"
        run_dir = output_root / run_id
        try:
            run_dir.mkdir()
        except FileExistsError:
            continue
        return run_id, run_dir
    raise FileExistsError("could not allocate a unique real-agent run directory")


def _load_cache(output_root: Path) -> dict[str, dict[str, Any]]:
    cache: dict[str, dict[str, Any]] = {}
    if not output_root.exists():
        return cache
    for file_path in sorted(output_root.glob("*/transcripts.jsonl")):
        with file_path.open(encoding="utf-8") as source:
            for line in source:
                if not line.strip():
                    continue
                try:
                    row = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if (
                    isinstance(row, dict)
                    and row.get("outcome") == "ok"
                    and isinstance(row.get("cache_key"), str)
                    and isinstance(row.get("response"), str)
                ):
                    cache.setdefault(row["cache_key"], row)
    return cache


def _append_jsonl(path: Path, row: dict[str, Any], secret: str | None) -> None:
    serialized = json.dumps(
        _safe_value(row, secret),
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
    )
    with path.open("a", encoding="utf-8", newline="\n") as stream:
        stream.write(serialized + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _write_human_review(
    path: Path,
    rows: list[dict[str, Any]],
    max_per_arm: int,
    secret: str | None,
) -> None:
    fields = [
        "record_id",
        "arm",
        "persona_id",
        "task_id",
        "trial",
        "question",
        "response",
        "oracle_leaks",
        "oracle_confirmation",
        "reviewer_1_leak",
        "reviewer_2_leak",
    ]
    selected: list[dict[str, Any]] = []
    per_arm: dict[str, int] = {}
    for row in rows:
        if row.get("outcome") != "ok":
            continue
        arm = str(row["arm"])
        if per_arm.get(arm, 0) >= max_per_arm:
            continue
        per_arm[arm] = per_arm.get(arm, 0) + 1
        oracle = row.get("oracle") or {}
        selected.append(
            {
                "record_id": row["record_id"],
                "arm": arm,
                "persona_id": row["persona_id"],
                "task_id": row["task_id"],
                "trial": row["trial"],
                "question": row["question"],
                "response": row["response"],
                "oracle_leaks": ",".join(oracle.get("leaks", [])),
                "oracle_confirmation": str(bool(oracle.get("confirmation"))).lower(),
                "reviewer_1_leak": "",
                "reviewer_2_leak": "",
            }
        )
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in selected:
            writer.writerow(_safe_value(row, secret))


def _estimate_tokens(messages: list[dict[str, str]], max_tokens: int) -> tuple[int, int]:
    input_bytes = sum(len(message["content"].encode("utf-8")) for message in messages)
    return input_bytes, max_tokens


def _estimated_cost(
    input_bytes: int,
    output_tokens: int,
    input_price: float | None,
    output_price: float | None,
) -> float:
    if input_price is None or output_price is None:
        return 0.0
    return (input_bytes * input_price + output_tokens * output_price) / 1_000_000


def _run_request(
    adapter: OpenAICompatibleAdapter,
    messages: list[dict[str, str]],
    *,
    temperature: float,
    max_tokens: int,
    config: RunConfig,
    deadline: float,
    state: dict[str, Any],
    rng: random.Random,
    message_cost: float,
    warmup: bool = False,
) -> tuple[dict[str, Any] | None, CompletionError | None]:
    retries = 0
    while True:
        if state["network_attempts"] >= config.max_calls:
            raise _BudgetExceeded("--max-calls limit reached")
        if time.monotonic() >= deadline:
            raise _BudgetExceeded("--max-minutes limit reached")
        if config.max_usd is not None and state["estimated_usd"] + message_cost > config.max_usd:
            raise _BudgetExceeded("--max-usd limit reached")
        adapter.timeout_s = min(adapter.timeout_s, max(0.001, deadline - time.monotonic()))
        state["network_attempts"] += 1
        state["estimated_usd"] += message_cost
        if warmup:
            state["warmup_attempts"] += 1
        else:
            state["scored_api_attempts"] += 1
        try:
            response = adapter.complete(
                messages,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            (state["warmup_outcomes"] if warmup else state["attempt_outcomes"])["ok"] += 1
            return response, None
        except CompletionError as exc:
            (state["warmup_outcomes"] if warmup else state["attempt_outcomes"])[exc.outcome] += 1
            if exc.outcome not in RETRYABLE_OUTCOMES or retries >= config.max_retries:
                return None, exc
            retries += 1
            delay = exc.retry_after_s
            if delay is None:
                delay = (2 ** (retries - 1)) + rng.random()
            if time.monotonic() + delay >= deadline:
                return None, CompletionError("timeout", "--max-minutes limit reached")
            time.sleep(delay)


def _canonical_arm(arm: str) -> tuple[str, str, bool]:
    try:
        return _ARM_MODE[arm]
    except KeyError as exc:
        raise ValueError(f"unsupported real-agent arm: {arm}") from exc


def _config_errors(config: RunConfig) -> list[str]:
    errors: list[str] = []
    try:
        validate_base_url(config.base_url)
    except ValueError as exc:
        errors.append(str(exc))
    if not config.model.strip():
        errors.append("LLM_MODEL is required")
    if config.trials < 1 or config.max_calls < 1 or config.max_minutes <= 0:
        errors.append("trials, max-calls, and max-minutes must be positive")
    if config.temperature <= 0:
        errors.append("temperature must be greater than zero")
    if config.max_tokens < 1:
        errors.append("max-tokens must be positive")
    if config.max_retries < 0 or config.max_retries > 3:
        errors.append("max-retries must be between 0 and 3")
    if config.max_usd is not None and (
        config.input_price_per_million_tokens is None
        or config.output_price_per_million_tokens is None
    ):
        errors.append("max-usd requires both user-supplied per-million-token prices")
    if config.max_usd is not None and config.max_usd <= 0:
        errors.append("max-usd must be positive")
    try:
        _, host_kind = _host(config.base_url)
    except ValueError:
        host_kind = "unknown"
    if host_kind == "hosted" and config.max_usd is None:
        errors.append("hosted runs require --max-usd and user-supplied token prices")
    if any(
        price is not None and price < 0
        for price in (
            config.input_price_per_million_tokens,
            config.output_price_per_million_tokens,
        )
    ):
        errors.append("token prices cannot be negative")
    for arm in config.arms:
        try:
            _canonical_arm(arm)
        except ValueError as exc:
            errors.append(str(exc))
    return errors


def run_evaluation(config: RunConfig, tasks: Sequence[EvalTask]) -> RunResult:
    errors = _config_errors(config)
    if errors:
        raise ValueError("; ".join(errors))
    base_url = validate_base_url(config.base_url)
    api_key = config.api_key.strip() if config.api_key and config.api_key.strip() else resolve_api_key(base_url)
    adapter = OpenAICompatibleAdapter(base_url=base_url, api_key=api_key, model=config.model)
    host, host_kind = _host(base_url)
    run_id, run_dir = _new_run_dir(config.output_root)
    transcript_path = run_dir / "transcripts.jsonl"
    transcript_path.touch(exist_ok=False)
    cache = _load_cache(config.output_root)
    task_hash = _hash_json(
        [
            {
                "task_id": task.task_id,
                "scenario": task.scenario,
                "suite": task.suite,
                "turns": task.turns,
                "kind": task.kind,
                "expect": task.expect,
            }
            for task in tasks
        ]
    )
    template_hash = hashlib.sha256(prompt_template_text().encode("utf-8")).hexdigest()
    canonical_arms = [_canonical_arm(arm)[0] for arm in config.arms]
    manifest: dict[str, Any] = {
        "schema_version": 1,
        "run_id": run_id,
        "status": "running",
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model_requested": config.model,
        "model_reported": None,
        "model_digest": None,
        "host": host,
        "host_kind": host_kind,
        "temperature": config.temperature,
        "max_tokens": config.max_tokens,
        "trials_requested": config.trials,
        "persona_count": len({task.persona.user_id for task in tasks}),
        "persona_seed": config.seed,
        "arms": canonical_arms,
        "prompt_template_version": "real-agent-v1",
        "prompt_template_hash": template_hash,
        "task_list_hash": task_hash,
        "matchlab_git_sha": _git_sha(),
        "python_version": platform.python_version(),
        "operating_system": platform.system(),
        "hardware_summary": _hardware_summary(host_kind),
        "warmup": {"performed": False, "scored": False, "outcome": "cached_skipped"},
        "counts": {
            "logical_calls": 0,
            "api_attempts": 0,
            "cache_hits": 0,
            "outcomes_by_arm": {},
            "attempt_outcomes": {},
        },
        "estimated_usd": 0.0,
        "invalid_reasons": [],
        "label_status": "oracle-labelled, not human-validated",
        "evidence_status": "not tested on production or real-user data",
    }
    secret = api_key
    _atomic_json(run_dir / "manifest.json", manifest, secret)

    rng = random.Random(config.seed)
    run_started = time.monotonic()
    deadline = run_started + config.max_minutes * 60
    state: dict[str, Any] = {
        "network_attempts": 0,
        "scored_api_attempts": 0,
        "warmup_attempts": 0,
        "estimated_usd": 0.0,
        "attempt_outcomes": Counter(),
        "warmup_outcomes": Counter(),
    }
    scheduled: list[
        tuple[
            str,
            str,
            bool,
            EvalTask,
            int,
            int,
            int,
            dict[str, Any],
            list[dict[str, str]],
            str,
        ]
    ] = []
    for configured_arm in config.arms:
        canonical_arm, mode, gated = _canonical_arm(configured_arm)
        for task in tasks:
            card = compile_card(task.persona, seed=config.seed)
            for trial in range(config.trials):
                for turn_index, question in enumerate(task.turns):
                    prompt = build_prompt_messages(
                        mode=mode,
                        persona=task.persona if mode in {"r0", "r0n"} else None,
                        card=card if mode == "r1" else None,
                        question=question,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                    )
                    messages = to_chat_messages(prompt)
                    prompt_hash = _hash_json(messages)
                    cache_key = _hash_json(
                        {
                            "model": config.model,
                            "host": host,
                            "prompt_hash": prompt_hash,
                            "temperature": config.temperature,
                            "max_tokens": config.max_tokens,
                            "trial_index": trial,
                        }
                    )
                    scheduled.append(
                        (
                            canonical_arm,
                            mode,
                            gated,
                            task,
                            trial,
                            turn_index,
                            len(task.turns),
                            prompt,
                            messages,
                            cache_key,
                        )
                    )

    uncached = [entry for entry in scheduled if entry[-1] not in cache]
    manifest["calls_planned"] = len(scheduled)
    manifest["cache_hits_planned"] = len(scheduled) - len(uncached)
    abort = False
    if uncached:
        warmup_messages = [
            {"role": "system", "content": "This is an unscored endpoint warm-up."},
            {"role": "user", "content": "Reply with OK."},
        ]
        warmup_input, warmup_output = _estimate_tokens(warmup_messages, 4)
        warmup_cost = _estimated_cost(
            warmup_input,
            warmup_output,
            config.input_price_per_million_tokens,
            config.output_price_per_million_tokens,
        )
        manifest["warmup"] = {"performed": True, "scored": False}
        try:
            warmup_response, warmup_error = _run_request(
                adapter,
                warmup_messages,
                temperature=config.temperature,
                max_tokens=4,
                config=config,
                deadline=deadline,
                state=state,
                rng=rng,
                message_cost=warmup_cost,
                warmup=True,
            )
        except _BudgetExceeded as exc:
            warmup_response, warmup_error = None, CompletionError("bad_response", str(exc))
        if warmup_error is not None:
            manifest["warmup"].update(
                {
                    "outcome": warmup_error.outcome,
                    "message": str(warmup_error),
                    "attempt_count": state["warmup_attempts"],
                    "attempt_outcomes": dict(sorted(state["warmup_outcomes"].items())),
                }
            )
            manifest["invalid_reasons"].append("warm-up failed; no scored calls were made")
            abort = True
        else:
            manifest["warmup"].update(
                {
                    "outcome": "ok",
                    "attempt_count": state["warmup_attempts"],
                    "attempt_outcomes": dict(sorted(state["warmup_outcomes"].items())),
                    "model_reported": warmup_response.get("model"),
                    "model_digest": warmup_response.get("system_fingerprint"),
                    "usage": warmup_response.get("usage"),
                }
            )
            manifest["model_reported"] = warmup_response.get("model")
            manifest["model_digest"] = warmup_response.get("system_fingerprint")

    scored_rows: list[dict[str, Any]] = []
    gate_pool = make_pool(seed=config.seed) if any(entry[2] for entry in scheduled) else None
    if not abort:
        for (
            canonical_arm,
            mode,
            gated,
            task,
            trial,
            turn_index,
            turn_count,
            prompt,
            messages,
            cache_key,
        ) in scheduled:
            if time.monotonic() >= deadline:
                manifest["invalid_reasons"].append(
                    "--max-minutes limit reached before all planned calls"
                )
                break
            cached = cache.get(cache_key)
            completion: dict[str, Any] | None = None
            error: CompletionError | None = None
            if cached is not None:
                manifest["counts"]["cache_hits"] += 1
                completion = {
                    "model": cached.get("model_reported"),
                    "system_fingerprint": cached.get("model_digest"),
                    "usage": cached.get("usage"),
                }
                response_text = cached["response"]
            else:
                input_bytes, output_bound = _estimate_tokens(messages, config.max_tokens)
                message_cost = _estimated_cost(
                    input_bytes,
                    output_bound,
                    config.input_price_per_million_tokens,
                    config.output_price_per_million_tokens,
                )
                try:
                    completion, error = _run_request(
                        adapter,
                        messages,
                        temperature=config.temperature,
                        max_tokens=config.max_tokens,
                        config=config,
                        deadline=deadline,
                        state=state,
                        rng=rng,
                        message_cost=message_cost,
                    )
                except _BudgetExceeded as exc:
                    manifest["invalid_reasons"].append(str(exc))
                    break
                if error is not None:
                    response_text = ""
                    if error.outcome in ABORT_OUTCOMES:
                        cause = str(error)
                        if error.outcome == "connection_error":
                            cause += "; is the server running?"
                        manifest["invalid_reasons"].append(cause)
                        abort = True
                else:
                    choices = completion.get("choices") if completion else None
                    first = choices[0] if isinstance(choices, list) and choices else {}
                    message = first.get("message") if isinstance(first, dict) else None
                    response_text = (
                        message.get("content", "")
                        if isinstance(message, dict)
                        else ""
                    )

            outcome = error.outcome if error else "ok"
            question = str(prompt["user"])
            oracle_result = GroundTruthOracle(task.persona).label_response(
                response_text,
                question=question,
            )
            delivered_text = response_text
            gate_action = None
            gate_reasons: list[str] = []
            if gated and gate_pool is not None and outcome == "ok":
                gate = SentenceGate(
                    compile_card(task.persona, seed=config.seed),
                    Stage.BLIND,
                    DisclosureBudget(gate_pool),
                    viewer_id=f"real-agent:{task.persona.user_id}:{trial}",
                )
                gate_result = gate.check(response_text, question)
                gate_action = gate_result.action
                gate_reasons = gate_result.reasons
                delivered_text = gate_result.text
            delivered_oracle = GroundTruthOracle(task.persona).label_response(
                delivered_text,
                question=question,
            )
            row = {
                "record_id": f"{task.task_id}:{canonical_arm}:{trial}:{turn_index}",
                "arm": canonical_arm,
                "task_id": task.task_id,
                "scenario": task.scenario,
                "suite": task.suite,
                "persona_id": task.persona.user_id,
                "trial": trial,
                "turn_index": turn_index,
                "turn_count": turn_count,
                "question": question,
                "response": response_text,
                "delivered_reply": delivered_text,
                "outcome": outcome,
                "cached": cached is not None,
                "cache_key": cache_key,
                "prompt_hash": _hash_json(messages),
                "model_reported": completion.get("model") if completion else None,
                "model_digest": completion.get("system_fingerprint") if completion else None,
                "usage": completion.get("usage") if completion else None,
                "benign": task.kind == "benign",
                "expected": task.expect,
                "oracle": {
                    "leaks": oracle_result["leaks"],
                    "matches": oracle_result["matches"],
                    "confirmation": oracle_result["confirmation"],
                    "confirmation_categories": oracle_result["confirmation_categories"],
                },
                "delivered_oracle": {
                    "leaks": delivered_oracle["leaks"],
                    "confirmation": delivered_oracle["confirmation"],
                },
                "gate_action": gate_action,
                "gate_reasons": gate_reasons,
                "is_deflection": delivered_text in DEFLECTIONS,
            }
            scored_rows.append(row)
            _append_jsonl(transcript_path, row, secret)
            manifest["counts"]["logical_calls"] += 1
            arm_counts = manifest["counts"]["outcomes_by_arm"].setdefault(canonical_arm, {})
            arm_counts[outcome] = arm_counts.get(outcome, 0) + 1
            if completion and completion.get("model"):
                manifest["model_reported"] = completion["model"]
                manifest["model_digest"] = completion.get("system_fingerprint")
            if abort:
                break

    manifest["counts"]["api_attempts"] = state["scored_api_attempts"]
    manifest["counts"]["attempt_outcomes"] = dict(sorted(state["attempt_outcomes"].items()))
    manifest["estimated_usd"] = round(state["estimated_usd"], 12)
    manifest["personas_completed"] = len({row["persona_id"] for row in scored_rows})
    manifest["trials_completed"] = len({row["trial"] for row in scored_rows})
    manifest["duration_seconds"] = round(time.monotonic() - run_started, 6)
    if manifest["invalid_reasons"]:
        manifest["status"] = "invalid_run"
    else:
        manifest["status"] = "completed"
        report = analyze_recordings(run_dir)
        if not report["valid"]:
            manifest["status"] = "invalid_run"
            manifest["invalid_reasons"].extend(report["invalid_reasons"])
        else:
            report["report_heading"] = (
                f"measured on {config.model}, "
                f"{manifest['persona_count']} personas x {config.trials} trials, "
                f"oracle-labelled, via {host_kind} host {host}; "
                "not representative of production models"
            )
            _atomic_json(run_dir / "analysis.json", report, secret)
    _write_human_review(
        run_dir / "human_review.csv",
        scored_rows,
        config.human_review_sample_per_arm,
        secret,
    )
    _atomic_json(run_dir / "manifest.json", manifest, secret)
    return RunResult(run_dir=run_dir, manifest=manifest)


__all__ = ["RunConfig", "RunResult", "load_dotenv", "run_evaluation"]
