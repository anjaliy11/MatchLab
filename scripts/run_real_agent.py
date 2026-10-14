from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from matchlab.adapters.prompts import build_prompt_messages, to_chat_messages
from matchlab.evals.analysis import analyze_recordings
from matchlab.evals.personas import compile_card, make_personas
from matchlab.evals.real_agent import RunConfig, load_dotenv, run_evaluation
from matchlab.evals.tasks import build_tasks


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the synthetic-persona real-agent evaluation against one configured endpoint."
    )
    parser.add_argument("--live", action="store_true", help="Allow calls to the explicitly configured LLM_BASE_URL.")
    parser.add_argument("--dry-run", action="store_true", help="Estimate the run without creating files or making requests.")
    parser.add_argument("--analyze", type=Path, help="Analyze an existing recorded run without making requests.")
    parser.add_argument("--env-file", type=Path, default=ROOT / ".env", help="Optional environment file (default: .env).")
    parser.add_argument("--output-root", type=Path, default=ROOT / "results" / "real_agent")
    parser.add_argument("--personas", type=int, default=2)
    parser.add_argument("--trials", type=int, default=1)
    parser.add_argument("--scenarios", help="Optional comma-separated scenario allowlist.")
    parser.add_argument("--arms", default="R0_full_profile,R1_card_only")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=150)
    parser.add_argument("--max-calls", type=int, default=100)
    parser.add_argument("--max-minutes", type=float, default=30)
    parser.add_argument("--max-usd", type=float)
    parser.add_argument("--input-price-per-million-tokens", type=float)
    parser.add_argument("--output-price-per-million-tokens", type=float)
    parser.add_argument("--max-retries", type=int, default=3)
    return parser.parse_args()


def _selected_tasks(args: argparse.Namespace):
    if args.personas < 1:
        raise ValueError("--personas must be positive")
    tasks = build_tasks(make_personas(args.personas, seed=args.seed))
    if args.scenarios:
        requested = {part.strip() for part in args.scenarios.split(",") if part.strip()}
        known = {task.scenario for task in tasks}
        unknown = requested - known
        if unknown:
            raise ValueError(f"unknown scenario(s): {', '.join(sorted(unknown))}")
        tasks = [task for task in tasks if task.scenario in requested]
    if not tasks:
        raise ValueError("no evaluation tasks selected")
    return tasks


def _dry_run(args: argparse.Namespace, tasks) -> int:
    arms = [part.strip() for part in args.arms.split(",") if part.strip()]
    logical_calls = sum(len(task.turns) for task in tasks) * args.trials * len(arms)
    warmup_calls = 1
    estimated_calls = min(logical_calls + warmup_calls, args.max_calls)
    input_bytes = 0
    for arm in arms:
        prompt_mode = "r0n" if arm in {"r0n", "R0n_no_instruction"} else (
            "r0" if arm in {"r0", "R0_full_profile", "r0_gate", "R0_full_profile+gate"} else "r1"
        )
        for task in tasks:
            card = compile_card(task.persona, seed=args.seed)
            for question in task.turns:
                prompt = build_prompt_messages(
                    mode=prompt_mode,
                    persona=task.persona if prompt_mode in {"r0", "r0n"} else None,
                    card=card if prompt_mode == "r1" else None,
                    question=question,
                    temperature=args.temperature,
                    max_tokens=args.max_tokens,
                )
                input_bytes += sum(
                    len(message["content"].encode("utf-8"))
                    for message in to_chat_messages(prompt)
                )
    estimated_tokens = estimated_calls * args.max_tokens + input_bytes
    print(f"Dry run: {logical_calls} logical calls across {len(arms)} arm(s).")
    print(f"Estimated outbound calls, including one warm-up: {estimated_calls} (cap {args.max_calls}).")
    print(f"Estimated token upper bound: {estimated_tokens}")
    throughput = None
    for manifest_path in sorted(
        args.output_root.glob("*/manifest.json"), reverse=True
    ):
        try:
            previous = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        elapsed = previous.get("duration_seconds")
        attempts = (previous.get("counts") or {}).get("api_attempts")
        if previous.get("status") == "completed" and elapsed and attempts:
            throughput = attempts / elapsed
            break
    if throughput:
        print(f"Estimated minutes: {estimated_calls / throughput / 60:.2f} (prior-run throughput).")
    else:
        print("Estimated minutes: unavailable (no completed prior run throughput recorded).")
    if args.max_usd is not None:
        if args.input_price_per_million_tokens is None or args.output_price_per_million_tokens is None:
            print("Estimated USD: unavailable; supply both per-million-token prices.")
        else:
            estimate = (
                (input_bytes * args.input_price_per_million_tokens)
                + (estimated_calls * args.max_tokens * args.output_price_per_million_tokens)
            ) / 1_000_000
            print(f"Estimated USD upper bound: {estimate:.6f} (cap {args.max_usd:.6f}).")
    print(f"Configured model: {args.model or 'unset'}")
    print("No endpoint requests were made.")
    return 0


def main() -> int:
    args = _parse_args()
    if args.analyze:
        report = analyze_recordings(args.analyze)
        print(json.dumps(report, ensure_ascii=False, sort_keys=True, indent=2))
        return 0 if report["valid"] else 2

    args.base_url = (os.environ.get("LLM_BASE_URL") or "").strip()
    args.model = (os.environ.get("LLM_MODEL") or "").strip()
    try:
        tasks = _selected_tasks(args)
    except ValueError as exc:
        print(f"Configuration error: {exc}")
        return 2

    if args.dry_run:
        return _dry_run(args, tasks)
    if not args.live:
        print("No requests made. Use --dry-run to estimate or --live to run against LLM_BASE_URL.")
        return 0
    load_dotenv(args.env_file)
    args.base_url = (os.environ.get("LLM_BASE_URL") or "").strip()
    args.model = (os.environ.get("LLM_MODEL") or "").strip()
    if not args.base_url or not args.model:
        print("Live runs require LLM_BASE_URL and LLM_MODEL; no endpoint has a default.")
        return 2
    arms = tuple(part.strip() for part in args.arms.split(",") if part.strip())
    config = RunConfig(
        base_url=args.base_url,
        model=args.model,
        output_root=args.output_root,
        arms=arms,
        trials=args.trials,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        max_calls=args.max_calls,
        max_minutes=args.max_minutes,
        max_usd=args.max_usd,
        input_price_per_million_tokens=args.input_price_per_million_tokens,
        output_price_per_million_tokens=args.output_price_per_million_tokens,
        seed=args.seed,
        max_retries=args.max_retries,
    )
    try:
        result = run_evaluation(config, tasks)
    except (OSError, ValueError) as exc:
        print(f"Run could not start: {exc}")
        return 2
    print(f"Run directory: {result.run_dir}")
    print(f"Run status: {result.manifest['status']}")
    print(f"Logical calls: {result.manifest['counts']['logical_calls']}")
    print(f"API attempts: {result.manifest['counts']['api_attempts']}")
    if result.manifest["invalid_reasons"]:
        print("Run invalid: " + "; ".join(result.manifest["invalid_reasons"]))
        return 2
    print("Analysis label: oracle-labelled, not human-validated.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
