"""Run a small shadow evaluation using a real agent adapter and produce transcripts for human adjudication.

This script does not compute ground-truth leak metrics automatically because real
agents require an independent oracle or human labels. Instead it runs the same
task set, queries the chosen provider in either 'card' or 'full' context, runs
the sentence gate (if requested), and writes per-task transcripts and a small
summary that is useful to kick off human adjudication.

Example:
  python scripts/run_real_agent_suite.py --provider huggingface --model gpt2
      --mode card --personas 20 --trials 3 --outdir results/real_agent_shadow
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from matchlab.evals.personas import make_personas, make_pool
from matchlab.evals.tasks import build_tasks
from matchlab.evals.real_agent import RealAgentAdapter
from matchlab.evals.personas import compile_card
from matchlab.guards.budget import AnonymityPool, DisclosureBudget
from matchlab.guards.gate import SentenceGate
from matchlab.schemas import Stage


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--provider", default="mock", help="Agent provider: mock|huggingface")
    ap.add_argument("--model", default="", help="Provider-specific model name (e.g. 'gpt2' or HF model id)")
    ap.add_argument("--mode", choices=("card", "full"), default="card",
                    help="Context to send the agent: card (speakable forms) or full (persona text)")
    ap.add_argument("--personas", type=int, default=20)
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--card-error", type=float, default=0.0)
    ap.add_argument("--k-min", type=int, default=20)
    ap.add_argument("--outdir", default="results/real_agent_shadow")
    a = ap.parse_args()

    outdir = Path(a.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    tasks = build_tasks(make_personas(a.personas, seed=0))
    pool = make_pool(20000, seed=0)

    transcripts = []
    summary = {"n_tasks": len(tasks), "trials": a.trials, "provider": a.provider, "model": a.model, "mode": a.mode}

    for i, task in enumerate(tasks):
        for t in range(a.trials):
            persona = task.persona
            card = compile_card(persona, a.card_error, seed=0)
            if a.mode == "card":
                adapter = RealAgentAdapter(provider=a.provider, model=a.model, card=card)
            else:
                # build a small persona text (avoid saving raw private values in outputs)
                persona_text = f"{persona.display_name} - short profile: {persona.short_summary}"
                adapter = RealAgentAdapter(provider=a.provider, model=a.model, persona_text=persona_text)

            # optional gate in front of the agent replies: we always run it so humans can see deflections
            gate = SentenceGate(card, Stage.BLIND, DisclosureBudget(pool, a.k_min),
                                viewer_id=f"v:{task.task_id}:{t}")

            q = task.turns[0] if task.turns else "Tell me about him."
            try:
                replies = adapter.respond(q, task.force_mode)
            except Exception as exc:
                # Real provider calls may fail offline; this keeps the pipeline usable.
                if a.provider == "huggingface":
                    adapter = RealAgentAdapter(provider="mock", model="", card=card)
                    replies = adapter.respond(q, task.force_mode)
                    transcripts.append({"task_id": task.task_id, "trial": t, "warning": str(exc), "provider_fallback": "mock"})
                else:
                    transcripts.append({"task_id": task.task_id, "trial": t, "error": str(exc)})
                    continue

            sent_records = []
            for s in replies:
                res = gate.check(s.text, q, s.declared or None)
                sent_records.append({
                    "text": s.text,
                    "gate_action": res.action,
                    "gate_reasons": res.reasons,
                })

            transcripts.append({
                "task_id": task.task_id,
                "trial": t,
                "question": q,
                "replies": sent_records,
            })

    # write transcripts JSONL
    out_file = outdir / "transcripts.jsonl"
    with out_file.open("w", encoding="utf-8") as fh:
        for rec in transcripts:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # small summary
    json.dump(summary, open(outdir / "summary.json", "w"), indent=2)
    print(f"Wrote {len(transcripts)} transcripts to {out_file}")
    print("Next: have two human raters annotate transcripts.jsonl with the card-tier template and a leak judgement.")


if __name__ == "__main__":
    main()

