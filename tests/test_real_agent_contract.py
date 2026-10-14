from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matchlab.adapters.openai_compat import classify_outcome
from matchlab.adapters.prompts import build_prompt_messages, load_instruction_template
from matchlab.evals.analysis import analyze_recordings
from matchlab.evals.oracle import GroundTruthOracle
from matchlab.evals.personas import make_personas


def test_outcome_classification_maps_common_failures() -> None:
    assert classify_outcome(401, "") == "auth_error"
    assert classify_outcome(402, "") == "credits_exhausted"
    assert classify_outcome(404, "") == "model_unavailable"
    assert classify_outcome(429, "") == "rate_limited"
    assert classify_outcome(500, "") == "server_error"
    assert classify_outcome(576, "") == "server_error"
    assert classify_outcome(None, "timeout") == "timeout"
    assert classify_outcome(None, "") == "connection_error"
    assert classify_outcome(200, '{"choices":[{"finish_reason":"length"}]}') == "length_truncated"
    assert classify_outcome(200, '{"choices":[{"finish_reason":"stop","message":{"content":""}}]}') == "empty_output"
    assert classify_outcome(200, '{"choices":[{"finish_reason":"stop","message":{"content":"OK"}}]}') == "ok"


def test_prompt_templates_share_instruction_text() -> None:
    persona = make_personas(1, seed=0)[0]
    instruction = load_instruction_template()
    r0 = build_prompt_messages(mode="r0", persona=persona, question="Where does he work?", temperature=0.7, max_tokens=128)
    r1 = build_prompt_messages(mode="r1", card=None, question="Where does he work?", temperature=0.7, max_tokens=128, context_text="He is a thoughtful person.")
    assert r0["instruction"] == instruction
    assert r1["instruction"] == instruction
    assert r0["context"] != r1["context"]


def test_ground_truth_oracle_detects_leaks_and_confirmation() -> None:
    persona = make_personas(1, seed=0)[0]
    oracle = GroundTruthOracle(persona)
    label = oracle.label_response(
        f"He works at {persona.employer}.", question="Where does he work?"
    )
    assert "employer" in label["leaks"]
    assert label["matches"]["employer"] == persona.employer
    label2 = oracle.label_response("No.", question=f"Is it {persona.employer}?")
    assert label2["confirmation"] is True


def test_analysis_from_recordings_counts_outcomes() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        base = Path(tmp)
        recs = [
            {"outcome": "ok", "question": "Q1"},
            {"outcome": "rate_limited", "question": "Q2"},
            {"outcome": "ok", "question": "Q3"},
        ]
        (base / "transcripts.jsonl").write_text("\n".join(json.dumps(r) for r in recs) + "\n", encoding="utf-8")
        summary = analyze_recordings(base)
        assert summary["total_calls"] == 3
        assert summary["outcome_counts"]["ok"] == 2
        assert summary["outcome_counts"]["rate_limited"] == 1
        assert summary["infra_failures"] == 1


def test_run_real_agent_dry_run_does_not_make_calls() -> None:
    env = os.environ.copy()
    env["LLM_BASE_URL"] = "http://localhost:11434/v1"
    env["LLM_MODEL"] = "tinyllama"
    proc = subprocess.run(
        [sys.executable, "scripts/run_real_agent.py", "--dry-run", "--max-calls", "2"],
        cwd=str(Path(__file__).resolve().parents[1]),
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    out = proc.stdout.lower()
    assert "dry run" in out or "dry-run" in out
    assert "estimated" in out


if __name__ == "__main__":
    test_outcome_classification_maps_common_failures()
    test_prompt_templates_share_instruction_text()
    test_ground_truth_oracle_detects_leaks_and_confirmation()
    test_analysis_from_recordings_counts_outcomes()
    test_run_real_agent_dry_run_does_not_make_calls()
