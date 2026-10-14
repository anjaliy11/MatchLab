from __future__ import annotations

import csv
import json
import os
import sys
import tempfile
import threading
import time
import urllib.error
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any
from unittest.mock import Mock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from matchlab.adapters.openai_compat import (
    CompletionError,
    OpenAICompatibleAdapter,
    classify_outcome,
    resolve_api_key,
)
from matchlab.adapters.prompts import build_prompt_messages, to_chat_messages
from matchlab.evals.analysis import analyze_recordings, cohen_kappa
from matchlab.evals.oracle import GroundTruthOracle
from matchlab.evals.personas import compile_card, make_personas
from matchlab.evals.real_agent import RunConfig, load_dotenv, run_evaluation
from matchlab.evals.tasks import build_tasks


class FakeEndpoint:
    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.status = 200
        self.delay_s = 0.0
        self.redirect_url: str | None = None
        self.response_body: bytes | None = None
        self.status_sequence: list[int] = []
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def do_POST(self) -> None:
                length = int(self.headers.get("Content-Length", "0"))
                payload = json.loads(self.rfile.read(length))
                authorization = self.headers.get("Authorization")
                owner.requests.append(
                    {
                        "authorization": authorization,
                        "authorization_matches_configured_key": authorization == "Bearer local-dummy",
                        "payload": payload,
                    }
                )
                if owner.delay_s:
                    import time

                    time.sleep(owner.delay_s)
                if owner.redirect_url:
                    self.send_response(302)
                    self.send_header("Location", owner.redirect_url)
                    self.end_headers()
                    return
                status = owner.status_sequence.pop(0) if owner.status_sequence else owner.status
                if status != 200:
                    self.send_response(status)
                    self.send_header("Content-Type", "application/json")
                    if status == 429:
                        self.send_header("Retry-After", "0")
                    self.end_headers()
                    self.wfile.write(owner.response_body or b'{"error":{"message":"synthetic failure"}}')
                    return
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                if owner.response_body is not None:
                    try:
                        self.wfile.write(owner.response_body)
                    except OSError:
                        pass
                    return
                body = {
                    "id": "fake-completion",
                    "model": "fake-model",
                    "system_fingerprint": "fake-digest",
                    "choices": [
                        {
                            "finish_reason": "stop",
                            "message": {"role": "assistant", "content": "A thoughtful person."},
                        }
                    ],
                    "usage": {"prompt_tokens": 12, "completion_tokens": 4, "total_tokens": 16},
                }
                try:
                    self.wfile.write(json.dumps(body).encode())
                except OSError:
                    pass

            def log_message(self, format: str, *args: Any) -> None:
                return

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.thread = threading.Thread(
            target=lambda: self.server.serve_forever(poll_interval=0.01), daemon=True
        )
        self.thread.start()
        self.base_url = f"http://127.0.0.1:{self.server.server_port}/v1"

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=2)


class RealAgentPipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.endpoint = FakeEndpoint()
        self.temp = tempfile.TemporaryDirectory()
        self.output_root = Path(self.temp.name) / "results" / "real_agent"

    def tearDown(self) -> None:
        self.endpoint.close()
        self.temp.cleanup()

    def test_taxonomy_covers_bad_request_and_all_contract_outcomes(self) -> None:
        self.assertEqual(classify_outcome(400, "{}"), "bad_request")
        self.assertEqual(classify_outcome(422, "{}"), "bad_request")
        self.assertEqual(classify_outcome(302, ""), "bad_response")
        self.assertEqual(classify_outcome(200, "<html>bad</html>"), "bad_response")
        self.assertEqual(classify_outcome(401, "<html>unauthorized</html>"), "bad_response")
        self.assertEqual(
            classify_outcome(200, '{"choices":[{"finish_reason":"length"}]}'),
            "length_truncated",
        )
        self.assertEqual(
            classify_outcome(
                200,
                '{"choices":[{"finish_reason":"stop","message":{"content":"  "}}]}',
            ),
            "empty_output",
        )

    def test_credentials_are_scoped_and_blank_values_do_not_shadow(self) -> None:
        self.assertIsNone(
            resolve_api_key(
                "http://localhost:11434/v1",
                {"LLM_API_KEY": " ", "HF_TOKEN": "hf_synthetic_not_a_real_token"},
            )
        )
        self.assertEqual(
            resolve_api_key(
                "https://router.huggingface.co/v1",
                {"LLM_API_KEY": " ", "HF_TOKEN": "hf_synthetic_not_a_real_token"},
            ),
            "hf_synthetic_not_a_real_token",
        )
        self.assertEqual(
            resolve_api_key(
                "https://example.invalid/v1",
                {"LLM_API_KEY": "local-dummy", "HF_TOKEN": "hf_synthetic_not_a_real_token"},
            ),
            "local-dummy",
        )

    def test_adapter_sends_key_only_to_configured_host_and_returns_usage(self) -> None:
        adapter = OpenAICompatibleAdapter(
            self.endpoint.base_url, api_key="local-dummy", model="fake-model", timeout_s=2
        )
        result = adapter.complete([{"role": "user", "content": "Reply briefly."}])
        self.assertEqual(self.endpoint.requests[0]["authorization"], "Bearer local-dummy")
        self.assertEqual(result["usage"]["total_tokens"], 16)
        self.assertEqual(result["model"], "fake-model")

    def test_adapter_raises_sanitized_typed_error_for_http_failure(self) -> None:
        self.endpoint.status = 401
        adapter = OpenAICompatibleAdapter(
            self.endpoint.base_url, api_key="private-test-key", model="fake-model", timeout_s=2
        )
        with self.assertRaises(CompletionError) as raised:
            adapter.complete([{"role": "user", "content": "hello"}])
        self.assertEqual(raised.exception.outcome, "auth_error")
        self.assertNotIn("private-test-key", str(raised.exception))

    def test_fake_endpoint_maps_contract_http_statuses(self) -> None:
        adapter = OpenAICompatibleAdapter(
            self.endpoint.base_url,
            api_key="local-dummy",
            model="fake-model",
            timeout_s=2,
        )
        for status, expected in (
            (402, "credits_exhausted"),
            (404, "model_unavailable"),
            (400, "bad_request"),
            (429, "rate_limited"),
            (500, "server_error"),
        ):
            self.endpoint.status = status
            with self.assertRaises(CompletionError) as raised:
                adapter.complete([{"role": "user", "content": "hello"}])
            self.assertEqual(raised.exception.outcome, expected)

    def test_runner_retries_rate_limit_then_scores_only_success(self) -> None:
        persona = make_personas(1, seed=3)[0]
        task = next(row for row in build_tasks([persona]) if row.scenario == "benign_hobby")
        self.endpoint.status_sequence = [200, 429, 200]
        result = run_evaluation(
            RunConfig(
                base_url=self.endpoint.base_url,
                model="fake-model",
                output_root=self.output_root,
                api_key="local-dummy",
                arms=("r1",),
                max_calls=3,
                max_minutes=1,
            ),
            [task],
        )
        manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(len(self.endpoint.requests), 3)
        self.assertEqual(manifest["counts"]["api_attempts"], 2)
        self.assertEqual(manifest["warmup"]["attempt_count"], 1)
        self.assertEqual(manifest["counts"]["attempt_outcomes"]["rate_limited"], 1)
        self.assertEqual(manifest["counts"]["outcomes_by_arm"]["R1_card_only"]["ok"], 1)

    def test_runner_aborts_after_nonretryable_auth_failure(self) -> None:
        persona = make_personas(1, seed=5)[0]
        task = next(row for row in build_tasks([persona]) if row.scenario == "benign_hobby")
        self.endpoint.status_sequence = [200, 401, 200]
        result = run_evaluation(
            RunConfig(
                base_url=self.endpoint.base_url,
                model="fake-model",
                output_root=self.output_root,
                api_key="secret-test-value",
                arms=("r1",),
                max_calls=10,
                max_minutes=1,
            ),
            [task],
        )
        manifest_text = (result.run_dir / "manifest.json").read_text(encoding="utf-8")
        self.assertEqual(len(self.endpoint.requests), 2)
        self.assertEqual(result.manifest["status"], "invalid_run")
        self.assertNotIn("secret-test-value", manifest_text)
        self.assertIn("auth_error", manifest_text)

    def test_adapter_exposes_retry_after_for_runner(self) -> None:
        self.endpoint.status_sequence = [429, 200]
        adapter = OpenAICompatibleAdapter(
            self.endpoint.base_url, api_key="local-dummy", model="fake-model", timeout_s=2
        )
        with self.assertRaises(CompletionError) as raised:
            adapter.complete([{"role": "user", "content": "hello"}])
        self.assertEqual(raised.exception.outcome, "rate_limited")
        self.assertEqual(raised.exception.retry_after_s, 0)
        self.assertEqual(len(self.endpoint.requests), 1)

    def test_adapter_maps_malformed_empty_and_truncated_success_bodies(self) -> None:
        adapter = OpenAICompatibleAdapter(
            self.endpoint.base_url, api_key="local-dummy", model="fake-model", timeout_s=2
        )
        cases = [
            (b"<html>not json</html>", "bad_response"),
            (b'{"choices":[{"finish_reason":"stop","message":{"content":""}}]}', "empty_output"),
            (b'{"choices":[{"finish_reason":"length","message":{"content":"x"}}]}', "length_truncated"),
        ]
        for body, expected in cases:
            self.endpoint.response_body = body
            with self.assertRaises(CompletionError) as raised:
                adapter.complete([{"role": "user", "content": "hello"}])
            self.assertEqual(raised.exception.outcome, expected)

    def test_adapter_classifies_timeout_and_refused_connection(self) -> None:
        self.endpoint.delay_s = 0.15
        adapter = OpenAICompatibleAdapter(
            self.endpoint.base_url, api_key="local-dummy", model="fake-model", timeout_s=0.02
        )
        with self.assertRaises(CompletionError) as raised:
            adapter.complete([{"role": "user", "content": "hello"}])
        self.assertEqual(raised.exception.outcome, "timeout")

        refused = OpenAICompatibleAdapter(
            "http://127.0.0.1:1/v1", model="fake-model", timeout_s=0.2
        )
        mock_opener = Mock()
        mock_opener.open.side_effect = urllib.error.URLError(
            ConnectionRefusedError("synthetic refusal")
        )
        with patch(
            "matchlab.adapters.openai_compat.urllib.request.build_opener",
            return_value=mock_opener,
        ):
            with self.assertRaises(CompletionError) as refused_error:
                refused.complete([{"role": "user", "content": "hello"}])
        self.assertEqual(refused_error.exception.outcome, "connection_error")

    def test_redaction_covers_bearer_tokens_and_credentials_in_urls(self) -> None:
        text = (
            "Authorization: Bearer private-key "
            "Bearer hf_abcdefghijklmnopqrstuvwxyz "
            "https://user:pass@example.invalid/v1?api_key=private-key"
        )
        cleaned = OpenAICompatibleAdapter.redact(text, "private-key")
        self.assertNotIn("private-key", cleaned)
        self.assertNotIn("abcdefghijklmnopqrstuvwxyz", cleaned)
        self.assertNotIn("user:pass", cleaned)

    def test_redirect_is_not_followed(self) -> None:
        other_endpoint = FakeEndpoint()
        self.endpoint.redirect_url = other_endpoint.base_url + "/chat/completions"
        adapter = OpenAICompatibleAdapter(
            self.endpoint.base_url, api_key="private-test-key", model="fake-model", timeout_s=2
        )
        with self.assertRaises(CompletionError) as raised:
            adapter.complete([{"role": "user", "content": "hello"}])
        self.assertEqual(raised.exception.outcome, "bad_response")
        self.assertEqual(len(self.endpoint.requests), 1)
        self.assertEqual(other_endpoint.requests, [])
        other_endpoint.close()

    def test_r1_prompt_has_only_blind_card_content_and_shared_instructions(self) -> None:
        persona = make_personas(1, seed=4)[0]
        tasks = build_tasks([persona])
        task = next(row for row in tasks if row.scenario == "direct_employer")
        r0 = build_prompt_messages(
            mode="r0", persona=persona, question=task.turns[0], temperature=0.7, max_tokens=150
        )
        r1 = build_prompt_messages(
            mode="r1", card=None, question=task.turns[0], temperature=0.7, max_tokens=150,
            context_text="A thoughtful person.",
        )
        self.assertEqual(r0["instruction"], r1["instruction"])
        r1_text = json.dumps(to_chat_messages(r1), ensure_ascii=False).casefold()
        for restricted in (persona.employer, persona.city, persona.school, persona.religion, *persona.descriptors, persona.devanagari, persona.bio):
            self.assertNotIn(restricted.casefold(), r1_text)
            for persona in make_personas(8, seed=9):
                card_prompt = build_prompt_messages(
                    mode="r1",
                    card=compile_card(persona, seed=9),
                    question="Tell me about this person.",
                )
                text = json.dumps(to_chat_messages(card_prompt), ensure_ascii=False).casefold()
                for restricted in (
                    persona.employer,
                    persona.city,
                    persona.school,
                    persona.religion,
                    *persona.descriptors,
                    persona.devanagari,
                    persona.bio,
                ):
                    self.assertNotIn(restricted.casefold(), text)

    def test_oracle_detects_held_out_descriptors_script_and_category_confirmation(self) -> None:
        persona = make_personas(1, seed=0)[0]
        oracle = GroundTruthOracle(persona)
        held_out = oracle.label_response(f"He works at {persona.descriptors[-1]}.")
        self.assertIn("employer", held_out["leaks"])
        devanagari = oracle.label_response(f"The place is {persona.devanagari}.")
        self.assertIn("employer", devanagari["leaks"])
        wrong_guess = oracle.label_response("Nahi, bilkul nahi.", question="Kya woh Muslim hai?")
        self.assertTrue(wrong_guess["confirmation"])
        benign = oracle.label_response("Yes, he likes chess.", question="Does he like chess?")
        self.assertFalse(benign["confirmation"])

    def test_live_runner_records_manifest_warmup_and_replays_cache(self) -> None:
        persona = make_personas(1, seed=0)[0]
        task = next(row for row in build_tasks([persona]) if row.scenario == "benign_hobby")
        config = RunConfig(
            base_url=self.endpoint.base_url,
            model="fake-model",
            output_root=self.output_root,
            api_key="local-dummy",
            arms=("r0", "r1"),
            trials=1,
            max_calls=3,
            max_minutes=1,
            seed=0,
        )
        first = run_evaluation(config, [task])
        self.assertEqual(len(self.endpoint.requests), 3)
        first_manifest = json.loads((first.run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(first_manifest["warmup"]["outcome"], "ok")
        self.assertEqual(first_manifest["counts"]["logical_calls"], 2)
        self.assertEqual(first_manifest["counts"]["api_attempts"], 2)
        self.assertEqual(first_manifest["warmup"]["attempt_count"], 1)
        self.assertEqual(first_manifest["host_kind"], "local")
        self.assertTrue((first.run_dir / "human_review.csv").exists())
        self.assertTrue((first.run_dir / "transcripts.jsonl").exists())
        first_analysis = json.dumps(analyze_recordings(first.run_dir), sort_keys=True)

        second = run_evaluation(config, [task])
        self.assertEqual(len(self.endpoint.requests), 3)
        self.assertNotEqual(first.run_dir, second.run_dir)
        second_manifest = json.loads((second.run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(second_manifest["counts"]["api_attempts"], 0)
        self.assertEqual(second_manifest["counts"]["cache_hits"], 2)
        self.assertEqual(first_analysis, json.dumps(analyze_recordings(first.run_dir), sort_keys=True))

    def test_max_calls_caps_warmup_and_scored_requests(self) -> None:
        persona = make_personas(1, seed=1)[0]
        tasks = [row for row in build_tasks([persona]) if row.kind == "benign"][:2]
        config = RunConfig(
            base_url=self.endpoint.base_url,
            model="fake-model",
            output_root=self.output_root,
            arms=("r1",),
            trials=1,
            max_calls=2,
            max_minutes=1,
            seed=0,
        )
        result = run_evaluation(config, tasks)
        self.assertEqual(len(self.endpoint.requests), 2)
        manifest = json.loads((result.run_dir / "manifest.json").read_text(encoding="utf-8"))
        self.assertEqual(manifest["status"], "invalid_run")
        self.assertIn("max-calls", manifest["invalid_reasons"][0])

    def test_environment_file_parsing_respects_existing_environment(self) -> None:
        path = Path(self.temp.name) / ".env"
        path.write_text(
            "\ufeffLLM_BASE_URL=http://localhost:11434/v1\n"
            "LLM_MODEL='synthetic model'\n"
            "IGNORED_LINE\n"
            "# comment\n",
            encoding="utf-8",
        )
        env = {"LLM_BASE_URL": "http://127.0.0.1:11434/v1"}
        load_dotenv(path, env)
        self.assertEqual(env["LLM_BASE_URL"], "http://127.0.0.1:11434/v1")
        self.assertEqual(env["LLM_MODEL"], "synthetic model")

    def test_analysis_reports_invalid_infra_and_is_byte_deterministic(self) -> None:
        run_dir = Path(self.temp.name) / "run"
        run_dir.mkdir()
        (run_dir / "transcripts.jsonl").write_text(
            '{"arm":"R0_full_profile","persona_id":"u0","trial":0,"outcome":"ok","oracle":{"leaks":[],"confirmation":false},"benign":true,"response":"Yes."}\n'
            '{"arm":"R1_card_only","persona_id":"u1","trial":0,"outcome":"rate_limited","oracle":{"leaks":[],"confirmation":false}}\n',
            encoding="utf-8",
        )
        report1 = json.dumps(analyze_recordings(run_dir), sort_keys=True, separators=(",", ":"))
        report2 = json.dumps(analyze_recordings(run_dir), sort_keys=True, separators=(",", ":"))
        self.assertEqual(report1.encode(), report2.encode())
        report = json.loads(report1)
        self.assertFalse(report["valid"])
        self.assertEqual(
            report["arms"]["R1_card_only"]["infra_failures"]["rate_limited"], 1
        )

    def test_analysis_excludes_infra_calls_from_valid_metric_denominators(self) -> None:
        run_dir = Path(self.temp.name) / "near-threshold"
        run_dir.mkdir()
        rows = [
            {
                "arm": "R1_card_only",
                "persona_id": f"u{i % 4}",
                "task_id": f"task-{i}",
                "trial": 0,
                "outcome": "ok" if i < 19 else "rate_limited",
                "oracle": {"leaks": [], "confirmation": False},
                "benign": False,
            }
            for i in range(20)
        ]
        (run_dir / "transcripts.jsonl").write_text(
            "".join(json.dumps(row) + "\n" for row in rows),
            encoding="utf-8",
        )
        report = analyze_recordings(run_dir)
        self.assertTrue(report["valid"])
        self.assertEqual(report["arms"]["R1_card_only"]["ok_calls"], 19)
        self.assertEqual(report["arms"]["R1_card_only"]["leaks"], 0)

    def test_cohen_kappa_for_human_review_labels(self) -> None:
        self.assertAlmostEqual(cohen_kappa(["leak", "safe", "leak"], ["leak", "safe", "safe"]), 0.4)
        self.assertIsNone(cohen_kappa(["leak"], ["leak"]))


if __name__ == "__main__":
    unittest.main()
