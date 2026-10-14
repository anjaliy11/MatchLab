"""Loopback-only HTTP staging adapter for the deterministic sentence gate.

This module is for local synthetic testing only. It is not an authenticated,
concurrent, persistent, or production service.
"""
from __future__ import annotations

import ipaddress
import json
import logging
from http.server import BaseHTTPRequestHandler, HTTPServer
from typing import Any

from .evals.personas import compile_card, make_personas
from .guards.gate import DEFLECTIONS, SentenceGate
from .schemas import DisclosureCard, Stage

MAX_REQUEST_BYTES = 16 * 1024
MAX_TEXT_CHARS = 8 * 1024
_LOG = logging.getLogger(__name__)
_FAIL_CLOSED_TEXT = DEFLECTIONS[0]


class _LoopbackHTTPServer(HTTPServer):
    def get_request(self):
        request, address = super().get_request()
        request.settimeout(5.0)
        return request, address


def create_local_gate_server(
    card: DisclosureCard,
    *,
    host: str = "127.0.0.1",
    port: int = 8765,
) -> HTTPServer:
    """Create a bounded, single-threaded server bound to IPv4 loopback only."""
    try:
        is_loopback = ipaddress.ip_address(host).is_loopback
    except ValueError:
        is_loopback = False
    if host != "127.0.0.1" or not is_loopback:
        raise ValueError("local staging may bind only to 127.0.0.1")
    if isinstance(port, bool) or not isinstance(port, int) or not 0 <= port <= 65535:
        raise ValueError("port must be an integer between 0 and 65535")

    class GateRequestHandler(BaseHTTPRequestHandler):
        server_version = "matchlab-local-staging"
        sys_version = ""

        def log_message(self, format: str, *args: Any) -> None:
            return

        def _send_json(self, status: int, payload: dict[str, Any]) -> None:
            body = json.dumps(payload, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
            self.send_response(status)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("Connection", "close")
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self) -> None:
            if self.path != "/health":
                self._send_json(404, {"error": "not_found"})
                return
            self._send_json(200, {"status": "ok", "mode": "local_staging"})

        def do_POST(self) -> None:
            if self.path != "/v1/check":
                self._send_json(404, {"error": "not_found"})
                return
            if self.headers.get("Transfer-Encoding") is not None:
                self._send_json(400, {"error": "invalid_request"})
                return
            if self.headers.get("Content-Type", "").split(";", 1)[0].strip().casefold() != "application/json":
                self._send_json(415, {"error": "unsupported_media_type"})
                return
            try:
                length = int(self.headers.get("Content-Length", ""))
            except ValueError:
                self._send_json(411, {"error": "content_length_required"})
                return
            if length < 0:
                self._send_json(400, {"error": "invalid_request"})
                return
            if length > MAX_REQUEST_BYTES:
                self._send_json(413, {"error": "request_too_large"})
                return
            raw_body = self.rfile.read(length)
            if len(raw_body) != length:
                self._send_json(400, {"error": "invalid_request"})
                return
            try:
                request = json.loads(raw_body.decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError, RecursionError):
                self._send_json(400, {"error": "invalid_request"})
                return
            if not isinstance(request, dict) or set(request) - {"sentence", "question", "stage"}:
                self._send_json(400, {"error": "invalid_request"})
                return

            sentence = request.get("sentence")
            question = request.get("question")
            stage_name = request.get("stage", "blind")
            if (
                not isinstance(sentence, str)
                or not sentence
                or len(sentence) > MAX_TEXT_CHARS
                or (question is not None and (
                    not isinstance(question, str) or len(question) > MAX_TEXT_CHARS
                ))
                or stage_name not in ("blind", "mutual")
            ):
                self._send_json(400, {"error": "invalid_request"})
                return
            try:
                sentence.encode("utf-8")
                if question is not None:
                    question.encode("utf-8")
            except UnicodeEncodeError:
                self._send_json(400, {"error": "invalid_request"})
                return

            stage = Stage.BLIND if stage_name == "blind" else Stage.MUTUAL
            try:
                result = SentenceGate(card, stage).check(sentence, question)
            except Exception as exc:
                _LOG.error("gate failed closed (exception_type=%s)", type(exc).__name__)
                self._send_json(
                    503,
                    {"action": "rewrite", "reasons": ["gate_error"], "text": _FAIL_CLOSED_TEXT},
                )
                return
            self._send_json(
                200,
                {"action": result.action, "reasons": result.reasons, "text": result.text},
            )

    return _LoopbackHTTPServer((host, port), GateRequestHandler)


def create_synthetic_staging_server(seed: int, port: int = 8765) -> HTTPServer:
    """Create a local gate server using one seeded synthetic persona."""
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")
    persona = make_personas(1, seed=seed)[0]
    card = compile_card(persona, error_rate=0.0, seed=seed)
    return create_local_gate_server(card, port=port)
