from __future__ import annotations

import html
import ipaddress
import json
import os
import re
import urllib.error
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from typing import Any, Mapping
from urllib.parse import urlsplit

OUTCOMES = frozenset(
    {
        "ok",
        "auth_error",
        "model_unavailable",
        "bad_request",
        "rate_limited",
        "credits_exhausted",
        "server_error",
        "timeout",
        "connection_error",
        "bad_response",
        "length_truncated",
        "empty_output",
    }
)
RETRYABLE_OUTCOMES = frozenset({"rate_limited", "server_error", "timeout"})
ABORT_OUTCOMES = frozenset(
    {
        "auth_error",
        "model_unavailable",
        "bad_request",
        "credits_exhausted",
        "connection_error",
    }
)


def _parse_json(raw: str) -> Any | None:
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return None


def validate_base_url(value: str) -> str:
    base_url = value.strip().rstrip("/")
    parts = urlsplit(base_url)
    if (
        parts.scheme not in {"http", "https"}
        or not parts.hostname
        or not parts.path.rstrip("/").endswith("/v1")
        or parts.username is not None
        or parts.password is not None
        or parts.query
        or parts.fragment
    ):
        raise ValueError("LLM_BASE_URL must be an http(s) OpenAI-compatible /v1 URL without credentials")
    return base_url


def _is_huggingface_host(base_url: str) -> bool:
    hostname = (urlsplit(base_url).hostname or "").lower().rstrip(".")
    return hostname == "huggingface.co" or hostname.endswith(".huggingface.co")


def _is_loopback_host(base_url: str) -> bool:
    hostname = (urlsplit(base_url).hostname or "").lower()
    if hostname == "localhost":
        return True
    try:
        return ipaddress.ip_address(hostname).is_loopback
    except ValueError:
        return False


def resolve_api_key(base_url: str, env: Mapping[str, str] | None = None) -> str | None:
    values = os.environ if env is None else env
    configured = (values.get("LLM_API_KEY") or "").strip()
    if configured:
        return configured
    fallback = (values.get("HF_TOKEN") or "").strip()
    if fallback and _is_huggingface_host(base_url):
        return fallback
    if not _is_loopback_host(base_url):
        raise ValueError("LLM_API_KEY is required for the configured non-local host")
    return None


def classify_outcome(status: int | None, body: str | None) -> str:
    """Map one HTTP completion response to the fixed Section 12 outcome taxonomy."""
    if status is not None and status >= 400 and body:
        if re.search(r"(?is)<(?:!doctype\s+html|html|head|body)\b", body):
            return "bad_response"
    if status in (401, 403):
        return "auth_error"
    if status == 402:
        return "credits_exhausted"
    if status == 404:
        return "model_unavailable"
    if status == 429:
        return "rate_limited"
    if status == 400 or (status is not None and 400 <= status < 500):
        return "bad_request"
    if status is not None and status >= 500:
        return "server_error"
    if status is not None and 300 <= status < 400:
        return "bad_response"
    if status is None:
        return "timeout" if body and "timeout" in body.lower() else "connection_error"
    if status != 200:
        return "bad_response"

    payload = _parse_json(body or "")
    if not isinstance(payload, dict):
        return "bad_response"
    choices = payload.get("choices")
    if not isinstance(choices, list) or not choices or not isinstance(choices[0], dict):
        return "bad_response"
    first = choices[0]
    if first.get("finish_reason") == "length":
        return "length_truncated"
    message = first.get("message")
    content = message.get("content") if isinstance(message, dict) else None
    if not isinstance(content, str) or not content.strip():
        return "empty_output"
    if first.get("finish_reason") not in (None, "stop"):
        return "bad_response"
    return "ok"


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    try:
        return max(0.0, float(value))
    except ValueError:
        try:
            retry_at = parsedate_to_datetime(value)
            if retry_at.tzinfo is None:
                retry_at = retry_at.replace(tzinfo=timezone.utc)
            return max(0.0, (retry_at - datetime.now(timezone.utc)).total_seconds())
        except (TypeError, ValueError, OverflowError):
            return None


def _safe_error_message(body: str, secret: str | None) -> str:
    value = html.unescape(re.sub(r"<[^>]*>", " ", body or ""))
    value = OpenAICompatibleAdapter.redact(value, secret)
    value = re.sub(r"(?i)\b(?:authorization\s*[:=]\s*)[^\s,;]+", "Authorization: [REDACTED]", value)
    value = re.sub(
        r"(?i)(https?://)[^/\s:@]+:[^@\s/]+@",
        r"\1[REDACTED]@",
        value,
    )
    value = re.sub(r"(?i)([?&](?:api[_-]?key|token|access_token)=)[^&\s]+", r"\1[REDACTED]", value)
    value = " ".join(value.split())
    if not value:
        return "empty error response"
    if "<html" in body.lower() or "<!doctype html" in body.lower():
        return "non-JSON error page"
    return value[:300]


class CompletionError(RuntimeError):
    def __init__(
        self,
        outcome: str,
        message: str,
        *,
        status: int | None = None,
        retry_after_s: float | None = None,
    ) -> None:
        super().__init__(f"completion failed: {outcome}: {message}")
        self.outcome = outcome
        self.status = status
        self.retry_after_s = retry_after_s


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(
        self,
        req: urllib.request.Request,
        fp: Any,
        code: int,
        msg: str,
        headers: Any,
        newurl: str,
    ) -> None:
        return None


@dataclass(frozen=True)
class _HttpResponse:
    status: int | None
    body: str
    headers: Mapping[str, str]


class OpenAICompatibleAdapter:
    """Standard-library OpenAI-compatible client for the configured endpoint only."""

    def __init__(
        self,
        base_url: str | None = None,
        api_key: str | None = None,
        model: str | None = None,
        timeout_s: float | None = None,
    ) -> None:
        configured_url = base_url or os.environ.get("LLM_BASE_URL", "")
        if not configured_url.strip():
            raise ValueError("LLM_BASE_URL is required; no default host is configured")
        self.base_url = validate_base_url(configured_url)
        self.model = (model if model is not None else os.environ.get("LLM_MODEL", "")).strip()
        if not self.model:
            raise ValueError("LLM_MODEL is required")
        self.api_key = (api_key or "").strip() or resolve_api_key(self.base_url)
        raw_timeout = timeout_s if timeout_s is not None else os.environ.get("LLM_TIMEOUT_S", "120")
        self.timeout_s = float(raw_timeout)
        if self.timeout_s <= 0:
            raise ValueError("LLM_TIMEOUT_S must be positive")

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        return headers

    def _request(self, payload: dict[str, Any]) -> _HttpResponse:
        url = f"{self.base_url}/chat/completions"
        req = urllib.request.Request(
            url,
            data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
            method="POST",
            headers=self._build_headers(),
        )
        opener = urllib.request.build_opener(urllib.request.ProxyHandler({}), _NoRedirect())
        try:
            with opener.open(req, timeout=self.timeout_s) as response:
                raw = response.read().decode("utf-8", errors="replace")
                return _HttpResponse(response.status, raw, dict(response.headers.items()))
        except urllib.error.HTTPError as exc:
            raw = exc.read(4096).decode("utf-8", errors="replace")
            return _HttpResponse(exc.code, raw, dict(exc.headers.items()) if exc.headers else {})
        except TimeoutError:
            return _HttpResponse(None, "timeout", {})
        except urllib.error.URLError as exc:
            reason = getattr(exc, "reason", "")
            if isinstance(reason, TimeoutError):
                return _HttpResponse(None, "timeout", {})
            return _HttpResponse(None, "connection_error", {})
        except OSError:
            return _HttpResponse(None, "connection_error", {})

    def complete(
        self,
        messages: list[dict[str, str]],
        *,
        temperature: float = 0.7,
        max_tokens: int = 150,
    ) -> dict[str, Any]:
        payload = {
            "model": self.model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        response = self._request(payload)
        outcome = classify_outcome(response.status, response.body)
        if outcome != "ok":
            message = _safe_error_message(response.body, self.api_key)
            if response.status is None and response.body in {"timeout", "connection_error"}:
                message = response.body
            raise CompletionError(
                outcome,
                message,
                status=response.status,
                retry_after_s=_retry_after_seconds(
                    next(
                        (
                            value
                            for key, value in response.headers.items()
                            if key.casefold() == "retry-after"
                        ),
                        None,
                    )
                ),
            )
        parsed = _parse_json(response.body)
        if not isinstance(parsed, dict):
            raise CompletionError("bad_response", "invalid JSON response", status=response.status)
        return parsed

    @staticmethod
    def redact(value: str, secret: str | None = None) -> str:
        text = value or ""
        if secret:
            text = text.replace(secret, "[REDACTED]")
        text = re.sub(
            r"(?i)(authorization\s*[:=]\s*bearer\s+)[^\s\"'<>]+",
            r"\1[REDACTED]",
            text,
        )
        text = re.sub(r"(?i)\bbearer\s+[^\s\"'<>]+", "Bearer [REDACTED]", text)
        text = re.sub(r"\bhf_[A-Za-z0-9]{20,}\b", "[REDACTED]", text)
        text = re.sub(
            r"(?i)(https?://)[^/\s:@]+:[^@\s/]+@",
            r"\1[REDACTED]@",
            text,
        )
        return re.sub(
            r"(?i)([?&](?:api[_-]?key|token|access_token)=)[^&\s]+",
            r"\1[REDACTED]",
            text,
        )


__all__ = [
    "ABORT_OUTCOMES",
    "OUTCOMES",
    "RETRYABLE_OUTCOMES",
    "CompletionError",
    "OpenAICompatibleAdapter",
    "classify_outcome",
    "resolve_api_key",
    "validate_base_url",
]
