"""Anthropic Messages API adapter (implements LLMProvider).

Talks to the Anthropic `/v1/messages` endpoint. The `json_schema` hint is
accepted for interface compatibility but ignored: callers rely on the prompt
asking for JSON and on tolerant parsing in generate/schema.py.
"""
from __future__ import annotations

import httpx

from ..config import Config

_DEFAULT_BASE = "https://api.anthropic.com"
_TIMEOUT = 120


class AnthropicProvider:
    """LLMProvider backed by the Anthropic Messages API."""

    def __init__(self, cfg: Config) -> None:
        self._key = cfg.api_key()
        self._model = cfg.provider.model
        self._base = (cfg.provider.base_url or _DEFAULT_BASE).rstrip("/")

    def generate(self, prompt: str, *, temperature: float,
                 json_schema: dict | None = None) -> str:
        if not self._key:
            raise RuntimeError("AnthropicProvider: missing API key")

        url = self._base + "/v1/messages"
        headers = {
            "x-api-key": self._key,
            "anthropic-version": "2023-06-01",
            "content-type": "application/json",
        }
        payload = {
            "model": self._model,
            "max_tokens": 4096,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = httpx.post(url, headers=headers, json=payload, timeout=_TIMEOUT)
        if not (200 <= resp.status_code < 300):
            raise RuntimeError(
                f"Anthropic API error {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()
        blocks = data.get("content") or []
        return "".join(
            b.get("text", "") for b in blocks if b.get("type") == "text"
        )
