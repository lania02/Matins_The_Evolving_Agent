"""OpenAI-compatible adapter for local / self-hosted servers (implements LLMProvider).

Same wire format as the OpenAI Chat Completions API, but for servers that expose
an OpenAI-compatible `/chat/completions` endpoint at a user-supplied base URL.
This covers, among others:

  * Ollama       (http://localhost:11434/v1)   -- no API key required
  * vLLM         (http://localhost:8000/v1)     -- key optional
  * LM Studio    (http://localhost:1234/v1)     -- key optional

`base_url` is REQUIRED. The API key is OPTIONAL: the Authorization header is only
sent when a key is configured (Ollama and friends need none). The `json_schema`
hint is accepted for interface compatibility but ignored.
"""
from __future__ import annotations

import httpx

from ..config import Config

_TIMEOUT = 120


class OpenAICompatibleProvider:
    """LLMProvider for OpenAI-compatible servers (Ollama / vLLM / LM Studio)."""

    def __init__(self, cfg: Config) -> None:
        if not cfg.provider.base_url:
            raise RuntimeError(
                "OpenAICompatibleProvider: base_url is required"
            )
        self._key = cfg.api_key()
        self._model = cfg.provider.model
        self._base = cfg.provider.base_url.rstrip("/")

    def generate(self, prompt: str, *, temperature: float,
                 json_schema: dict | None = None) -> str:
        url = self._base + "/chat/completions"
        headers = {"content-type": "application/json"}
        if self._key:
            headers["Authorization"] = "Bearer " + self._key
        payload = {
            "model": self._model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = httpx.post(url, headers=headers, json=payload, timeout=_TIMEOUT)
        if not (200 <= resp.status_code < 300):
            raise RuntimeError(
                f"OpenAI-compatible API error {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()
        return data["choices"][0]["message"]["content"]
