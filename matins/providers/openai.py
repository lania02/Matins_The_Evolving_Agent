"""OpenAI Chat Completions adapter (implements LLMProvider).

Talks to the OpenAI `/chat/completions` endpoint. The `json_schema` hint is
accepted for interface compatibility but ignored: callers rely on the prompt
asking for JSON and on tolerant parsing in generate/schema.py.
"""
from __future__ import annotations

import httpx

from ..config import Config

_DEFAULT_BASE = "https://api.openai.com/v1"
_TIMEOUT = 120


class OpenAIProvider:
    """LLMProvider backed by the OpenAI Chat Completions API."""

    def __init__(self, cfg: Config) -> None:
        self._key = cfg.api_key()
        self._model = cfg.provider.model
        self._base = (cfg.provider.base_url or _DEFAULT_BASE).rstrip("/")

    def generate(self, prompt: str, *, temperature: float,
                 json_schema: dict | None = None) -> str:
        if not self._key:
            raise RuntimeError("OpenAIProvider: missing API key")

        url = self._base + "/chat/completions"
        headers = {
            "Authorization": "Bearer " + self._key,
            "content-type": "application/json",
        }
        payload = {
            "model": self._model,
            "temperature": temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        resp = httpx.post(url, headers=headers, json=payload, timeout=_TIMEOUT)
        if not (200 <= resp.status_code < 300):
            raise RuntimeError(
                f"OpenAI API error {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()
        return data["choices"][0]["message"]["content"]
