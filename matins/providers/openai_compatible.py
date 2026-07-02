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

# Short connect (fail fast if the endpoint is down) but a long read: the on-demand
# deep dive stuffs up to ~40 retrieved sources into one prompt and asks for a long
# structured briefing, and reasoning-tier models (minimax-m3) can think for minutes
# on a full generation prompt. A 300s read ceiling was tripping both.
_TIMEOUT = httpx.Timeout(600.0, connect=15.0)


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
            # Some gateways (e.g. NVIDIA's for minimax-m3) return HTTP 200 with an EMPTY
            # choices list when max_tokens is omitted, instead of applying a default.
            # Always send a generous cap so every model on the roster behaves.
            "max_tokens": 8192,
            "messages": [{"role": "user", "content": prompt}],
        }
        try:
            resp = httpx.post(url, headers=headers, json=payload, timeout=_TIMEOUT)
        except httpx.HTTPError as exc:
            # Map transport failures (read timeout, connect error, ...) to RuntimeError so
            # the slot-retry loop treats them as a failed ATTEMPT. A raw httpx.ReadTimeout
            # is not in the loop's except clause and killed a whole batch (seen live with
            # minimax-m3 thinking past the read ceiling).
            raise RuntimeError(f"OpenAI-compatible transport error: {exc!r}") from exc
        if not (200 <= resp.status_code < 300):
            raise RuntimeError(
                f"OpenAI-compatible API error {resp.status_code}: {resp.text[:500]}"
            )
        data = resp.json()
        choices = data.get("choices") or []
        if not choices:
            # Fail loud with the body head -- an empty-choices 200 otherwise surfaces as a
            # bare IndexError that hides which endpoint/model misbehaved.
            raise RuntimeError(
                f"OpenAI-compatible API returned no choices (model={self._model}): "
                f"{resp.text[:300]}"
            )
        return choices[0]["message"]["content"]
