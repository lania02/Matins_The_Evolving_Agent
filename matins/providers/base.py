"""LLM and Search provider abstractions + factories (DESIGN.md section 11).

The core depends only on these Protocols, never on a concrete vendor. Concrete
adapters live alongside this file (anthropic.py, openai.py, openai_compatible.py,
search_web.py) and are imported lazily by the factories so that an unused
provider's optional dependencies never need to be installed.
"""
from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

from ..config import Config


@runtime_checkable
class LLMProvider(Protocol):
    def generate(self, prompt: str, *, temperature: float,
                 json_schema: dict | None = None) -> str:
        """Return the model's text completion for `prompt`.

        `json_schema` is an optional hint: adapters with native JSON modes may use
        it; others ignore it and rely on the prompt asking for JSON (tolerant
        parsing happens in generate/schema.py).
        """
        ...


@runtime_checkable
class SearchProvider(Protocol):
    def search(self, query: str, *, k: int = 5) -> list[dict]:
        """Return up to k results as dicts with keys: title, url, snippet."""
        ...


def get_llm_provider(cfg: Config, model: str | None = None) -> LLMProvider:
    """Instantiate the configured LLM adapter.

    `model` optionally overrides cfg.provider.model (used by the deep-dive, which
    runs on a stronger model than the daily generation) without mutating cfg.
    """
    if model and model != cfg.provider.model:
        cfg = dataclasses.replace(cfg, provider=dataclasses.replace(cfg.provider, model=model))
    name = cfg.provider.name
    if name == "anthropic":
        from .anthropic import AnthropicProvider
        return AnthropicProvider(cfg)
    if name == "openai":
        from .openai import OpenAIProvider
        return OpenAIProvider(cfg)
    if name == "openai_compatible":
        from .openai_compatible import OpenAICompatibleProvider
        return OpenAICompatibleProvider(cfg)
    raise ValueError(f"unknown LLM provider: {name!r}")


def get_search_provider(cfg: Config) -> SearchProvider | None:
    """Instantiate the configured search adapter, or None when disabled.

    Returning None makes the novelty step a no-op that flags ideas as
    prior_art='[unchecked]' (DESIGN.md section 7).
    """
    name = cfg.novelty.search_provider
    if name in ("none", "", None):
        return None
    from .search_web import get_search_provider as _factory
    return _factory(cfg)
