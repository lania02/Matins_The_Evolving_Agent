"""Offline tests for novelty search-provider routing (instantiation only, no network)."""
from __future__ import annotations

from matins.config import Config, NoveltyCfg
from matins.providers.base import get_search_provider
from matins.providers.search_web import (
    ArxivSearchProvider,
    OpenAlexSearchProvider,
    WebSearchProvider,
)


def _cfg(provider: str) -> Config:
    return Config(novelty=NoveltyCfg(search_provider=provider))


def test_default_novelty_provider_is_openalex() -> None:
    # The shipped default must be the reliable, cross-domain scholarly source -- NOT the
    # DuckDuckGo scraper, which silently rate-limits and degrades novelty to "[no prior art]".
    default = NoveltyCfg().search_provider
    assert isinstance(get_search_provider(_cfg(default)), OpenAlexSearchProvider)


def test_openalex_routes_to_openalex_not_web() -> None:
    # Regression: 'openalex' must NOT fall through to the web adapter (the bug that an
    # arxiv-only `if` would have hidden when the default was switched to openalex).
    assert isinstance(get_search_provider(_cfg("openalex")), OpenAlexSearchProvider)


def test_arxiv_web_and_none_routing() -> None:
    assert isinstance(get_search_provider(_cfg("arxiv")), ArxivSearchProvider)
    assert isinstance(get_search_provider(_cfg("web")), WebSearchProvider)
    assert get_search_provider(_cfg("none")) is None        # disabled -> novelty is a no-op
