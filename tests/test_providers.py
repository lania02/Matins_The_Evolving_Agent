"""Offline tests for novelty search-provider routing (instantiation only, no network)."""
from __future__ import annotations

from matins.config import Config, DeepDiveCfg, NoveltyCfg, ProviderCfg
from matins.providers.base import get_dig_llm_provider, get_search_provider
from matins.providers.openai_compatible import OpenAICompatibleProvider
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


def test_reddit_routes_as_retrieval_and_demand_source() -> None:
    # 'reddit' must resolve for both the daily feed and verify.demand_source (same factory).
    from matins.providers.search_web import RedditSearchProvider, get_retrieval_searcher
    p = get_retrieval_searcher("reddit", _cfg("arxiv"))
    assert isinstance(p, RedditSearchProvider)


def test_dig_provider_override_routes_to_its_own_endpoint() -> None:
    # End-to-end wiring (config -> factory -> adapter): a deep_dive provider override
    # builds the dig LLM against THAT vendor's base_url/model, while the main provider
    # (here a different vendor entirely) is never consulted for the dig.
    cfg = Config(
        provider=ProviderCfg(name="anthropic", model="main"),     # NOT used for the dig
        deep_dive=DeepDiveCfg(provider_name="openai_compatible",
                              base_url="https://apihub.agnes-ai.com/v1",
                              api_key_env="AGNES_API_KEY", model="agnes-2.0-flash"),
    )
    prov = get_dig_llm_provider(cfg)
    assert isinstance(prov, OpenAICompatibleProvider)
    assert prov._base == "https://apihub.agnes-ai.com/v1"
    assert prov._model == "agnes-2.0-flash"


def test_dig_provider_default_inherits_main_adapter() -> None:
    # No override -> the dig adapter is built from the MAIN provider (same vendor as
    # generation), proving the original behavior is the default.
    cfg = Config(
        provider=ProviderCfg(name="openai_compatible", model="main-model",
                             base_url="https://main/v1", api_key_env="MAIN_KEY"),
        deep_dive=DeepDiveCfg(model=""),
    )
    prov = get_dig_llm_provider(cfg)
    assert isinstance(prov, OpenAICompatibleProvider)
    assert prov._base == "https://main/v1"
    assert prov._model == "main-model"
