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


def test_trending_pulse_caches_interleaves_and_degrades_keyless(monkeypatch) -> None:
    from matins.providers.search_web import TrendingPulseProvider

    p = TrendingPulseProvider(subreddits=["ClaudeAI", "LocalLLaMA"], serp_key="k")
    calls = {"hn": 0, "sub": 0}
    monkeypatch.setattr(p, "_fetch_hn_front",
                        lambda k=8: calls.__setitem__("hn", calls["hn"] + 1) or
                        [{"title": "[HN front] H1", "url": "http://h1", "snippet": ""}])
    monkeypatch.setattr(p, "_fetch_subreddit",
                        lambda s, k=3: calls.__setitem__("sub", calls["sub"] + 1) or
                        [{"title": f"[r/{s}] T", "url": f"http://{s}", "snippet": ""}])
    pool1 = p.search("q1", k=3)
    pool2 = p.search("q2", k=3)                      # second query: served from cache
    assert pool1 is pool2 and calls["hn"] == 1 and calls["sub"] == 2   # fetched ONCE
    titles = [x["title"] for x in pool1]
    assert any(t.startswith("[r/ClaudeAI]") for t in titles)
    assert any(t.startswith("[HN front]") for t in titles)

    keyless = TrendingPulseProvider(subreddits=["ClaudeAI"], serp_key=None)
    monkeypatch.setattr(keyless, "_fetch_hn_front",
                        lambda k=8: [{"title": "[HN front] H", "url": "http://h", "snippet": ""}])
    pool = keyless.search("q", k=3)                  # no serp key -> HN-only, no crash
    assert pool and all(t["title"].startswith("[HN front]") for t in pool)


def test_serp_and_trending_route_via_factory(monkeypatch) -> None:
    from matins.providers.search_web import (
        SerpSearchProvider,
        TrendingPulseProvider,
        get_retrieval_searcher,
    )

    monkeypatch.setenv("SERP_API_KEY", "test-key")
    cfg = _cfg("arxiv")
    assert isinstance(get_retrieval_searcher("serp", cfg), SerpSearchProvider)
    assert isinstance(get_retrieval_searcher("trending", cfg), TrendingPulseProvider)
    monkeypatch.delenv("SERP_API_KEY")
    assert get_retrieval_searcher("serp", cfg) is None       # keyless serp -> unavailable
    assert get_retrieval_searcher("trending", cfg) is not None   # trending degrades, not None


def test_openai_compatible_sends_max_tokens_and_fails_loud_on_empty_choices(monkeypatch) -> None:
    # NVIDIA's gateway returns HTTP 200 with choices=[] when max_tokens is omitted
    # (seen live with minimax-m3): the adapter must always send a cap, and an empty
    # choices list must raise a labelled RuntimeError, not a bare IndexError.
    import httpx as _httpx
    import pytest

    from matins.providers import openai_compatible as oc

    captured = {}

    class FakeResp:
        status_code = 200
        text = '{"choices": []}'

        def json(self):
            return {"choices": []}

    def fake_post(url, headers=None, json=None, timeout=None):
        captured["payload"] = json
        return FakeResp()

    monkeypatch.setattr(oc.httpx, "post", fake_post)
    cfg = Config(provider=ProviderCfg(name="openai_compatible", model="m",
                                      base_url="http://x/v1", api_key_env="NOPE"))
    provider = oc.OpenAICompatibleProvider(cfg)
    with pytest.raises(RuntimeError, match="no choices"):
        provider.generate("hi", temperature=0.0)
    assert captured["payload"]["max_tokens"] > 0             # cap always sent
