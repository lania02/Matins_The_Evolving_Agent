"""Offline tests for the daily news radar (matins.news) -- no network.

The radar is advisory: it must rank by velocity rather than raw volume, never re-report a
story, and degrade to an empty list rather than break a run.
"""
from __future__ import annotations

from matins.config import Config
from matins.digest.render import render_news
from matins.news import NewsItem, collect_news, score_and_rank
from matins.store.db import Store


def _item(title, url, *, source="HN", metric=100, age_hours=10.0):
    return NewsItem(title=title, url=url, source=source, metric=metric,
                    age_hours=age_hours, velocity=metric / age_hours)


def test_ranks_by_velocity_not_raw_volume():
    # 2000 points over 3 days is ordinary; 600 points in 2 hours is an event.
    slow = _item("slow burner", "http://a", metric=2000, age_hours=72.0)   # ~28/h
    fast = _item("breaking thing", "http://b", metric=600, age_hours=2.0)  # 300/h
    ranked = score_and_rank([slow, fast], [])
    assert [r.url for r in ranked] == ["http://b", "http://a"]


def test_cross_source_corroboration_boosts_and_is_marked():
    lonely = _item("some isolated story", "http://a", metric=500, age_hours=2.0)
    echoed = _item("llama quantization breakthrough", "http://b", metric=300, age_hours=2.0)
    subs = ["[r/LocalLLaMA] new llama quantization breakthrough lands"]
    ranked = score_and_rank([lonely, echoed], subs)
    assert ranked[0].url == "http://b"            # lower velocity, but corroborated -> wins
    assert ranked[0].corroborated is True
    assert ranked[1].corroborated is False


def test_score_and_rank_handles_empty_pool():
    assert score_and_rank([], ["[r/x] anything"]) == []


def test_magnitude_keeps_a_landmark_from_being_buried():
    # Observed live: a 1011-point Terence Tao thread (climbing all day) ranked BELOW a
    # 176-point news-cycle piece under pure-velocity scoring. Magnitude must rescue it.
    landmark = _item("tao on the jacobian conjecture", "http://big", metric=1011, age_hours=21.0)
    spike = _item("routine industry news piece", "http://spike", metric=176, age_hours=1.4)
    ranked = score_and_rank([landmark, spike], [])
    assert ranked[0].url == "http://big"


def test_sources_are_normalized_separately():
    # GitHub stars/hour and HN points/hour are different units: a repo's 22000 stars must
    # not drown out every HN story just because the number is bigger.
    repo = _item("some repo", "http://repo", source="GitHub", metric=22000, age_hours=210.0)
    story = _item("top hn story", "http://hn", source="HN", metric=900, age_hours=3.0)
    ranked = score_and_rank([repo, story], [])
    # each is the peak of its own source, so both score 1.0 -- neither is suppressed
    assert {r.score for r in ranked} == {1.0}


def _cfg(**news):
    cfg = Config()
    cfg.news.enabled = True
    for k, v in news.items():
        setattr(cfg.news, k, v)
    return cfg


def test_collect_dedups_against_recent_and_records(monkeypatch):
    store = Store(":memory:")
    store.insert_news_event("http://old", "a story from yesterday", "HN", 1.0)

    fresh = _item("a genuinely new thing", "http://new", metric=400, age_hours=2.0)
    repeat_url = _item("different words entirely", "http://old", metric=900, age_hours=2.0)
    repeat_story = _item("a story from yesterday", "http://mirror", metric=800, age_hours=2.0)

    monkeypatch.setattr("matins.news.fetch_hn_top",
                        lambda **kw: [fresh, repeat_url, repeat_story])
    monkeypatch.setattr("matins.news.fetch_github_new", lambda **kw: [])
    monkeypatch.setattr("matins.news._subreddit_titles", lambda cfg: [])

    picked = collect_news(_cfg(count=5), store)
    assert [p.url for p in picked] == ["http://new"]      # seen url AND seen title dropped

    # the pick is recorded, so a second run the same day reports nothing
    assert collect_news(_cfg(count=5), store) == []


def test_collect_survives_dead_sources(monkeypatch):
    store = Store(":memory:")

    def boom(**kw):
        raise RuntimeError("network down")

    # the fetchers swallow their own errors; assert collect tolerates an empty world
    monkeypatch.setattr("matins.news.fetch_hn_top", lambda **kw: [])
    monkeypatch.setattr("matins.news.fetch_github_new", lambda **kw: [])
    monkeypatch.setattr("matins.news._subreddit_titles", boom)
    assert collect_news(_cfg(), store) == []


def test_collect_respects_count_and_clusters_near_duplicates(monkeypatch):
    store = Store(":memory:")
    items = [
        _item("openai releases a new reasoning model", "http://1", metric=900, age_hours=2.0),
        _item("openai releases new reasoning model today", "http://2", metric=800, age_hours=2.0),
        _item("rust compiler gets faster builds", "http://3", metric=700, age_hours=2.0),
        _item("kubernetes adds sidecar containers", "http://4", metric=600, age_hours=2.0),
    ]
    monkeypatch.setattr("matins.news.fetch_hn_top", lambda **kw: items)
    monkeypatch.setattr("matins.news.fetch_github_new", lambda **kw: [])
    monkeypatch.setattr("matins.news._subreddit_titles", lambda cfg: [])

    picked = collect_news(_cfg(count=2), store)
    assert len(picked) == 2
    urls = [p.url for p in picked]
    assert "http://1" in urls and "http://2" not in urls   # near-duplicate story collapsed


def test_render_news_shows_evidence_and_is_labelled_unvetted():
    items = [_item("something broke", "http://x", metric=640, age_hours=2.0)]
    items[0].corroborated = True
    out = render_news(items, "2026-07-09")
    assert "Radar — 2026-07-09" in out
    assert "hot, not vetted" in out          # honest framing: attention, not truth
    assert "HN ↑640" in out and "320/h" in out and "×subreddit" in out
    assert "http://x" in out
    assert render_news([], "2026-07-09") == ""
