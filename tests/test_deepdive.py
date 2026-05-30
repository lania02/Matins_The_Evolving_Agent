"""Offline tests for the on-demand deep-dive feature (fakes; no network/keys)."""
from __future__ import annotations

from pathlib import Path

from matins.feedback.capture import parse_dig
from matins.generate.deepdive import gather_sources, propose_queries, synthesize_brief
from matins.store.db import Store
from matins.store.models import Idea

REPO_ROOT = Path(__file__).resolve().parent.parent
PROMPTS = REPO_ROOT / "prompts"


class FakeSearch:
    def __init__(self, results):
        self._r = results

    def search(self, query, *, k=5):
        return self._r


class FakeLLM:
    def __init__(self, reply):
        self.reply = reply

    def generate(self, prompt, *, temperature, json_schema=None):
        return self.reply


def test_parse_dig_variants() -> None:
    assert parse_dig("dig #3", 4) == [3]
    assert parse_dig("DIG 2", 4) == [2]
    assert parse_dig("deep dive #1 and deep-dive #4", 4) == [1, 4]
    assert parse_dig("dig #9", 4) == []                       # out of range
    assert parse_dig("digital natives", 4) == []              # 'dig' inside a word ignored
    assert parse_dig("dig #2 ... dig 2", 4) == [2]            # de-duped


def test_gather_sources_dedup_and_tag() -> None:
    a = FakeSearch([{"title": "A", "url": "http://x/1", "snippet": "sa"}])
    w = FakeSearch([
        {"title": "A2", "url": "http://x/1", "snippet": "dup"},   # duplicate url
        {"title": "B", "url": "http://x/2", "snippet": "sb"},
    ])
    out = gather_sources([(a, "arxiv"), (w, "web")], ["q1", "q2"], k=5)
    assert [s["url"] for s in out] == ["http://x/1", "http://x/2"]
    assert out[0]["via"] == "arxiv"


def test_propose_queries_parses_and_falls_back() -> None:
    idea = Idea(idea_id="i1", batch_id="b", slot="highfit", idx=1, title="Spectral Markets Model")
    qs = propose_queries(FakeLLM('["spectral radius markets", "market microstructure"]'),
                         idea, PROMPTS, 4)
    assert qs == ["spectral radius markets", "market microstructure"]
    qs2 = propose_queries(FakeLLM("not json"), idea, PROMPTS, 4)
    assert qs2 and all(isinstance(q, str) and q for q in qs2)   # title fallback


def test_synthesize_brief_runs() -> None:
    idea = Idea(idea_id="i1", batch_id="b", slot="highfit", idx=1, title="X")
    sources = [{"title": "Paper", "url": "http://x/1", "snippet": "sn", "via": "arxiv"}]
    out = synthesize_brief(FakeLLM("BRIEF [1]"), idea, sources, PROMPTS, "bilingual")
    assert out == "BRIEF [1]"


def test_deep_dive_store_roundtrip() -> None:
    store = Store(":memory:")
    store.save_deep_dive("i1", "the brief", '[{"url": "u"}]')
    assert store.get_deep_dive("i1")["brief"] == "the brief"
    store.save_deep_dive("i1", "updated", "[]")                 # upsert
    assert store.get_deep_dive("i1")["brief"] == "updated"
    assert store.get_deep_dive("missing") is None
