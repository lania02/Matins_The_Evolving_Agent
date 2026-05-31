"""End-to-end integration of the generation pipeline, fully offline.

Exercises the real spine (run_batch) + the real leaf modules (slots/schema/
novelty/digest) wired together, using a FakeLLM and no search/messaging. This is
the one test that proves the whole daily-run path integrates -- the unit tests
deliberately avoid run_batch because it needs an LLM.
"""
from __future__ import annotations

from pathlib import Path

from matins.config import load_config
from matins.digest.render import render_digest
from matins.generate.pipeline import run_batch
from matins.store.db import Store
from matins.store.models import SLOTS

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeLLM:
    """Returns a valid idea object for generation calls, a rank list for self-rank."""

    def __init__(self) -> None:
        self.n = 0

    def generate(self, prompt: str, *, temperature: float, json_schema=None) -> str:
        if "single JSON object" in prompt:  # from the IDEA_SCHEMA instruction
            self.n += 1
            return (
                '{"title": "Idea %d", "mechanism": "m", "why_now": "w", '
                '"math_structure": "", "tractability": "t", "fit_to_program": "f"}'
            ) % self.n
        # self-rank prompt -> JSON list of {idx, rank, rationale}
        return (
            '[{"idx":1,"rank":1,"rationale":"r1"},{"idx":2,"rank":2,"rationale":"r2"},'
            '{"idx":3,"rank":3,"rationale":"r3"},{"idx":4,"rank":4,"rationale":"r4"}]'
        )


def _cfg():
    # root resolves to the repo, so prompts_dir / skills_dir / interest_seed exist.
    return load_config(str(REPO_ROOT / "config.example.yaml"))


def test_run_batch_full_pipeline():
    cfg = _cfg()
    store = Store(":memory:")
    llm = FakeLLM()

    batch, ideas = run_batch(cfg, store, llm, None, date="2026-01-01")

    assert len(ideas) == 4
    assert [i.slot for i in ideas] == SLOTS
    assert [i.idx for i in ideas] == [1, 2, 3, 4]
    # self-rank applied
    assert all(i.self_rank is not None for i in ideas)
    # novelty with search=None flags every idea unchecked
    assert all(i.prior_art == "[unchecked]" for i in ideas)
    # random slot recorded its sampled genes
    rnd = next(i for i in ideas if i.slot == "random")
    assert rnd.random_genes != ""

    # digest renders one message per idea, header carries the date
    header, msgs = render_digest(batch, ideas, cfg.generation.output_language)
    assert "2026-01-01" in header
    assert len(msgs) == 4
    assert all(len(m) < 4096 for m in msgs)


def test_interleave_balances_and_caps():
    from matins.generate.pipeline import _interleave
    a = [{"via": "arxiv"}, {"via": "arxiv"}]
    b = [{"via": "web"}]
    c = [{"via": "hn"}, {"via": "hn"}, {"via": "hn"}]
    out = _interleave([a, b, c], cap=4)
    assert len(out) == 4                                         # cap honored
    assert [o["via"] for o in out] == ["arxiv", "web", "hn", "arxiv"]   # round-robin


def test_collect_channel_dedup_quota_and_tag():
    from matins.generate.pipeline import _collect_channel

    class FakeP:
        def __init__(self, per_query):
            self.per_query = per_query

        def search(self, q, *, k=5):
            return self.per_query.get(q, [])

    seen = {"http://dup"}                                        # already seen -> skipped
    provider = FakeP({
        "q1": [{"url": "http://dup"}, {"url": "http://a"}],      # dup skipped, takes 'a'
        "q2": [{"url": "http://b"}],
        "q3": [{"url": "http://c"}],
    })
    out = _collect_channel(provider, ["q1", "q2", "q3"], quota=2, seen=seen, label="arxiv")
    assert [o["url"] for o in out] == ["http://a", "http://b"]  # one per query, quota stops at 2
    assert all(o["via"] == "arxiv" for o in out)


def test_run_batch_is_idempotent_per_date():
    cfg = _cfg()
    store = Store(":memory:")
    llm = FakeLLM()

    b1, ideas1 = run_batch(cfg, store, llm, None, date="2026-01-02")
    b2, ideas2 = run_batch(cfg, store, llm, None, date="2026-01-02")

    assert b1.batch_id == b2.batch_id
    assert len(ideas2) == 4
