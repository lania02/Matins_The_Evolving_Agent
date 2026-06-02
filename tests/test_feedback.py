"""Offline tests for reply parsing and rank-divergence math (matins.feedback)."""
from __future__ import annotations

from matins.feedback.capture import parse_ranking
from matins.feedback.diverge import kendall_tau


def _split_ranks_comments(result):
    """Normalize parse_ranking's return into (ranks, comments) mappings.

    The reply parser must yield a number->rank mapping and a number->comment
    mapping; tolerate the common shapes an implementation might use (tuple,
    attributes, or dict) so the test asserts behavior, not representation.
    """
    if isinstance(result, tuple):
        ranks, comments = result[0], (result[1] if len(result) > 1 else {})
    elif isinstance(result, dict) and "ranks" in result:
        ranks, comments = result["ranks"], result.get("comments", {})
    elif hasattr(result, "ranks"):
        ranks, comments = result.ranks, getattr(result, "comments", {})
    else:
        # Bare mapping number->rank.
        ranks, comments = result, {}
    return dict(ranks), dict(comments or {})


def test_parse_ranking_maps_numbers_to_ranks() -> None:
    ranks, _comments = _split_ranks_comments(parse_ranking("3>1>4>2", 4))
    assert ranks[3] == 1
    assert ranks[1] == 2
    assert ranks[4] == 3
    assert ranks[2] == 4


def test_parse_ranking_extracts_comments() -> None:
    text = "3>1>4>2\n#3 note"
    _ranks, comments = _split_ranks_comments(parse_ranking(text, 4))
    assert 3 in comments
    assert "note" in comments[3]


def test_replies_for_batch_binds_to_digest() -> None:
    from matins.feedback.capture import replies_for_batch

    replies = [
        {"text": "to this batch", "reply_to_message_id": "100"},   # replies to our digest
        {"text": "direct message", "reply_to_message_id": None},   # no target -> this batch
        {"text": "to yesterday", "reply_to_message_id": "57"},     # older digest -> dropped
    ]
    kept = [r["text"] for r in replies_for_batch(replies, "100")]
    assert kept == ["to this batch", "direct message"]             # cross-batch reply excluded


def test_replies_for_batch_without_digest_takes_only_direct() -> None:
    from matins.feedback.capture import replies_for_batch

    replies = [
        {"text": "direct", "reply_to_message_id": None},
        {"text": "reply to something", "reply_to_message_id": "9"},
    ]
    # digest never sent (None) -> only no-reply direct messages qualify.
    assert [r["text"] for r in replies_for_batch(replies, None)] == ["direct"]


def test_kendall_tau_perfect_agreement() -> None:
    assert kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0


def test_kendall_tau_perfect_disagreement() -> None:
    assert kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0


def test_classify_comment_routes_and_defaults() -> None:
    from matins.feedback.capture import classify_comment

    class FakeLLM:
        def __init__(self, reply):
            self.reply = reply

        def generate(self, prompt, *, temperature, json_schema=None):
            return self.reply

    assert classify_comment(FakeLLM("novelty"), "already done by Smith 2024") == "novelty"
    assert classify_comment(FakeLLM("FEASIBILITY"), "can't build it") == "feasibility"
    assert classify_comment(FakeLLM("garbage word"), "x") == "taste"   # unrecognized -> default
    assert classify_comment(FakeLLM("novelty"), "") == ""              # empty comment -> no channel

    class BoomLLM:
        def generate(self, *a, **k):
            raise RuntimeError("endpoint down")

    assert classify_comment(BoomLLM(), "x") == "taste"                 # failure -> default, never raises


def test_ingest_with_classify_persists_channel() -> None:
    from matins.feedback.capture import ingest_cli_feedback
    from matins.store.db import Store, new_id, now_iso, today_iso
    from matins.store.models import Batch, Idea

    store = Store(":memory:")
    batch = Batch(batch_id=new_id(), date=today_iso(), provider="x", model="m", created_at=now_iso())
    store.insert_batch(batch)
    for i in (1, 2):
        store.insert_idea(Idea(idea_id=new_id(), batch_id=batch.batch_id, slot="highfit",
                               idx=i, title=f"T{i}", mechanism="m"))
    ingest_cli_feedback(store, batch, "1>2\n#1 already done elsewhere",
                        classify=lambda c: "novelty")
    by_idx = {i.idx: i for i in store.ideas_for_batch(batch.batch_id)}
    assert store.feedback_for_idea(by_idx[1].idea_id).comment_kind == "novelty"  # tagged
    assert store.feedback_for_idea(by_idx[2].idea_id).comment_kind == ""         # no comment -> no channel


def test_reflect_on_batch_is_idempotent_per_batch() -> None:
    # reflect_on_batch runs on EVERY collect. Re-collecting (cron + manual) or
    # re-ranking the same batch must not inflate a taste hypothesis's occurrence /
    # confidence, or it would falsely trip the consolidation threshold.
    from matins.feedback.diverge import reflect_on_batch
    from matins.store.db import Store, new_id, now_iso, today_iso
    from matins.store.models import Batch, Feedback, Idea

    class FakeLLM:
        def generate(self, prompt, *, temperature, json_schema=None):
            return "topic|over-weighted topical fit"

    store = Store(":memory:")
    batch = Batch(batch_id=new_id(), date=today_iso(), skill_version=None,
                  temperature=0.4, provider="x", model="m", created_at=now_iso())
    store.insert_batch(batch)
    # self ranks A best; user ranks B best -> tau = -1 (< 0.5) -> logs a hypothesis.
    a = Idea(idea_id=new_id(), batch_id=batch.batch_id, slot="highfit", idx=1,
             title="A", mechanism="m", self_rank=1)
    b = Idea(idea_id=new_id(), batch_id=batch.batch_id, slot="adjacent", idx=2,
             title="B", mechanism="m", self_rank=2)
    store.insert_idea(a)
    store.insert_idea(b)
    store.insert_feedback(Feedback(idea_id=a.idea_id, user_rank=2, source="cli"))
    store.insert_feedback(Feedback(idea_id=b.idea_id, user_rank=1, source="cli"))

    reflect_on_batch(None, store, FakeLLM(), batch)
    h1 = store.find_hypothesis("over-weighted topical fit")
    assert h1 is not None and h1.occurrence == 1

    reflect_on_batch(None, store, FakeLLM(), batch)   # same batch, re-reflected
    h2 = store.find_hypothesis("over-weighted topical fit")
    assert h2.occurrence == 1                          # not inflated
    assert h2.confidence == h1.confidence
