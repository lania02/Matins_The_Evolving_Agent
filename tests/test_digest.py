"""Offline tests for digest rendering (matins.digest.render)."""
from __future__ import annotations

from matins.digest.render import render_digest, render_overview
from matins.store.db import Store
from matins.store.models import SLOTS, Batch, Feedback, Idea


def _make_batch_and_ideas() -> tuple[Batch, list[Idea]]:
    batch = Batch(
        batch_id="b-1",
        date="2026-05-29",
        skill_version=1,
        temperature=0.4,
        provider="anthropic",
        model="claude-opus-4-8",
    )
    ideas = [
        Idea(
            idea_id=f"idea-{i}",
            batch_id=batch.batch_id,
            slot=SLOTS[i],
            idx=i + 1,
            title=f"Idea {i + 1}",
            mechanism="a concrete mechanism",
            why_now="timely because of recent work",
            math_structure="a category-theoretic framing",
            tractability="a clear first step",
            fit_to_program="aligns with the program",
            prior_art="[unchecked]",
            self_rank=i + 1,
            self_rationale="because",
        )
        for i in range(4)
    ]
    return batch, ideas


def test_render_digest_shape_and_limits() -> None:
    batch, ideas = _make_batch_and_ideas()
    header, msgs = render_digest(batch, ideas, "bilingual")

    assert isinstance(header, str)
    assert isinstance(msgs, list)
    assert len(msgs) == 4
    for msg in msgs:
        assert isinstance(msg, str)
        assert len(msg) < 4096

    # The batch date is shown in the header.
    assert batch.date in header


def test_self_critique_surfaced_right_after_intuition() -> None:
    # The model's own self-rank rationale is the most independent skepticism available
    # (e.g. "already studied by X et al.") -- it must be visible near the TOP of the card,
    # right after Intuition and BEFORE the pitch (Vantage/Bridge), not buried at the bottom.
    batch, ideas = _make_batch_and_ideas()
    idea = ideas[0]
    idea.intuition = "a plain-language pitch"
    idea.lens = "Some Vantage"
    idea.bridge = "the collision"
    idea.self_rank = 4
    idea.self_rationale = "this exact pairing is already studied by Author et al. 2020"

    _header, msgs = render_digest(batch, ideas, "bilingual")
    msg = msgs[0]
    assert "Self rank: 4/4 — this exact pairing is already studied" in msg
    intuition_pos = msg.index("Intuition:")
    self_rank_pos = msg.index("Self rank:")
    vantage_pos = msg.index("Vantage:")
    bridge_pos = msg.index("Bridge:")
    assert intuition_pos < self_rank_pos < vantage_pos < bridge_pos


def test_self_critique_absent_when_no_rationale() -> None:
    batch, ideas = _make_batch_and_ideas()
    idea = ideas[0]
    idea.self_rationale = ""
    _header, msgs = render_digest(batch, ideas, "bilingual")
    assert "Self rank:" not in msgs[0]


def test_render_overview_pulls_from_store() -> None:
    batch, ideas = _make_batch_and_ideas()
    store = Store(":memory:")
    store.insert_batch(batch)
    for idea in ideas:
        store.insert_idea(idea)
    store.insert_feedback(Feedback(idea_id="idea-0", user_rank=2, user_comment="like it", source="cli"))
    store.log_retrieval(batch.batch_id, query="q", source="blend:arxiv",
                        result_ids=["http://x/1"])

    md = render_overview(store, [batch], db_path="mem.db")
    assert "database view" in md
    assert batch.date in md
    assert "Idea 1" in md                      # idea title rendered
    assert "Fresh retrieval fed in" in md      # blend log surfaced
    assert "http://x/1" in md
    assert "like it" in md                     # user feedback surfaced
