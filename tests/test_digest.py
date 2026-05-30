"""Offline tests for digest rendering (matins.digest.render)."""
from __future__ import annotations

from matins.digest.render import render_digest
from matins.store.models import SLOTS, Batch, Idea


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
