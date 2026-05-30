"""Offline tests for the curated 'must try #N' favorites feature."""
from __future__ import annotations

from matins.digest.render import render_favorites_md
from matins.feedback.capture import ingest_must_try, parse_must_try
from matins.store.db import Store
from matins.store.models import Batch, Idea


def _seed_batch(store: Store) -> Batch:
    b = Batch(batch_id="b1", date="2026-01-01")
    store.insert_batch(b)
    for idx in range(1, 5):
        store.insert_idea(
            Idea(idea_id=f"i{idx}", batch_id="b1", slot="highfit", idx=idx,
                 title=f"Idea {idx}", mechanism="m", created_at="2026-01-01T00:00:00")
        )
    return b


def test_parse_must_try_variants() -> None:
    assert parse_must_try("must try #3", 4) == [3]
    assert parse_must_try("MUST TRY 2", 4) == [2]
    assert parse_must_try("must-try #1 and musttry#4", 4) == [1, 4]
    assert parse_must_try("must try #9", 4) == []          # out of range dropped
    assert parse_must_try("must try #2 ... must try #2", 4) == [2]  # de-duped


def test_favorite_store_roundtrip() -> None:
    store = Store(":memory:")
    _seed_batch(store)
    assert store.is_favorite("i2") is False
    assert store.add_favorite("i2") is True
    assert store.add_favorite("i2") is False               # idempotent
    assert store.is_favorite("i2") is True
    favs = store.list_favorites()
    assert len(favs) == 1
    idea, note, fav_at = favs[0]
    assert idea.idea_id == "i2" and idea.title == "Idea 2"


def test_ingest_must_try_from_replies() -> None:
    store = Store(":memory:")
    b = _seed_batch(store)
    replies = [
        {"text": "3>1>4>2", "ts": "0", "reply_to_message_id": None, "update_id": "1"},
        {"text": "must try #4\nmust try #1", "ts": "0", "reply_to_message_id": None, "update_id": "2"},
    ]
    added = ingest_must_try(store, b, replies)
    assert {i.idx for i in added} == {1, 4}
    assert {idea.idea_id for idea, _, _ in store.list_favorites()} == {"i1", "i4"}
    # Re-ingesting the same replies adds nothing new (idempotent).
    assert ingest_must_try(store, b, replies) == []


def test_render_favorites_md() -> None:
    assert "none yet" in render_favorites_md([])
    store = Store(":memory:")
    _seed_batch(store)
    store.add_favorite("i3", note="love this")
    md = render_favorites_md(store.list_favorites())
    assert "Idea 3" in md and "love this" in md
