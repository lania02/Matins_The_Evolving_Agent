"""Offline tests for the SQLite Store (in-memory database, no network)."""
from __future__ import annotations

from matins.store.db import Store, new_id, now_iso, today_iso
from matins.store.models import Batch, Feedback, Idea


def _make_store() -> Store:
    return Store(":memory:")


def _seed_batch_with_two_ideas(store: Store) -> tuple[Batch, Idea, Idea]:
    batch = Batch(
        batch_id=new_id(),
        date=today_iso(),
        skill_version=None,
        temperature=0.4,
        provider="anthropic",
        model="claude-opus-4-8",
        created_at=now_iso(),
    )
    store.insert_batch(batch)

    # Insert idx=2 first, then idx=1, to prove ordering is by idx, not insert order.
    idea_b = Idea(idea_id=new_id(), batch_id=batch.batch_id, slot="adjacent", idx=2,
                  title="second", mechanism="m2")
    idea_a = Idea(idea_id=new_id(), batch_id=batch.batch_id, slot="highfit", idx=1,
                  title="first", mechanism="m1")
    store.insert_idea(idea_b)
    store.insert_idea(idea_a)
    return batch, idea_a, idea_b


def test_insert_and_read_ideas_sorted_by_idx() -> None:
    store = _make_store()
    batch, idea_a, idea_b = _seed_batch_with_two_ideas(store)

    store.insert_feedback(Feedback(idea_id=idea_a.idea_id, user_rank=1,
                                   user_comment="nice", source="telegram"))

    ideas = store.ideas_for_batch(batch.batch_id)
    assert [i.idx for i in ideas] == [1, 2]
    assert ideas[0].title == "first"

    fb = store.feedback_for_idea(idea_a.idea_id)
    assert fb is not None
    assert fb.user_rank == 1
    assert store.batch_has_feedback(batch.batch_id) is True


def test_recent_events_returns_events_with_expected_keys() -> None:
    store = _make_store()
    batch, idea_a, _ = _seed_batch_with_two_ideas(store)
    store.insert_feedback(Feedback(idea_id=idea_a.idea_id, user_rank=1, source="cli"))

    events = store.recent_events(window_days=3650, stride=1)
    assert len(events) >= 1
    expected_keys = {
        "batch_id", "date", "tau", "idea_id", "slot", "idx", "title",
        "mechanism", "math_structure", "tractability", "self_rank",
        "self_rationale", "user_rank", "user_comment",
    }
    assert expected_keys.issubset(events[0].keys())


def test_recent_events_uses_only_latest_feedback_per_idea() -> None:
    # Append-only log: re-ranking an idea inserts a second feedback row. recent_events
    # must NOT multiply that idea (which would skew fast memory + consolidation); it
    # takes only the latest feedback per idea.
    store = _make_store()
    batch, idea_a, idea_b = _seed_batch_with_two_ideas(store)
    store.insert_feedback(Feedback(idea_id=idea_a.idea_id, user_rank=1,
                                   user_comment="first take", source="telegram"))
    store.insert_feedback(Feedback(idea_id=idea_a.idea_id, user_rank=3,
                                   user_comment="changed my mind", source="telegram"))

    events = store.recent_events(window_days=3650, stride=1)
    assert len(events) == 2                                   # one row per idea, not 3
    a_event = next(e for e in events if e["idea_id"] == idea_a.idea_id)
    assert a_event["user_rank"] == 3                          # latest wins
    assert a_event["user_comment"] == "changed my mind"


def test_skill_versioning_and_approval() -> None:
    store = _make_store()
    v1 = store.insert_skill_version("taste v1", parent_version=None, diff_summary="init")
    v2 = store.insert_skill_version("taste v2", parent_version=1, diff_summary="edit")
    assert v1 == 1
    assert v2 == 2
    assert store.latest_skill_version() == 2

    # No approved version yet.
    assert store.active_skill() is None

    store.approve_skill(1)
    active = store.active_skill()
    assert active is not None
    assert active.version == 1


def test_offset_round_trip() -> None:
    store = _make_store()
    assert store.get_offset("telegram") is None
    store.set_offset("telegram", "12345")
    assert store.get_offset("telegram") == "12345"
    # Upsert overwrites.
    store.set_offset("telegram", "67890")
    assert store.get_offset("telegram") == "67890"


def test_retrieval_log_and_recent_result_ids() -> None:
    store = _make_store()
    batch = Batch(batch_id=new_id(), date=today_iso(), created_at=now_iso())
    store.insert_batch(batch)
    store.log_retrieval(batch.batch_id, query="phase transitions",
                        source="web", result_ids=["arxiv:1234", "arxiv:5678"])

    recent = store.recent_result_ids(days=30)
    assert "arxiv:1234" in recent
    assert "arxiv:5678" in recent
