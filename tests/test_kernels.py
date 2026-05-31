"""Offline tests for event rendering (matins.memory.kernels.format_events).

format_events feeds BOTH memory tiers, so the learning signals it surfaces inline
(algo-update.md #1 clean-probe, #2 positive-surprise, #3 comment channel) are what
the fast snapshot and the slow skill proposal actually condition on.
"""
from __future__ import annotations

from matins.memory.kernels import format_events


def test_random_slot_tagged_as_clean_probe() -> None:
    out = format_events([{"date": "2026-01-01", "slot": "random", "idx": 4, "title": "T"}])
    assert "[D: clean probe]" in out
    # A conditioned slot is NOT tagged.
    out2 = format_events([{"date": "2026-01-01", "slot": "highfit", "idx": 1, "title": "T"}])
    assert "[D: clean probe]" not in out2


def test_positive_surprise_tagged_when_user_outranks_self() -> None:
    # self_rank 4 (system thought worst), user_rank 1 (user thought best) -> underrated by 3.
    out = format_events([{"slot": "highfit", "self_rank": 4, "user_rank": 1, "title": "T"}])
    assert "[+underrated by 3]" in out


def test_no_surprise_tag_when_gap_small_or_negative() -> None:
    # gap = 1 (< 2 threshold) -> no tag; the system slightly under-ranked, weak signal.
    assert "[+underrated" not in format_events([{"self_rank": 2, "user_rank": 1, "title": "T"}])
    # gap negative (system over-ranked) -> not a positive surprise.
    assert "[+underrated" not in format_events([{"self_rank": 1, "user_rank": 3, "title": "T"}])
    # a missing rank -> no tag, no crash.
    assert "[+underrated" not in format_events([{"self_rank": 4, "title": "T"}])


def test_comment_channel_label_surfaced() -> None:
    out = format_events([{"title": "T", "user_comment": "already done by X", "comment_kind": "novelty"}])
    assert "comment[novelty]=already done by X" in out
    # No kind -> plain label (back-compat with un-classified rows).
    out2 = format_events([{"title": "T", "user_comment": "nice"}])
    assert "comment=nice" in out2
    assert "comment[" not in out2
