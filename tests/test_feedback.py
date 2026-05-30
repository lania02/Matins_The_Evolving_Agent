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


def test_kendall_tau_perfect_agreement() -> None:
    assert kendall_tau([1, 2, 3, 4], [1, 2, 3, 4]) == 1.0


def test_kendall_tau_perfect_disagreement() -> None:
    assert kendall_tau([1, 2, 3, 4], [4, 3, 2, 1]) == -1.0
