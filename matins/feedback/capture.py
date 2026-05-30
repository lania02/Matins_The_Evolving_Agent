"""Parse human feedback (a ranking line + optional per-idea comments) into rows.

The feedback format is deliberately terse so it is easy to type from a phone:

    3>1>4>2          <- the ranking line (best first); rank = position + 1
    #1 too incremental
    #4 love the math structure

Parsing is tolerant: malformed input never raises. Anything we cannot interpret
is surfaced as a human-readable string in `problems` so the daily loop can warn
without crashing.
"""
from __future__ import annotations

import re

from ..store.models import Feedback

# A ranking line is idea numbers joined by '>' (spaces allowed around each).
_RANK_LINE = re.compile(r"^\s*\d+(?:\s*>\s*\d+)+\s*$")
# A comment line: '#<n> <text>'.
_COMMENT_LINE = re.compile(r"^\s*#\s*(\d+)\s*(.*)$")


def parse_ranking(
    text: str, n_ideas: int
) -> tuple[dict[int, int], dict[int, str], list[str]]:
    """Parse a feedback blob into ranks, comments, and a list of problems.

    Returns:
        ranks:    {idea_idx -> rank} where rank = position in the ranking line + 1.
        comments: {idea_idx -> comment text}.
        problems: human-readable strings describing anything unparseable.
    """
    ranks: dict[int, int] = {}
    comments: dict[int, str] = {}
    problems: list[str] = []

    lines = (text or "").splitlines()

    # ---- ranking line: take the first line that looks like a ranking --------
    ranking_indices: list[int] = []
    found_ranking = False
    for line in lines:
        if _RANK_LINE.match(line):
            parts = [p.strip() for p in line.split(">")]
            for p in parts:
                try:
                    ranking_indices.append(int(p))
                except ValueError:
                    continue
            found_ranking = True
            break

    if not found_ranking:
        problems.append("no ranking line found")

    for position, idx in enumerate(ranking_indices):
        if 1 <= idx <= n_ideas:
            # First occurrence wins if a number is duplicated.
            ranks.setdefault(idx, position + 1)
        else:
            problems.append(f"unknown idea #{idx}")

    # ---- comment lines -------------------------------------------------------
    for line in lines:
        m = _COMMENT_LINE.match(line)
        if not m:
            continue
        idx = int(m.group(1))
        body = m.group(2).strip()
        if 1 <= idx <= n_ideas:
            comments[idx] = body
        else:
            problems.append(f"unknown idea #{idx}")

    # ---- missing ranks -------------------------------------------------------
    if found_ranking:
        for k in range(1, n_ideas + 1):
            if k not in ranks:
                problems.append(f"missing rank for #{k}")

    return ranks, comments, problems


def _ingest_text(store, batch, text: str, source: str) -> int:
    """Parse `text` against the batch's ideas and insert one Feedback per hit."""
    ideas = store.ideas_for_batch(batch.batch_id)
    ranks, comments, _ = parse_ranking(text, len(ideas))
    count = 0
    for idea in ideas:
        rank = ranks.get(idea.idx)
        comment = comments.get(idea.idx, "")
        if rank is not None or comment:
            store.insert_feedback(
                Feedback(
                    idea_id=idea.idea_id,
                    user_rank=rank,
                    user_comment=comment,
                    source=source,
                )
            )
            count += 1
    return count


def ingest_replies(store, batch, replies, source: str = "telegram") -> int:
    """Ingest a list of messaging Reply dicts as feedback for `batch`."""
    if not replies:
        return 0
    joined = "\n".join(r["text"] for r in replies)
    return _ingest_text(store, batch, joined, source)


def ingest_cli_feedback(store, batch, text: str, source: str = "cli") -> int:
    """Ingest a single free-text blob (typed at the CLI) as feedback."""
    return _ingest_text(store, batch, text, source)
