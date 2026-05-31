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
# A 'must try #N' marker -> copy idea N into the curated favorites.
# Tolerates 'must try 3', 'must-try #3', 'musttry#3' (case-insensitive).
_MUST_TRY_RE = re.compile(r"must[\s\-]?try\s*#?\s*(\d+)", re.IGNORECASE)
# A 'dig #N' / 'deep dive #N' marker -> run an on-demand deep-dive briefing.
_DIG_RE = re.compile(r"\b(?:dig|deep[\s\-]?dive)\b\s*#?\s*(\d+)", re.IGNORECASE)


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
        # Strip a leading ':' / '：' so '#1: note' yields 'note', not ': note'.
        body = m.group(2).strip().lstrip(":：").strip()
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


COMMENT_KINDS = ("taste", "novelty", "feasibility", "structure")


def classify_comment(llm, comment: str) -> str:
    """Classify a feedback comment into ONE routing channel (algo-update.md #3).

    Different kinds of comment update different parts of the system: 'already done'
    is novelty evidence, 'boring topic' is taste, 'can't build it' is feasibility,
    'sloppy framing' is structure. Routing them apart eases the credit-assignment
    ambiguity of a single lumped comment field.

    Advisory: empty comment -> "" (no channel); any failure or unrecognized reply
    defaults to 'taste' (the broadest channel) so a flaky call never drops signal.
    """
    text = (comment or "").strip()
    if not text:
        return ""
    prompt = (
        "Classify this one-line reaction to a research idea into exactly ONE channel:\n"
        "- taste: about the topic / how interesting or relevant it is\n"
        "- novelty: it has already been done, prior art, not original\n"
        "- feasibility: it cannot be built / done, too hard, a resource problem\n"
        "- structure: its framing, rigor, or how it is argued / built\n\n"
        f"Reaction: {text}\n\n"
        "Answer with ONE word: taste, novelty, feasibility, or structure."
    )
    try:
        raw = (llm.generate(prompt, temperature=0.0) or "").strip().lower()
    except Exception:
        return "taste"
    for k in COMMENT_KINDS:
        if k in raw:
            return k
    return "taste"


def _ingest_text(store, batch, text: str, source: str, *, classify=None) -> int:
    """Parse `text` against the batch's ideas and insert one Feedback per hit.

    `classify`, if given, is a callable comment->kind used to tag each non-empty
    comment with a routing channel. Left None (offline / tests) -> kind stays "".
    """
    ideas = store.ideas_for_batch(batch.batch_id)
    ranks, comments, _ = parse_ranking(text, len(ideas))
    count = 0
    for idea in ideas:
        rank = ranks.get(idea.idx)
        comment = comments.get(idea.idx, "")
        if rank is not None or comment:
            kind = classify(comment) if (classify and comment) else ""
            store.insert_feedback(
                Feedback(
                    idea_id=idea.idea_id,
                    user_rank=rank,
                    user_comment=comment,
                    comment_kind=kind,
                    source=source,
                )
            )
            count += 1
    return count


def ingest_replies(store, batch, replies, source: str = "telegram", *, classify=None) -> int:
    """Ingest a list of messaging Reply dicts as feedback for `batch`."""
    if not replies:
        return 0
    joined = "\n".join(r["text"] for r in replies)
    return _ingest_text(store, batch, joined, source, classify=classify)


def ingest_cli_feedback(store, batch, text: str, source: str = "cli", *, classify=None) -> int:
    """Ingest a single free-text blob (typed at the CLI) as feedback."""
    return _ingest_text(store, batch, text, source, classify=classify)


def parse_must_try(text: str, n_ideas: int) -> list[int]:
    """Extract idea indices flagged 'must try #N' (curated favorites), in order,
    de-duplicated, bounded to 1..n_ideas. Tolerant; never raises."""
    out: list[int] = []
    for m in _MUST_TRY_RE.finditer(text or ""):
        idx = int(m.group(1))
        if 1 <= idx <= n_ideas and idx not in out:
            out.append(idx)
    return out


def parse_dig(text: str, n_ideas: int) -> list[int]:
    """Extract idea indices flagged 'dig #N' / 'deep dive #N', de-duplicated,
    bounded to 1..n_ideas. Tolerant; never raises."""
    out: list[int] = []
    for m in _DIG_RE.finditer(text or ""):
        idx = int(m.group(1))
        if 1 <= idx <= n_ideas and idx not in out:
            out.append(idx)
    return out


def ingest_must_try(store, batch, replies) -> list:
    """Copy ideas flagged 'must try #N' in `replies` into the curated favorites.

    Returns the list of Idea objects newly added (ideas already favorited are
    skipped, so re-collecting the same reply is idempotent).
    """
    if not replies:
        return []
    text = "\n".join(r.get("text", "") for r in replies)
    ideas = store.ideas_for_batch(batch.batch_id)
    by_idx = {i.idx: i for i in ideas}
    added = []
    for idx in parse_must_try(text, len(ideas)):
        idea = by_idx.get(idx)
        if idea is not None and store.add_favorite(idea.idea_id):
            added.append(idea)
    return added
