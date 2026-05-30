"""Single-batch divergence: compare the model's self-ranking to the human's.

We score agreement with Kendall's tau. Low agreement (the model weighted the
wrong feature) is the signal that the taste skill may be miscalibrated -- but a
single batch is weak evidence, so here we only *record a hypothesis* (FAST
memory). Promoting a hypothesis into an actual skill edit happens elsewhere
(memory/consolidate.py) and always requires the recurrence threshold + human
approval. This step NEVER edits a skill.
"""
from __future__ import annotations

import json

from ..store.db import new_id
from ..store.models import TasteHypothesis

# Below this tau the self/human orderings disagree enough to log a hypothesis.
_DIVERGENCE_THRESHOLD = 0.5


def kendall_tau(order_a, order_b) -> float:
    """Kendall's tau-a between two equal-length rank lists aligned by item.

    Counts concordant (C) and discordant (D) pairs:
        tau = (C - D) / (n*(n-1)/2)
    Identical orderings -> 1.0, exact reverse -> -1.0. Fewer than 2 items -> 0.0.
    """
    n = len(order_a)
    if n < 2:
        return 0.0
    concordant = 0
    discordant = 0
    for i in range(n):
        for j in range(i + 1, n):
            da = order_a[i] - order_a[j]
            db = order_b[i] - order_b[j]
            prod = da * db
            if prod > 0:
                concordant += 1
            elif prod < 0:
                discordant += 1
            # prod == 0 (a tie on either side) counts as neither.
    denom = n * (n - 1) / 2
    return (concordant - discordant) / denom


def reflect_on_batch(cfg, store, llm, batch) -> float | None:
    """Score self/human agreement for a batch; log a hypothesis if it diverges.

    Returns the computed tau, or None when there are too few comparable pairs.
    """
    ideas = store.ideas_for_batch(batch.batch_id)

    self_list: list[int] = []
    user_list: list[int] = []
    idea_ids: list[str] = []
    comments: list[str] = []
    for idea in ideas:
        if idea.self_rank is None:
            continue
        fb = store.feedback_for_idea(idea.idea_id)
        if fb is None or fb.user_rank is None:
            continue
        self_list.append(idea.self_rank)
        user_list.append(fb.user_rank)
        idea_ids.append(idea.idea_id)
        if fb.user_comment:
            comments.append(f"#{idea.idx} {fb.user_comment}")

    if len(self_list) < 2:
        store.set_batch_tau(batch.batch_id, None)
        return None

    tau = kendall_tau(self_list, user_list)
    store.set_batch_tau(batch.batch_id, tau)

    if tau < _DIVERGENCE_THRESHOLD:
        try:
            comment_block = "\n".join(comments) if comments else "(no comments)"
            prompt = (
                "My self-ranking of today's ideas diverged sharply from the "
                "human's ranking (Kendall tau = "
                f"{tau:.2f}).\n\n"
                "Self ranks (lower is better): "
                f"{self_list}\n"
                "Human ranks (lower is better): "
                f"{user_list}\n\n"
                "Human comments:\n"
                f"{comment_block}\n\n"
                "In ONE sentence, name the single feature I most likely "
                "mis-weighted, and classify it as either 'topic' (what the idea "
                "is about) or 'structure' (how the idea is built/argued).\n"
                "Answer in exactly this format: kind|sentence\n"
                "where kind is 'topic' or 'structure'."
            )
            raw = llm.generate(prompt, temperature=0.3)
            kind, text = _parse_kind_sentence(raw)

            existing = store.find_hypothesis(text)
            if existing is not None:
                existing.occurrence += 1
                existing.confidence = min(1.0, existing.confidence + 0.2)
                if idea_ids:
                    merged = _merge_evidence(existing.evidence, idea_ids)
                    existing.evidence = merged
                store.upsert_hypothesis(existing)
            else:
                store.upsert_hypothesis(
                    TasteHypothesis(
                        hyp_id=new_id(),
                        text=text,
                        kind=kind,
                        evidence=json.dumps(idea_ids),
                        confidence=0.3,
                        occurrence=1,
                        status="open",
                    )
                )
        except Exception:
            # Reflection is advisory; never crash the daily loop on it.
            pass

    return tau


def _parse_kind_sentence(raw: str) -> tuple[str, str]:
    """Parse a 'kind|sentence' response. Tolerant; defaults kind='structure'."""
    text = (raw or "").strip()
    kind = "structure"
    if "|" in text:
        head, _, tail = text.partition("|")
        head = head.strip().lower()
        if head in ("topic", "structure"):
            kind = head
        text = tail.strip()
    if not text:
        text = (raw or "").strip()
    return kind, text


def _merge_evidence(existing_evidence: str, new_ids: list[str]) -> str:
    """Union new idea_ids into an existing JSON-list evidence string."""
    try:
        current = json.loads(existing_evidence or "[]")
        if not isinstance(current, list):
            current = []
    except (ValueError, TypeError):
        current = []
    for i in new_ids:
        if i not in current:
            current.append(i)
    return json.dumps(current)
