"""Prior-art / novelty attachment (DESIGN.md section 7).

For each freshly generated idea we run one cheap search and record the closest
prior art on the idea row. This is advisory: every idea is handled in its own
try/except so one bad query never aborts the rest, and a missing search provider
simply flags ideas as '[unchecked]'.
"""
from __future__ import annotations

import re

from ..providers.base import SearchProvider
from ..store.db import Store
from ..store.models import Idea


def _build_query(idea: Idea) -> str:
    """Compose a search query from the idea title plus a few mechanism words."""
    title = (idea.title or "").strip()
    words = re.findall(r"[A-Za-z0-9\-]+", idea.mechanism or "")
    extra = " ".join(words[:5])
    query = (title + " " + extra).strip()
    return query or title


def attach_prior_art(
    ideas: list[Idea],
    search: SearchProvider | None,
    store: Store,
    *,
    k: int,
    batch_id: str,
) -> None:
    """Set each idea's prior_art field (in memory and in the store).

    When `search` is None the step is a no-op flagging '[unchecked]'. Otherwise
    the top result becomes the prior-art note, retrieval is logged for later
    de-duplication, and any single failure is isolated to its own idea.
    """
    for idea in ideas:
        try:
            if search is None:
                store.update_idea_prior_art(idea.idea_id, "[unchecked]")
                idea.prior_art = "[unchecked]"
                continue

            query = _build_query(idea)
            results = search.search(query, k=k) or []
            if results:
                top = results[0]
                prior_art = (
                    "closest prior art: "
                    + str(top.get("title", "")).strip()
                    + " -- "
                    + str(top.get("url", "")).strip()
                )
            else:
                prior_art = "[no close prior art found]"

            store.update_idea_prior_art(idea.idea_id, prior_art)
            idea.prior_art = prior_art

            urls = [str(r.get("url", "")).strip() for r in results if r.get("url")]
            store.log_retrieval(batch_id, query, "novelty", urls)
        except Exception:
            # Advisory step: never let one idea break the batch.
            continue
