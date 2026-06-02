"""Prior-art / novelty attachment (DESIGN.md section 7).

For each freshly generated idea we run one cheap search and record the closest
prior art on the idea row. This is advisory: every idea is handled in its own
try/except so one bad query never aborts the rest, and a missing search provider
simply flags ideas as '[unchecked]'.
"""
from __future__ import annotations

import logging
import re

from ..providers.base import SearchProvider
from ..store.db import Store
from ..store.models import Idea

logger = logging.getLogger("matins.novelty")


# Generic words that hurt arXiv relevance ranking if left in the query.
_STOPWORDS = {
    "a", "an", "the", "of", "for", "and", "or", "to", "in", "on", "with", "via",
    "using", "from", "by", "as", "at", "into", "under", "over", "is", "are", "be",
    "this", "that", "analysis", "approach", "method", "methods", "towards",
    "toward", "novel", "new", "study", "based", "framework",
}


# English terms annotated inside (parentheses) or （全角括号）, e.g. 谱半径(spectral radius).
_PAREN_RE = re.compile(r"[(（]([^)）]*[A-Za-z][^)）]*)[)）]")


def _english_terms(text: str) -> list[str]:
    """Extract English technical terms annotated in parentheses (bilingual gloss)."""
    out: list[str] = []
    for inner in _PAREN_RE.findall(text or ""):
        phrase = re.sub(r"[^A-Za-z0-9 \-]", " ", inner.split(",")[0])
        phrase = re.sub(r"\s+", " ", phrase).strip()
        if phrase:
            out.append(phrase)
    return out


def build_query_from_fields(title: str, math_structure: str = "", mechanism: str = "") -> str:
    """Build a focused English keyword query from idea fields.

    With Chinese-primary output the title is Chinese, so we mine the English terms
    the model annotates in parentheses across title/math/mechanism, plus any ASCII
    acronyms. arXiv is an English corpus -- a Chinese query returns nothing useful.
    Keywords are de-duplicated, stopword-filtered, and capped at four (the arXiv
    adapter ANDs them, so fewer = less likely to over-constrain to zero hits).

    Shared by the post-generation novelty check and the saturation gate, so both ask
    the corpus the same question about a given idea.
    """
    blob = " ".join([title or "", math_structure or "", mechanism or ""])
    candidates: list[str] = []
    for term in _english_terms(blob):                                       # parenthetical glosses
        candidates.extend(term.split())
    candidates.extend(re.findall(r"[A-Za-z][A-Za-z0-9\-]+", title or ""))            # title ASCII
    # math_structure is terse and term-dense -- the DOMAIN + method English often lives
    # here unglossed. Mining it keeps the query specific to the idea (method AND domain)
    # rather than collapsing to a bare popular method name, whose corpus count would
    # measure the method's popularity, not THIS idea's crowdedness.
    candidates.extend(re.findall(r"[A-Za-z][A-Za-z0-9\-]+", math_structure or ""))

    seen: set[str] = set()
    words: list[str] = []
    for w in candidates:
        lw = w.lower()
        if len(lw) < 2 or lw in _STOPWORDS or lw in seen:
            continue
        seen.add(lw)
        words.append(w)

    chosen = words[:4]
    return " ".join(chosen) if chosen else (title or "").strip()


def _build_query(idea: Idea) -> str:
    return build_query_from_fields(idea.title, idea.math_structure, idea.mechanism)


def format_prior_art(results: list[dict]) -> str:
    """Format the closest-prior-art note from a search result list (top hit, or none)."""
    if results:
        top = results[0]
        return (
            "closest prior art: "
            + str(top.get("title", "")).strip()
            + " -- "
            + str(top.get("url", "")).strip()
        )
    return "[no close prior art found]"


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
            if idea.prior_art and idea.prior_art not in ("[unchecked]", ""):
                # Already noted by the in-loop saturation gate's search; don't search twice.
                continue
            if search is None:
                store.update_idea_prior_art(idea.idea_id, "[unchecked]")
                idea.prior_art = "[unchecked]"
                continue

            query = _build_query(idea)
            results = search.search(query, k=k) or []
            prior_art = format_prior_art(results)

            store.update_idea_prior_art(idea.idea_id, prior_art)
            idea.prior_art = prior_art

            urls = [str(r.get("url", "")).strip() for r in results if r.get("url")]
            store.log_retrieval(batch_id, query, "novelty", urls)
        except Exception as exc:
            # Advisory step: never let one idea break the batch -- but log it, so a
            # systematically failing search isn't silently read as "no prior art".
            logger.warning("prior-art search failed for idea %s: %s", idea.idea_id, exc)
            continue
