"""Verifier panel (MATINS_UPGRADE_PLAN phases A + B): evidence-anchored per-axis verdicts.

Each verifier scores ONE axis of "is this a good idea" against EXTERNAL evidence and emits a
calibrated `Verdict` (score + confidence + evidence + advisory action). The whole point is that
the judgment points at the WORLD (a real novelty search, a real demand thread), not at the
model's own opinion -- the anti-self-confirmation anchor that lets the daily feedback loop
improve the system safely (see MATINS_UPGRADE_PLAN section 7 on RSI).

Phase A ships the **unique** verifier (the calibrated successor to the lone novelty note, with the
"searched != novel" discipline); phase B adds the **useful** verifier (anchored in observed demand).
`feasible` is phase C. ADVISORY in A/B: verdicts are recorded, not acted on -- no regenerate, no
re-rank yet (phases C/D). Fail-open throughout: any error yields a low-confidence "unverified"
verdict, never a crash, and network-bound verifiers run only when the batch is online.
"""
from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field

from ..store.db import Store
from ..store.models import Idea
from .novelty import build_query_from_fields, format_prior_art

logger = logging.getLogger("matins.verify")


@dataclass
class Verdict:
    axis: str                                   # "unique" | "useful" | "feasible"
    score: float                                # 0..1, higher = better on this axis
    confidence: float                           # 0..1, how well we could ANCHOR it (low when poorly grounded)
    evidence: list[dict] = field(default_factory=list)   # [{"claim":..., "url":...}]
    action: str = "keep"                        # keep | revise | kill  (advisory in A/B)
    note: str = ""

    def to_dict(self) -> dict:
        return {
            "axis": self.axis,
            "score": round(self.score, 2),
            "confidence": round(self.confidence, 2),
            "evidence": self.evidence,
            "action": self.action,
            "note": self.note,
        }


def _evidence(results: list[dict], *, limit: int = 3) -> list[dict]:
    return [{"claim": str(r.get("title", "")).strip(), "url": str(r.get("url", "")).strip()}
            for r in results[:limit] if r.get("title")]


class UniqueVerifier:
    """Anchor: a real novelty search. Calibrated so that 'I searched and found nothing' is a
    LOW-confidence signal, never a confident 'novel' -- the anti-mirage discipline."""

    axis = "unique"

    def __init__(self, search) -> None:
        self.search = search                    # a SearchProvider, or None when offline

    def assess(self, idea: Idea, *, k: int = 5, store: Store | None = None,
               batch_id: str | None = None) -> Verdict:
        # Reuse a prior_art already filled by the in-loop saturation gate -> no second search.
        existing = (idea.prior_art or "").strip()
        if existing and existing != "[unchecked]":
            if existing.startswith("[no close"):
                return Verdict(self.axis, 0.55, 0.30, [],
                               note="no close prior art found earlier -- weak signal, NOT confirmed novel")
            return Verdict(self.axis, 0.45, 0.55, [{"claim": existing}],
                           note="a close prior work exists (from the saturation gate); confirm overlap")
        if self.search is None:
            return Verdict(self.axis, 0.50, 0.0, [], note="offline: novelty unverified")

        query = build_query_from_fields(idea.title, idea.math_structure, idea.mechanism)
        try:
            results = self.search.search(query, k=k) or []
        except Exception as exc:
            logger.warning("unique search failed for %s: %s", idea.idea_id, exc)
            return Verdict(self.axis, 0.50, 0.0, [], note=f"novelty search failed: {exc}")

        # Back-compat: keep filling the prior_art note + retrieval log, exactly like the legacy step.
        note = format_prior_art(results)
        if store is not None:
            store.update_idea_prior_art(idea.idea_id, note)
            idea.prior_art = note
            urls = [str(r.get("url", "")).strip() for r in results if r.get("url")]
            if urls:
                store.log_retrieval(batch_id, query, "novelty", urls)

        if results:
            return Verdict(self.axis, 0.45, 0.55, _evidence(results),
                           note="keyword neighbors found -- a keyword hit is not the same idea; confirm overlap")
        return Verdict(self.axis, 0.55, 0.30, [],
                       note="no close hits -- absence is weak evidence, NOT confirmed novel")


class UsefulVerifier:
    """Anchor: a real demand corpus (Hacker News by default). Usefulness is a claim about the
    WORLD -- someone has this pain -- so it must be evidenced, never asserted from the idea text."""

    axis = "useful"

    def __init__(self, demand_search) -> None:
        self.demand = demand_search             # a SearchProvider over a demand corpus, or None

    def assess(self, idea: Idea, *, k: int = 5, store: Store | None = None,
               batch_id: str | None = None) -> Verdict:
        if self.demand is None:
            return Verdict(self.axis, 0.50, 0.0, [], note="no demand source: usefulness unverified")
        query = build_query_from_fields(idea.title, "", idea.mechanism)
        try:
            results = self.demand.search(query, k=k) or []
        except Exception as exc:
            logger.warning("useful search failed for %s: %s", idea.idea_id, exc)
            return Verdict(self.axis, 0.50, 0.0, [], note=f"demand search failed: {exc}")
        if results:
            return Verdict(self.axis, 0.60, 0.55, _evidence(results),
                           note="observed demand signal -- real discussion / requests exist around this")
        return Verdict(self.axis, 0.35, 0.40, [],
                       note="no observed demand found -- may be latent, but no evidence of pull")


def build_verifiers(cfg, search, *, demand_search=None) -> list:
    """Instantiate the enabled verifiers. `search` is the novelty provider (None offline);
    `demand_search` is the pre-built demand provider for the useful axis (None when offline
    or disabled). feasible is phase C."""
    axes = set(cfg.verify.axes or [])
    verifiers: list = []
    if "unique" in axes:
        verifiers.append(UniqueVerifier(search))
    if "useful" in axes:
        verifiers.append(UsefulVerifier(demand_search))
    return verifiers


def run_panel(ideas: list[Idea], cfg, search, store: Store, *, batch_id: str,
              demand_search=None) -> None:
    """Run the enabled verifier panel over each idea and persist the verdicts.

    Replaces the legacy single novelty step when cfg.verify.axes is non-empty: the unique
    verifier still fills prior_art + the retrieval log (back-compat), and every axis' verdict
    is stored as JSON on the idea. Each idea is isolated in its own try/except so one bad
    verifier never sinks the batch.
    """
    verifiers = build_verifiers(cfg, search, demand_search=demand_search)
    if not verifiers:
        return
    for idea in ideas:
        verdicts: dict = {}
        for v in verifiers:
            try:
                verdict = v.assess(idea, k=cfg.verify.k, store=store, batch_id=batch_id)
            except Exception as exc:                       # never let a verifier break the batch
                logger.warning("verifier %s failed for %s: %s", v.axis, idea.idea_id, exc)
                verdict = Verdict(v.axis, 0.50, 0.0, [], note=f"verifier error: {exc}")
            verdicts[verdict.axis] = verdict.to_dict()
        payload = json.dumps(verdicts, ensure_ascii=False)
        store.update_idea_verdicts(idea.idea_id, payload)
        idea.verdicts = payload
