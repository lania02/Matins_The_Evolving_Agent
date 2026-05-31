"""Self-evolution orchestration (Phase 5 §2.1-2.3).

The step that takes the project past self-training: propose a genuinely NEW taste
dimension from persistent residuals (§2.1), VERIFY it on held-out data via backtest.py
(§2.2 -- the RLVR-style verifiable gate), and only if it earns its place, park it as a
human-approvable skill version (§2.3, Assisted mode -- reuses the existing approval +
versioning + rollback). The generator's variation slots are untouched, so selection
pressure never collapses into Goodhart.

Train/holdout discipline: BOTH the raw events AND the divergence hypotheses fed to the
proposer are restricted to train batches, so no holdout-derived signal steers the
proposal the held-out tau later "verifies". Hypotheses are an all-history artifact, so
they are re-scoped to train-only evidence here before reaching the proposer.

OFF by default and data-gated: with ~4 ideas/day the held-out set is small, so early on
the verifier is weak (see backtest.py "honest limits"). Enable via
consolidation.evolve_dimensions once enough log has accumulated; until then it reports
rather than acts.
"""
from __future__ import annotations

import json
from dataclasses import replace

from ..generate.slots import load_prompt, render_template
from .backtest import backtest_dimension
from .kernels import format_events

# Fraction of the (oldest-first) comparable history held out for verification, and the
# data gate below which dimension evolution is not even attempted. Raised so the gate
# fires only with >=3 held-out batches (one lucky ordering flip cannot carry the verdict).
_HOLDOUT_FRAC = 0.34
_MIN_TOTAL_BATCHES = 8
# Above this lexical overlap with the current skill, flag the proposal as a likely
# reweight rather than a new axis (a human-facing caution, not a hard gate).
_REWEIGHT_OVERLAP = 0.6


def _comparable_batches(store) -> list:
    """Batches (oldest first) with >=2 ideas carrying BOTH a self_rank and a user_rank.

    self_rank is required because the train split feeds the proposer, which mines the
    self_rank-vs-user_rank residual ([+underrated]); the held-out backtest itself only
    needs user_rank, so this gate is intentionally stricter than the verifier needs.
    """
    out = []
    for b in reversed(store.list_batches()):          # list_batches is newest-first
        n = 0
        for i in store.ideas_for_batch(b.batch_id):
            if i.self_rank is None:
                continue
            fb = store.feedback_for_idea(i.idea_id)
            if fb is not None and fb.user_rank is not None:
                n += 1
        if n >= 2:
            out.append(b)
    return out


def _train_only_hypotheses(store, hyps, train_ids: set) -> list:
    """Re-scope hypotheses to TRAIN evidence so no holdout signal reaches the proposer.

    Each hypothesis's evidence is a JSON list of idea_ids; keep only those in train
    batches, drop hypotheses left with none, and recompute occurrence as the number of
    distinct train batches the surviving evidence touches.
    """
    idea_to_batch = {}
    for bid in train_ids:
        for i in store.ideas_for_batch(bid):
            idea_to_batch[i.idea_id] = bid
    out = []
    for h in hyps:
        try:
            ev = json.loads(h.evidence or "[]")
        except (ValueError, TypeError):
            ev = []
        train_ev = [iid for iid in ev if iid in idea_to_batch]
        if not train_ev:
            continue
        touched = {idea_to_batch[iid] for iid in train_ev}
        out.append(replace(h, evidence=json.dumps(train_ev), occurrence=len(touched)))
    return out


def _render_hypotheses(hyps) -> str:
    if not hyps:
        return "(none yet)"
    return "\n".join(f"- [{h.kind}] {h.text} (seen x{h.occurrence})" for h in hyps)


def _propose_dimension(cfg, store, llm, train_ids: set, hyps) -> str:
    """LLM proposes ONE new dimension grounded in train-only events + hypotheses, or 'NONE'."""
    all_events = store.recent_events(100000, 1)
    train_events = [e for e in all_events if e.get("batch_id") in train_ids]
    cur = store.active_skill()
    prompt = render_template(load_prompt(cfg.prompts_dir(), "propose_dimension.txt"), {
        "CURRENT_SKILL": cur.content if cur else "(none yet)",
        "HYPOTHESES": _render_hypotheses(hyps),
        "EVENTS": format_events(train_events),
    })
    try:
        return (llm.generate(prompt, temperature=0.2) or "").strip()
    except Exception:
        return "NONE"


def evolve_dimension(cfg, store, llm, messaging) -> dict:
    """Propose -> backtest -> (if it earns its place) park a human-approvable dimension.

    Returns a dict with a 'message'. Never edits the skill directly: a passing dimension
    is inserted unapproved, exactly like a normal consolidation proposal.
    """
    if not getattr(cfg.consolidation, "evolve_dimensions", False):
        return {"message": "dimension evolution disabled (consolidation.evolve_dimensions=false)"}

    threshold = cfg.consolidation.hypothesis_occurrence_threshold
    hyps = store.hypotheses_over_threshold(threshold)
    if not hyps:
        return {"message": "no persistent divergence pattern over threshold yet; evolution deferred"}

    batches = _comparable_batches(store)
    if len(batches) < _MIN_TOTAL_BATCHES:
        return {"message": f"insufficient data ({len(batches)} comparable batches < "
                f"{_MIN_TOTAL_BATCHES}); evolution deferred"}

    h = max(2, round(len(batches) * _HOLDOUT_FRAC))
    train, holdout = batches[:-h], batches[-h:]
    if not train:
        return {"message": "insufficient training history; evolution deferred"}

    train_ids = {b.batch_id for b in train}
    train_hyps = _train_only_hypotheses(store, hyps, train_ids)
    if not train_hyps:
        return {"message": "the persistent pattern is not present in the training window; "
                "evolution deferred"}

    proposal = _propose_dimension(cfg, store, llm, train_ids, train_hyps)
    if not proposal or proposal.strip().upper().startswith("NONE"):
        return {"message": "no new dimension warranted by current evidence"}

    cur = store.active_skill()
    base_skill = cur.content if cur else ""
    result = backtest_dimension(cfg, store, llm, proposal, holdout, base_skill)
    if not result.get("passed"):
        return {"message": "proposed a dimension but it FAILED the held-out backtest "
                f"(status={result['status']}, mean_delta_tau={result['mean_delta_tau']}); not adopted",
                "proposal": proposal, "backtest": result}

    # Earned its place: park as an unapproved skill version (Assisted -- human approves).
    overlap = result.get("vocab_overlap") or 0.0
    likely_reweight = overlap > _REWEIGHT_OVERLAP
    label = ("skill refinement (high lexical overlap -- review as a possible reweight, not a new axis)"
             if likely_reweight else "evolved taste dimension")
    diff = (f"{label}; held-out delta-tau={result['mean_delta_tau']:.3f} "
            f"(on base-failures={result['mean_delta_on_base_failures']}), "
            f"lexical-overlap={overlap:.0%}, n={result['n']}")
    new_content = (base_skill + "\n\n" + proposal).strip()
    version = store.insert_skill_version(new_content, cur.version if cur else None, diff, approved=0)
    if messaging is not None:
        try:
            messaging.send(
                f"Candidate taste-skill update (v{version}) -- {label}. It passed the held-out "
                f"backtest: delta-tau={result['mean_delta_tau']:.3f} over {result['n']} batches, "
                f"lift where the base was wrong={result['mean_delta_on_base_failures']}, "
                f"lexical overlap with current skill={overlap:.0%}.\n\n"
                + proposal[:2400]
                + f"\n\nApprove with: matins consolidate --approve {version}")
        except Exception:
            pass
    return {"message": f"{label} passed backtest; parked as skill v{version} awaiting approval",
            "version": version, "backtest": result}
