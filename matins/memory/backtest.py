"""Held-out backtest: the verifiable gate that turns self-training into self-evolution.

Phase 5 §2.2. A candidate taste dimension earns its place ONLY if, on held-out batches
the proposer never saw, conditioning the skill-aware scorer on it improves agreement
(Kendall tau) with the user's ACTUAL ranks. This is the project's analogue of RLVR: the
reward is grounded in out-of-sample human data, so a dimension cannot be minted from
noise. No gradients: the "policy" is the natural-language skill and the "update" is
human-approved consolidation.

HONEST LIMITS (do not overclaim):
- Proposer and verifier are the SAME model, and the verifier is shown the proposal's
  text, so the tau-lift is mediated by one model's reading of its own words. The held-out
  human ranks anchor the reward (a self-consistent but human-irrelevant reorder scores
  delta-tau<=0 and is rejected) but do NOT fully decouple it from the model's priors.
  Human approval is the only fully independent check (Assisted mode).
- To distinguish a genuinely NEW axis from a louder reweight of an existing one, the gate
  requires the lift to show up where the base skill is WRONG (a reweight mostly sharpens
  orderings the base already gets right). It also reports a lexical-overlap signal for the
  human. Neither is a placebo-controlled proof of novelty -- that, and an independent
  verifier model, are deferred (algo-upgrade-plan.md Phase 5 "known limits").
- Small-sample power is low at the data scale this runs at; hence off-by-default,
  data-gated, and human-approved.
"""
from __future__ import annotations

import re

from ..feedback.diverge import kendall_tau

# Verdict thresholds. Module constants (policy levers with no current variation point).
_MARGIN = 0.05          # mean held-out Delta-tau a dimension must clear to "earn its place"
_MIN_HOLDOUT = 2        # need at least this many comparable held-out batches to judge


def score_under_skill(llm, ideas, skill_text, prompts_dir, output_language) -> dict:
    """Predict the user's ranking of `ideas` under `skill_text`. Returns {idx: rank}.

    Advisory: any failure or unparseable reply -> {} (the caller drops that fold), so a
    flaky scoring call never crashes the consolidation step.
    """
    from ..generate.slots import build_predict_rank_prompt, parse_self_ranks

    if len(ideas) < 2:
        return {}
    prompt = build_predict_rank_prompt(ideas, skill_text, prompts_dir, output_language)
    try:
        text = llm.generate(prompt, temperature=0.0)
        ranks = parse_self_ranks(text, len(ideas))
    except Exception:
        return {}
    return {r["idx"]: r["rank"] for r in ranks}


def _is_permutation(predicted: dict, want: set) -> bool:
    """True iff `predicted` covers exactly `want` with distinct ranks 1..len(want).

    A partial / duplicated scorer reply is unreliable evidence: scoring it would compare
    tau_without and tau_with over DIFFERENT idea subsets, so the per-batch delta would
    conflate a taste lift with a change in which ideas were ranked. Reject such folds.
    """
    if set(predicted.keys()) != want:
        return False
    return sorted(predicted.values()) == list(range(1, len(want) + 1))


def _tau_vs_user(ideas, predicted: dict, user: dict) -> float | None:
    """Kendall tau between predicted ranks and the user's actual ranks, aligned by idea."""
    a: list[int] = []
    b: list[int] = []
    for idea in ideas:
        if idea.idx in predicted and idea.idx in user:
            a.append(predicted[idea.idx])
            b.append(user[idea.idx])
    if len(a) < 2:
        return None
    return kendall_tau(a, b)


def _tokens(s: str) -> set:
    """Crude lexical tokens for an overlap heuristic (English words + CJK runs)."""
    return {w for w in re.findall(r"[0-9a-z一-鿿]+", (s or "").lower()) if len(w) > 2}


def vocab_overlap(base_skill: str, dimension_text: str) -> float:
    """Fraction of the candidate's tokens already present in the skill. 1.0 = fully
    restated (likely a reweight), 0.0 = orthogonal vocabulary. A soft human-facing
    signal, NOT a gate (lexical overlap is not semantic novelty)."""
    b, d = _tokens(base_skill), _tokens(dimension_text)
    return len(b & d) / len(d) if d else 1.0


def backtest_dimension(cfg, store, llm, dimension_text: str, holdout_batches,
                       base_skill: str) -> dict:
    """Measure a candidate dimension's out-of-sample lift on held-out batches.

    For each held-out batch, predict the user's ranking under `base_skill` and under
    base_skill + the candidate, both over the SAME comparable idea set, and compare each
    to the user's real ranks. Delta-tau = tau_with - tau_without is the per-batch lift.

    Returns {status, n, mean_delta_tau, frac_positive, mean_delta_on_base_failures,
    vocab_overlap, passed}. status='insufficient_data' below _MIN_HOLDOUT comparable
    folds. `passed` requires a positive mean lift, a STRICT majority of folds positive,
    and -- the novelty check -- a real lift where the base skill was WRONG (so a pure
    reweight that only sharpens already-correct orderings cannot pass).
    """
    prompts_dir = cfg.prompts_dir()
    lang = cfg.generation.output_language
    skill_with = (base_skill + "\n\n" + dimension_text).strip()

    deltas: list[float] = []
    deltas_on_base_failures: list[float] = []
    for b in holdout_batches:
        ideas = store.ideas_for_batch(b.batch_id)
        user = {}
        for i in ideas:
            fb = store.feedback_for_idea(i.idea_id)
            if fb is not None and fb.user_rank is not None:
                user[i.idx] = fb.user_rank
        comparable = [i for i in ideas if i.idx in user]
        if len(comparable) < 2:
            continue
        want = {i.idx for i in comparable}
        pred_without = score_under_skill(llm, comparable, base_skill, prompts_dir, lang)
        pred_with = score_under_skill(llm, comparable, skill_with, prompts_dir, lang)
        if not _is_permutation(pred_without, want) or not _is_permutation(pred_with, want):
            continue                                   # malformed/partial fold -> unreliable
        tau_without = _tau_vs_user(comparable, pred_without, user)
        tau_with = _tau_vs_user(comparable, pred_with, user)
        if tau_without is None or tau_with is None:
            continue
        delta = tau_with - tau_without
        deltas.append(delta)
        if tau_without < 1.0:        # base got this batch (partly) WRONG: where a new axis must help
            deltas_on_base_failures.append(delta)

    if len(deltas) < _MIN_HOLDOUT:
        return {"status": "insufficient_data", "n": len(deltas), "mean_delta_tau": None,
                "frac_positive": None, "mean_delta_on_base_failures": None,
                "vocab_overlap": None, "passed": False}

    mean_delta = sum(deltas) / len(deltas)
    frac_positive = sum(1 for d in deltas if d > 0) / len(deltas)
    mean_fail = (sum(deltas_on_base_failures) / len(deltas_on_base_failures)
                 if deltas_on_base_failures else None)
    # Novelty check: when the base skill is imperfect somewhere, a genuinely new axis must
    # lift tau THERE. (If the base is already perfect everywhere, no dimension can clear
    # the mean-margin anyway, so this is vacuously satisfied.)
    novelty_ok = mean_fail is None or mean_fail >= _MARGIN
    passed = mean_delta >= _MARGIN and frac_positive > 0.5 and novelty_ok
    return {"status": "ok", "n": len(deltas), "mean_delta_tau": mean_delta,
            "frac_positive": frac_positive, "mean_delta_on_base_failures": mean_fail,
            "vocab_overlap": vocab_overlap(base_skill, dimension_text), "passed": passed}
