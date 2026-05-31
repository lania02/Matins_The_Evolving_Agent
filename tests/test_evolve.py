"""Offline tests for self-evolution orchestration (matins.memory.evolve).

Exercises the full Phase 5 loop with one FakeLLM that answers BOTH prompt types: it
proposes a dimension carrying a marker, and (as the skill-conditioned scorer) only
predicts the user's true order when that marker is in the skill it is shown -- so a
genuinely-predictive dimension passes the held-out backtest and gets parked for approval.
Also pins the train/holdout discipline: a hypothesis grounded only in holdout batches
must NOT steer the proposer.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from matins.config import load_config
from matins.memory.evolve import evolve_dimension
from matins.store.db import Store, new_id, now_iso
from matins.store.models import Batch, Feedback, Idea, TasteHypothesis

REPO_ROOT = Path(__file__).resolve().parent.parent
_MARK = "CROSSDOMAIN-BRIDGE"
_SLOTS = ["highfit", "adjacent", "orthogonal", "random"]


class FakeEvolveLLM:
    def generate(self, prompt, *, temperature, json_schema=None):
        if "PERSISTENT DIVERGENCE PATTERNS" in prompt:          # the propose_dimension prompt
            return f"### Cross-domain bridge\nRewards {_MARK} ideas.\nEvidence: recurring [+underrated] events."
        idxs = sorted({int(m) for m in re.findall(r"#(\d+) \[", prompt)})   # predict_rank prompt
        order = list(reversed(idxs)) if _MARK in prompt else idxs
        return json.dumps([{"idx": idx, "rank": r + 1, "rationale": "r"} for r, idx in enumerate(order)])


def _cfg(evolve: bool):
    cfg = load_config(str(REPO_ROOT / "config.example.yaml"))
    cfg.consolidation.evolve_dimensions = evolve
    return cfg


def _seed(store, n):
    """n comparable batches (4 ideas each): self ranks forward, user prefers reverse.

    Returns [(batch, [ideas])] oldest-first so a test can ground a hypothesis in chosen
    (train or holdout) batches.
    """
    seeded = []
    for d in range(n):
        b = Batch(batch_id=new_id(), date=f"2026-03-{d + 1:02d}", provider="x", model="m",
                  created_at=now_iso())
        store.insert_batch(b)
        ideas = []
        for idx in range(1, 5):
            i = Idea(idea_id=new_id(), batch_id=b.batch_id, slot=_SLOTS[idx - 1], idx=idx,
                     title=f"i{idx}", mechanism="m", self_rank=idx)
            store.insert_idea(i)
            store.insert_feedback(Feedback(idea_id=i.idea_id, user_rank=5 - idx, source="cli"))
            ideas.append(i)
        seeded.append((b, ideas))
    return seeded


def _add_hypothesis(store, evidence_ids, occurrence=3):
    store.upsert_hypothesis(TasteHypothesis(
        hyp_id=new_id(), text="underrates cross-domain bridges", kind="structure",
        evidence=json.dumps(evidence_ids), confidence=0.6, occurrence=occurrence, status="open"))


def test_evolution_disabled_by_default():
    store = Store(":memory:")
    _seed(store, 8)
    _add_hypothesis(store, [])
    assert "disabled" in evolve_dimension(_cfg(False), store, FakeEvolveLLM(), None)["message"]


def test_evolution_needs_a_threshold_hypothesis():
    store = Store(":memory:")
    _seed(store, 8)                                    # plenty of data, but no persistent pattern
    res = evolve_dimension(_cfg(True), store, FakeEvolveLLM(), None)
    assert "no persistent divergence pattern" in res["message"]


def test_evolution_data_gated_on_thin_history():
    store = Store(":memory:")
    _seed(store, 6)                                    # below _MIN_TOTAL_BATCHES (8)
    _add_hypothesis(store, [])
    assert "insufficient data" in evolve_dimension(_cfg(True), store, FakeEvolveLLM(), None)["message"]


def test_holdout_only_hypothesis_does_not_steer_proposer():
    # Leak guard: a pattern grounded ONLY in held-out batches must not reach the proposer.
    store = Store(":memory:")
    seeded = _seed(store, 8)
    _add_hypothesis(store, [seeded[-1][1][0].idea_id])  # newest batch -> holdout side
    res = evolve_dimension(_cfg(True), store, FakeEvolveLLM(), None)
    assert "training window" in res["message"]


def test_evolution_parks_a_backtest_passing_dimension():
    store = Store(":memory:")
    seeded = _seed(store, 8)
    # Ground the hypothesis in the three OLDEST batches (the train side).
    _add_hypothesis(store, [seeded[i][1][0].idea_id for i in range(3)])

    res = evolve_dimension(_cfg(True), store, FakeEvolveLLM(), None)

    assert "version" in res                            # a dimension earned its place
    assert res["backtest"]["passed"] is True
    assert res["backtest"]["mean_delta_tau"] > 0
    pending = store.pending_skill_version()            # parked, NOT auto-approved
    assert pending is not None and pending.approved == 0
    assert _MARK in pending.content                    # the evolved dimension is in it
    assert store.active_skill() is None                # nothing activated without human approval
