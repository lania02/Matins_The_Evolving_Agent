"""Offline tests for the held-out backtest -- the verifiable gate of self-evolution.

The crux of Phase 5: a candidate taste dimension must lift out-of-sample agreement
with the user's real ranks (reward a REAL dimension, REJECT a noise one). Batches carry
4 ideas (the production regime, 6 comparable pairs), and the FakeLLM scores the user's
true order ONLY when the skill it is shown contains the hidden dimension marker.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from matins.config import load_config
from matins.memory.backtest import backtest_dimension, score_under_skill, vocab_overlap
from matins.store.db import Store, new_id, now_iso
from matins.store.models import Batch, Feedback, Idea

REPO_ROOT = Path(__file__).resolve().parent.parent
_MARK = "CROSSDOMAIN-BRIDGE"
_SLOTS = ["highfit", "adjacent", "orthogonal", "random"]


class FakeScorer:
    """Predicts the user's true (reversed-idx) order ONLY when the skill carries _MARK.

    Without the marker it predicts forward idx order, which is the reverse of the seeded
    user preference -- so a skill containing the real dimension scores a higher tau and an
    irrelevant ('noise') dimension does not move the score.
    """

    def generate(self, prompt, *, temperature, json_schema=None):
        idxs = sorted({int(m) for m in re.findall(r"#(\d+) \[", prompt)})
        order = list(reversed(idxs)) if _MARK in prompt else idxs
        return json.dumps([{"idx": idx, "rank": r + 1, "rationale": "r"} for r, idx in enumerate(order)])


def _cfg():
    return load_config(str(REPO_ROOT / "config.example.yaml"))


def _seed_holdout(store, n_batches=3):
    """n batches of 4 ideas; the user prefers the higher-idx idea (user_rank = 5 - idx)."""
    holdout = []
    for d in range(n_batches):
        b = Batch(batch_id=new_id(), date=f"2026-02-{d + 1:02d}", provider="x", model="m",
                  created_at=now_iso())
        store.insert_batch(b)
        for idx in range(1, 5):
            i = Idea(idea_id=new_id(), batch_id=b.batch_id, slot=_SLOTS[idx - 1], idx=idx,
                     title=f"i{idx}", mechanism="m")
            store.insert_idea(i)
            store.insert_feedback(Feedback(idea_id=i.idea_id, user_rank=5 - idx, source="cli"))
        holdout.append(b)
    return holdout


def test_backtest_rewards_real_dimension_and_rejects_noise():
    store = Store(":memory:")
    holdout = _seed_holdout(store)
    cfg, llm = _cfg(), FakeScorer()

    real = backtest_dimension(
        cfg, store, llm, f"Principle: rewards {_MARK} ideas.", holdout, base_skill="(cold start)")
    noise = backtest_dimension(
        cfg, store, llm, "Principle: likes the colour blue.", holdout, base_skill="(cold start)")

    assert real["status"] == "ok" and real["passed"] is True
    assert real["mean_delta_tau"] > 0                      # real dimension lifts held-out tau
    assert real["mean_delta_on_base_failures"] >= 0.05     # and lifts where the base was wrong
    assert noise["passed"] is False                        # noise earns no place
    assert abs(noise["mean_delta_tau"]) < 1e-9             # and moves the score not at all


def test_backtest_data_gate_blocks_thin_history():
    store = Store(":memory:")
    holdout = _seed_holdout(store, n_batches=1)             # below _MIN_HOLDOUT
    res = backtest_dimension(_cfg(), store, FakeScorer(), _MARK, holdout, base_skill="")
    assert res["status"] == "insufficient_data" and res["passed"] is False


def test_score_under_skill_returns_idx_to_rank():
    store = Store(":memory:")
    ideas = store.ideas_for_batch(_seed_holdout(store, 1)[0].batch_id)
    out = score_under_skill(FakeScorer(), ideas, f"has {_MARK}", _cfg().prompts_dir(), "bilingual")
    assert out == {4: 1, 3: 2, 2: 3, 1: 4}                 # reversed order under the marker


def test_vocab_overlap_flags_restatement_not_orthogonal():
    # A restatement of words already in the skill -> high overlap (likely reweight).
    assert vocab_overlap("rewards explicit isomorphism structure", "rewards explicit isomorphism") > 0.9
    # Orthogonal vocabulary -> low overlap (plausibly a genuinely new axis).
    assert vocab_overlap("rewards explicit isomorphism", "prefers empirical falsifiability quickly") < 0.2
