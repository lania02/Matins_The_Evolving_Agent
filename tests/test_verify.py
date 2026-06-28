"""Offline tests for the verifier panel (MATINS_UPGRADE_PLAN phases A + B).

Covers the calibrated unique verifier ("searched != novel"), the demand-anchored useful
verifier, run_panel persistence (verdicts + back-compat prior_art), and the run_batch wiring
being opt-in (off by default = legacy novelty path; on = verdicts attached, offline-safe).
"""
from __future__ import annotations

from pathlib import Path

from matins.config import load_config
from matins.generate.pipeline import run_batch
from matins.generate.verify import UniqueVerifier, UsefulVerifier, Verdict, run_panel
from matins.store.db import Store, new_id
from matins.store.models import Batch, Idea

REPO_ROOT = Path(__file__).resolve().parent.parent

_RANKS = ('[{"idx":1,"rank":1,"rationale":"r"},{"idx":2,"rank":2,"rationale":"r"},'
          '{"idx":3,"rank":3,"rationale":"r"},{"idx":4,"rank":4,"rationale":"r"}]')


class FakeSearch:
    def __init__(self, results):
        self.results = results
        self.queries = []

    def search(self, q, *, k=5):
        self.queries.append(q)
        return self.results


def _idea(prior_art="[unchecked]"):
    return Idea(idea_id=new_id(), batch_id="b", slot="highfit", idx=1,
                title="Spectral radius market stability (spectral radius)",
                math_structure="spectral radius (operator)",
                mechanism="a market contagion operator",
                prior_art=prior_art)


# ---- unit: unique verifier (the calibration crux) ----------------------------
def test_unique_verifier_calibrates_absence_as_low_confidence():
    # close neighbor found -> uniqueness uncertain, but we have something to compare (mid conf)
    hit = UniqueVerifier(FakeSearch([{"title": "A related operator paper", "url": "http://x"}]))
    v = hit.assess(_idea(), k=5)
    assert v.axis == "unique" and v.evidence and 0 < v.confidence < 1

    # NOTHING found -> must NOT be a confident "novel": low confidence + honest note
    empty = UniqueVerifier(FakeSearch([]))
    v2 = empty.assess(_idea(), k=5)
    assert v2.confidence <= 0.4
    assert "not confirmed novel" in v2.note.lower()

    # offline -> unverified (zero confidence, no crash)
    assert UniqueVerifier(None).assess(_idea(), k=5).confidence == 0.0


def test_unique_verifier_reuses_saturation_prior_art_without_searching():
    s = FakeSearch([{"title": "should not be used", "url": "http://x"}])
    v = UniqueVerifier(s).assess(_idea(prior_art="closest prior art: Foo -- http://foo"), k=5)
    assert s.queries == []                      # did not search again
    assert v.evidence and v.confidence > 0


# ---- unit: useful verifier (demand anchor) -----------------------------------
def test_useful_verifier_scores_observed_demand():
    found = UsefulVerifier(FakeSearch([{"title": "Ask HN: I wish a tool existed for X", "url": "http://hn"}]))
    v = found.assess(_idea(), k=5)
    assert v.axis == "useful" and v.score > 0.5 and v.evidence

    none = UsefulVerifier(FakeSearch([]))
    assert none.assess(_idea(), k=5).score < 0.5            # no observed pull
    assert UsefulVerifier(None).assess(_idea(), k=5).confidence == 0.0


# ---- run_panel persistence ---------------------------------------------------
def test_run_panel_attaches_verdicts_and_fills_prior_art():
    cfg = load_config(str(REPO_ROOT / "config.example.yaml"))
    cfg.verify.axes = ["unique", "useful"]

    store = Store(":memory:")
    store.insert_batch(Batch(batch_id="b1", date="2026-01-01"))
    idea = Idea(idea_id="i1", batch_id="b1", slot="highfit", idx=1,
                title="X (optimal transport)", mechanism="m (operator)")
    store.insert_idea(idea)

    uni = FakeSearch([{"title": "Prior OT work", "url": "http://p"}])
    dem = FakeSearch([{"title": "Ask HN: need this", "url": "http://d"}])
    run_panel([idea], cfg, uni, store, batch_id="b1", demand_search=dem)

    reread = store.ideas_for_batch("b1")[0]
    assert '"unique"' in reread.verdicts and '"useful"' in reread.verdicts
    assert reread.prior_art.startswith("closest prior art")   # unique still fills prior_art


# ---- run_batch wiring: opt-in + offline-safe ---------------------------------
def _idea_json(n):
    bridge = ("finance 与 spectral radius 的结构对应：把后者的算子搬到前者对象上，"
              "二者共享同一不动点结构，可迁移其收敛性定理，最小验证可在玩具网络上扫一遍。")
    return ('{"title": "Idea %d", "mechanism": "m", "why_now": "w", "math_structure": "", '
            '"tractability": "t", "fit_to_program": "f", "behavior": "domain%d . method%d", '
            '"bridge": "%s"}' % (n, n, n, bridge))


class _LLM:
    def __init__(self):
        self.n = 0

    def generate(self, prompt, *, temperature, json_schema=None):
        if "single JSON object" in prompt or "RANDOM-MUTATION" in prompt:
            self.n += 1
            return _idea_json(self.n)
        return _RANKS


def test_run_batch_verify_off_by_default_uses_legacy_path():
    cfg = load_config(str(REPO_ROOT / "config.example.yaml"))   # verify.axes == [] in the example
    assert cfg.verify.axes == []
    _b, ideas = run_batch(cfg, Store(":memory:"), _LLM(), None, date="2026-06-01")
    assert all(i.verdicts == "" for i in ideas)                 # no panel -> no verdicts


def test_run_batch_verify_on_attaches_verdicts_offline():
    cfg = load_config(str(REPO_ROOT / "config.example.yaml"))
    cfg.verify.axes = ["unique", "useful"]
    # search=None -> offline: both axes degrade to unverified verdicts, NO network call
    _b, ideas = run_batch(cfg, Store(":memory:"), _LLM(), None, date="2026-06-02")
    assert ideas and all('"unique"' in i.verdicts and '"useful"' in i.verdicts for i in ideas)
