"""Offline tests for the grounding lens (Idea Oracle port).

Covers the lens module (load/filter/sample/render) and its integration into run_batch:
the configured slots get a distinct real vantage, it is recorded on the idea and rendered
into the prompt, and random is never lensed.
"""
from __future__ import annotations

from pathlib import Path

from matins.config import load_config
from matins.generate.lens import (
    Lens,
    lenses_for_mode,
    load_lenses,
    render_lens_block,
    sample_lenses,
)
from matins.generate.pipeline import run_batch
from matins.store.db import Store

REPO_ROOT = Path(__file__).resolve().parent.parent

_RANKS = ('[{"idx":1,"rank":1,"rationale":"r"},{"idx":2,"rank":2,"rationale":"r"},'
          '{"idx":3,"rank":3,"rationale":"r"},{"idx":4,"rank":4,"rationale":"r"}]')


def _idea(title, behavior):
    # carry a substantive bridge so the depth gate does not interfere with lens tests
    dom, _, meth = behavior.partition(".")
    bridge = (f"{dom.strip()} 与 {meth.strip()} 的结构对应：把后者的算子搬到前者的对象上，"
              f"二者共享同一不动点结构，这一映射并不显然，却能迁移其收敛性定理。")
    return ('{"title": "%s", "mechanism": "m", "why_now": "w", "math_structure": "", '
            '"tractability": "t", "fit_to_program": "f", "behavior": "%s", "bridge": "%s"}'
            % (title, behavior, bridge))


# ---- unit: lens module -------------------------------------------------------
def test_load_and_filter_lenses_from_repo_pool():
    pool = load_lenses(REPO_ROOT / "prompts")
    assert pool, "the shipped prompts/lenses.yaml should load some lenses"
    kinds = {l.kind for l in pool}
    assert {"occupation", "research"} <= kinds
    assert all(l.name for l in pool)                       # every lens has a tag

    assert all(l.kind == "research" for l in lenses_for_mode(pool, "research"))
    assert all(l.kind == "occupation" for l in lenses_for_mode(pool, "product"))
    assert {l.kind for l in lenses_for_mode(pool, "mixed")} == {"occupation", "research"}
    assert lenses_for_mode(pool, "off") == []


def test_sample_lenses_is_distinct_and_avoids_recent():
    pool = [Lens("research", f"L{i}", {}) for i in range(5)]
    picks = sample_lenses(pool, 3, exclude_names={"L0", "L1"})
    assert len(picks) == 3
    assert len({p.name for p in picks}) == 3               # distinct within the draw
    assert {"L0", "L1"}.isdisjoint({p.name for p in picks})  # recent ones avoided

    # pool smaller than k -> still returns k (repeats allowed as a last resort)
    small = [Lens("research", "only", {})]
    assert len(sample_lenses(small, 3)) == 3
    assert sample_lenses([], 3) == []


def test_render_lens_block_shapes():
    occ = Lens("occupation", "Radiologist",
               {"tasks": ["read scans"], "frictions": ["fatigue misses"]})
    block = render_lens_block(occ)
    assert "GROUNDING VANTAGE" in block and "Radiologist" in block
    assert "weekend-buildable" in block                    # occupation -> product framing

    res = Lens("research", "Urban mobility", {"open_questions": ["how does a fare ripple?"]})
    rblock = render_lens_block(res)
    assert "Urban mobility" in rblock and "falsifiable" in rblock

    assert render_lens_block(None) == ""                   # no lens -> empty placeholder


# ---- integration: run_batch with lenses on -----------------------------------
def test_lens_assigned_recorded_rendered_and_skips_random():
    cfg = load_config(str(REPO_ROOT / "config.example.yaml"))
    cfg.generation.lens_mode = "mixed"                     # default lens_slots = [adjacent, orthogonal]

    class RecordingLLM:
        def __init__(self):
            self.prompts: dict[str, str] = {}
            self.n = 0

        def generate(self, prompt, *, temperature, json_schema=None):
            if "single JSON object" in prompt or "RANDOM-MUTATION" in prompt:
                for key in ("HIGH-FIT", "ADJACENT-STRETCH", "ORTHOGONAL", "RANDOM-MUTATION"):
                    if key in prompt:
                        self.prompts[key] = prompt
                self.n += 1
                return _idea(f"Idea {self.n}", f"domain{self.n} . method{self.n}")
            return _RANKS

    store = Store(":memory:")
    llm = RecordingLLM()
    _b, ideas = run_batch(cfg, store, llm, None, date="2026-05-01")
    by_slot = {i.slot: i for i in ideas}

    assert by_slot["adjacent"].lens != ""                  # adjacent got a real vantage
    assert by_slot["orthogonal"].lens != ""                # orthogonal too
    assert by_slot["adjacent"].lens != by_slot["orthogonal"].lens   # distinct vantages
    assert by_slot["highfit"].lens == ""                   # not in default lens_slots
    assert by_slot["random"].lens == ""                    # random is never lensed

    assert "GROUNDING VANTAGE" in llm.prompts["ADJACENT-STRETCH"]
    assert "GROUNDING VANTAGE" in llm.prompts["ORTHOGONAL"]
    assert "GROUNDING VANTAGE" not in llm.prompts["RANDOM-MUTATION"]

    reread = {i.slot: i for i in store.ideas_for_batch(ideas[0].batch_id)}
    assert reread["orthogonal"].lens == by_slot["orthogonal"].lens   # round-trips through SQLite


def test_lens_off_by_default_leaves_generation_unlensed():
    cfg = load_config(str(REPO_ROOT / "config.example.yaml"))   # lens_mode: off in the example
    assert cfg.generation.lens_mode == "off"

    class LLM:
        def __init__(self):
            self.n = 0

        def generate(self, prompt, *, temperature, json_schema=None):
            if "single JSON object" in prompt or "RANDOM-MUTATION" in prompt:
                self.n += 1
                return _idea(f"Idea {self.n}", f"domain{self.n} . method{self.n}")
            return _RANKS

    _b, ideas = run_batch(cfg, Store(":memory:"), LLM(), None, date="2026-05-02")
    assert all(i.lens == "" for i in ideas)               # nothing lensed when off
