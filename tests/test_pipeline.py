"""End-to-end integration of the generation pipeline, fully offline.

Exercises the real spine (run_batch) + the real leaf modules (slots/schema/
novelty/digest) wired together, using a FakeLLM and no search/messaging. This is
the one test that proves the whole daily-run path integrates -- the unit tests
deliberately avoid run_batch because it needs an LLM.
"""
from __future__ import annotations

from pathlib import Path

from matins.config import load_config
from matins.digest.render import render_digest
from matins.generate.pipeline import run_batch
from matins.store.db import Store
from matins.store.models import SLOTS

REPO_ROOT = Path(__file__).resolve().parent.parent


class FakeLLM:
    """Returns a valid idea object for generation calls, a rank list for self-rank."""

    def __init__(self) -> None:
        self.n = 0

    def generate(self, prompt: str, *, temperature: float, json_schema=None) -> str:
        if "single JSON object" in prompt:  # from the IDEA_SCHEMA instruction
            self.n += 1
            return (
                '{"title": "Idea %d", "mechanism": "m", "why_now": "w", '
                '"math_structure": "", "tractability": "t", "fit_to_program": "f"}'
            ) % self.n
        # self-rank prompt -> JSON list of {idx, rank, rationale}
        return (
            '[{"idx":1,"rank":1,"rationale":"r1"},{"idx":2,"rank":2,"rationale":"r2"},'
            '{"idx":3,"rank":3,"rationale":"r3"},{"idx":4,"rank":4,"rationale":"r4"}]'
        )


def _cfg():
    # root resolves to the repo, so prompts_dir / skills_dir / interest_seed exist.
    return load_config(str(REPO_ROOT / "config.example.yaml"))


def test_run_batch_full_pipeline():
    cfg = _cfg()
    store = Store(":memory:")
    llm = FakeLLM()

    batch, ideas = run_batch(cfg, store, llm, None, date="2026-01-01")

    assert len(ideas) == 4
    assert [i.slot for i in ideas] == SLOTS
    assert [i.idx for i in ideas] == [1, 2, 3, 4]
    # self-rank applied
    assert all(i.self_rank is not None for i in ideas)
    # novelty with search=None flags every idea unchecked
    assert all(i.prior_art == "[unchecked]" for i in ideas)
    # random slot recorded its sampled genes
    rnd = next(i for i in ideas if i.slot == "random")
    assert rnd.random_genes != ""

    # digest renders one message per idea, header carries the date
    header, msgs = render_digest(batch, ideas, cfg.generation.output_language)
    assert "2026-01-01" in header
    assert len(msgs) == 4
    assert all(len(m) < 4096 for m in msgs)


def test_interleave_balances_and_caps():
    from matins.generate.pipeline import _interleave
    a = [{"via": "arxiv"}, {"via": "arxiv"}]
    b = [{"via": "web"}]
    c = [{"via": "hn"}, {"via": "hn"}, {"via": "hn"}]
    out = _interleave([a, b, c], cap=4)
    assert len(out) == 4                                         # cap honored
    assert [o["via"] for o in out] == ["arxiv", "web", "hn", "arxiv"]   # round-robin


def test_collect_channel_dedup_quota_and_tag():
    from matins.generate.pipeline import _collect_channel

    class FakeP:
        def __init__(self, per_query):
            self.per_query = per_query

        def search(self, q, *, k=5):
            return self.per_query.get(q, [])

    seen = {"http://dup"}                                        # already seen -> skipped
    provider = FakeP({
        "q1": [{"url": "http://dup"}, {"url": "http://a"}],      # dup skipped, takes 'a'
        "q2": [{"url": "http://b"}],
        "q3": [{"url": "http://c"}],
    })
    out = _collect_channel(provider, ["q1", "q2", "q3"], quota=2, seen=seen, label="arxiv")
    assert [o["url"] for o in out] == ["http://a", "http://b"]  # one per query, quota stops at 2
    assert all(o["via"] == "arxiv" for o in out)


def test_run_batch_is_idempotent_per_date():
    cfg = _cfg()
    store = Store(":memory:")
    llm = FakeLLM()

    b1, ideas1 = run_batch(cfg, store, llm, None, date="2026-01-02")
    b2, ideas2 = run_batch(cfg, store, llm, None, date="2026-01-02")

    assert b1.batch_id == b2.batch_id
    assert len(ideas2) == 4


_VALID = ('{"title": "T", "mechanism": "m", "why_now": "w", "math_structure": "", '
          '"tractability": "t", "fit_to_program": "f"}')
_RANKS = ('[{"idx":1,"rank":1,"rationale":"r"},{"idx":2,"rank":2,"rationale":"r"},'
          '{"idx":3,"rank":3,"rationale":"r"},{"idx":4,"rank":4,"rationale":"r"}]')


def test_random_slot_recovers_via_retry():
    # The pure-perturbation slot fails its first attempt (gen + both repairs) but succeeds
    # on a retry -> all 4 slots present, instead of an over-fit 3-idea batch.
    class FlakyRandomLLM:
        def __init__(self):
            self.random_gens = 0

        def generate(self, prompt, *, temperature, json_schema=None):
            if "definitely not json" in prompt:          # repairs of the failed random gen
                return "definitely not json"
            if "RANDOM-MUTATION" in prompt:              # the random slot's generation prompt
                self.random_gens += 1
                return "definitely not json" if self.random_gens < 2 else _VALID
            if "single JSON object" in prompt:           # other slots' generation
                return _VALID
            return _RANKS                                # self-rank

    _batch, ideas = run_batch(_cfg(), Store(":memory:"), FlakyRandomLLM(), None, date="2026-02-10")
    assert [i.slot for i in ideas] == SLOTS              # random recovered via retry
    assert len(ideas) == 4


def test_siblings_injected_for_within_batch_distinctness():
    # Each slot is shown the ideas already produced THIS batch, so adjacent cannot collapse
    # into a near-duplicate of high-fit (the slot prompts enforce "distinct from above").
    class RecordingLLM:
        def __init__(self):
            self.prompts = []
            self.n = 0

        def generate(self, prompt, *, temperature, json_schema=None):
            if "single JSON object" in prompt or "RANDOM-MUTATION" in prompt:
                self.prompts.append(prompt)
                self.n += 1
                return (f'{{"title": "TITLE{self.n}", "mechanism": "m", "why_now": "w", '
                        '"math_structure": "", "tractability": "t", "fit_to_program": "f"}')
            return _RANKS

    llm = RecordingLLM()
    run_batch(_cfg(), Store(":memory:"), llm, None, date="2026-02-11")
    adjacent_prompt = next(p for p in llm.prompts if "ADJACENT-STRETCH" in p)
    assert "TITLE1" in adjacent_prompt                   # the high-fit sibling is shown to adjacent


def test_within_batch_duplicate_is_rejected_and_retried():
    # If a slot restates a sibling verbatim (the model ignoring the "distinct" prompt),
    # the deterministic backstop rejects it and retries to something genuinely different.
    def _idea(t):
        return (f'{{"title": "{t}", "mechanism": "m", "why_now": "w", "math_structure": "", '
                '"tractability": "t", "fit_to_program": "f"}')

    class DupAdjacentLLM:
        def __init__(self):
            self.adj = 0

        def generate(self, prompt, *, temperature, json_schema=None):
            if "HIGH-FIT" in prompt:
                return _idea("Spectral radius market stability")
            if "ADJACENT-STRETCH" in prompt:
                self.adj += 1
                return _idea("Spectral radius market stability") if self.adj == 1 \
                    else _idea("Population genetics drift model")
            if "ORTHOGONAL" in prompt:
                return _idea("Orthogonal contrarian probe")
            if "RANDOM-MUTATION" in prompt:
                return _idea("Random mutation perturbation")
            return _RANKS

    _b, ideas = run_batch(_cfg(), Store(":memory:"), DupAdjacentLLM(), None, date="2026-02-12")
    adjacent = next(i for i in ideas if i.slot == "adjacent")
    assert "Population genetics" in adjacent.title       # the verbatim duplicate was rejected
    assert len(ideas) == 4


def test_transient_api_error_skips_slot_not_whole_batch():
    # A 429-style provider error on one slot must degrade to a smaller batch, not crash.
    def _idea(t):
        return (f'{{"title": "{t}", "mechanism": "m", "why_now": "w", "math_structure": "", '
                '"tractability": "t", "fit_to_program": "f"}')

    class OrthoErrorsLLM:
        def generate(self, prompt, *, temperature, json_schema=None):
            if "ORTHOGONAL" in prompt:
                raise RuntimeError("OpenAI-compatible API error 429: Too Many Requests")
            if "HIGH-FIT" in prompt:
                return _idea("Spectral market stability")
            if "ADJACENT-STRETCH" in prompt:
                return _idea("Population genetics drift")
            if "RANDOM-MUTATION" in prompt:
                return _idea("Random matrix chaos")
            return _RANKS

    _b, ideas = run_batch(_cfg(), Store(":memory:"), OrthoErrorsLLM(), None, date="2026-02-13")
    slots = [i.slot for i in ideas]
    assert "orthogonal" not in slots                     # the 429 slot was skipped, not fatal
    assert len(ideas) == 3                               # the other three still made a batch


def _idea_json(t):
    return (f'{{"title": "{t}", "mechanism": "m", "why_now": "w", "math_structure": "", '
            '"tractability": "t", "fit_to_program": "f"}')


def _idea_b(title, behavior, bridge=None):
    """An idea JSON carrying an explicit 'domain . method' behavior tag (A2 dedup key).

    A substantive `bridge` (naming both poles of the cell) is supplied by default so these
    behavior-dedup tests are not incidentally caught by the bridge-depth gate (B); pass
    bridge="" to exercise that gate explicitly.
    """
    if bridge is None:
        dom, _, meth = behavior.partition(".")
        bridge = (f"{dom.strip()} 与 {meth.strip()} 的结构对应：把后者的算子搬到前者的对象上，"
                  f"二者共享同一不动点结构，这一映射并不显然，却能迁移其收敛性定理。")
    return ('{"title": "%s", "mechanism": "m", "why_now": "w", "math_structure": "", '
            '"tractability": "t", "fit_to_program": "f", "behavior": "%s", "bridge": "%s"}'
            % (title, behavior, bridge))


def test_behavior_cell_and_conflict_helpers():
    # A2 unit: the 'domain . method' tag parses, and the per-slot conflict rule is correct.
    from matins.generate.pipeline import _behavior_cell, _behavior_conflict

    assert _behavior_cell("Causal Inference . Optimal Transport") == (
        "causal inference", "optimal transport")
    assert _behavior_cell("") == ("", "")
    assert _behavior_cell("solo-domain") == ("solo-domain", "")

    sibs = [{"behavior": "finance . spectral radius"}]
    # orthogonal: a shared DOMAIN is a conflict no matter the method...
    assert _behavior_conflict("orthogonal", "finance . auction theory", sibs) is True
    # ...but a fresh domain passes even when the method coincides.
    assert _behavior_conflict("orthogonal", "biology . spectral radius", sibs) is False
    # adjacent: only the FULL cell (same domain AND method) collapses; new method is fine.
    assert _behavior_conflict("adjacent", "finance . spectral radius", sibs) is True
    assert _behavior_conflict("adjacent", "finance . optimal transport", sibs) is False
    # an empty tag never matches -> the title backstop takes over.
    assert _behavior_conflict("orthogonal", "", sibs) is False


def test_orthogonal_rejects_shared_domain_even_with_a_new_title():
    # A2: the orthogonal slot's first try lands in a domain already used today (finance), but
    # with a title that does NOT overlap any sibling's -- so the title backstop would miss it.
    # The behavior-cell check must still reject it and retry into a genuinely new domain.
    class OrthoSameDomainLLM:
        def __init__(self):
            self.ortho = 0

        def generate(self, prompt, *, temperature, json_schema=None):
            if "HIGH-FIT" in prompt:
                return _idea_b("Spectral radius market stability", "finance . spectral radius")
            if "ADJACENT-STRETCH" in prompt:
                return _idea_b("Population genetics drift", "biology . genetic drift")
            if "ORTHOGONAL" in prompt:
                self.ortho += 1
                return _idea_b("Auction design for spectrum allocation",
                               "finance . auction theory") if self.ortho == 1 \
                    else _idea_b("Topology of language change", "linguistics . algebraic topology")
            if "RANDOM-MUTATION" in prompt:
                return _idea_b("Random matrix chaos", "physics . random matrix")
            return _RANKS

    _b, ideas = run_batch(_cfg(), Store(":memory:"), OrthoSameDomainLLM(), None,
                          date="2026-04-01")
    ortho = next(i for i in ideas if i.slot == "orthogonal")
    assert "Topology" in ortho.title            # the same-domain (finance) first try was rejected
    assert "linguistics" in ortho.behavior
    assert len(ideas) == 4


def test_adjacent_rejects_full_cell_collision_but_keeps_a_new_method():
    # A2: adjacent's first try copies high-fit's WHOLE cell (same domain AND method) under a
    # different title -> rejected; the retry keeps the domain but changes the method -> kept.
    # Proves the slot-graded rule: a near-core extension may stay in-domain if it shifts method.
    class AdjFullCellLLM:
        def __init__(self):
            self.adj = 0

        def generate(self, prompt, *, temperature, json_schema=None):
            if "HIGH-FIT" in prompt:
                return _idea_b("Spectral radius market stability", "finance . spectral radius")
            if "ADJACENT-STRETCH" in prompt:
                self.adj += 1
                return _idea_b("Spectral radius for credit networks",
                               "finance . spectral radius") if self.adj == 1 \
                    else _idea_b("Optimal transport for market contagion",
                                 "finance . optimal transport")
            if "ORTHOGONAL" in prompt:
                return _idea_b("Topology of language change", "linguistics . topology")
            if "RANDOM-MUTATION" in prompt:
                return _idea_b("Random matrix chaos", "physics . random matrix")
            return _RANKS

    llm = AdjFullCellLLM()
    _b, ideas = run_batch(_cfg(), Store(":memory:"), llm, None, date="2026-04-02")
    adjacent = next(i for i in ideas if i.slot == "adjacent")
    assert "Optimal transport" in adjacent.title    # full-cell collision rejected, new method kept
    assert adjacent.behavior == "finance . optimal transport"  # same domain retained on purpose
    assert llm.adj == 2                              # exactly one rejection + one accept
    assert len(ideas) == 4


def test_occupied_cells_injected_into_later_slot_prompts():
    # B2: each later slot is told the 'domain . method' cells already taken THIS batch, as a
    # hard prohibition -- the prompt-side companion to the code-level behavior dedup.
    class RecordingLLM:
        def __init__(self):
            self.prompts = []
            self.n = 0

        def generate(self, prompt, *, temperature, json_schema=None):
            if "single JSON object" in prompt or "RANDOM-MUTATION" in prompt:
                self.prompts.append(prompt)
                self.n += 1
                return _idea_b(f"TITLE{self.n}", f"domain{self.n} . method{self.n}")
            return _RANKS

    llm = RecordingLLM()
    run_batch(_cfg(), Store(":memory:"), llm, None, date="2026-04-03")
    adj_prompt = next(p for p in llm.prompts if "ADJACENT-STRETCH" in p)
    ortho_prompt = next(p for p in llm.prompts if "ORTHOGONAL" in p)
    assert "ALREADY TAKEN TODAY" in adj_prompt
    assert "domain1 . method1" in adj_prompt         # high-fit's cell shown to adjacent
    assert "domain1 . method1" in ortho_prompt       # ...and to orthogonal
    assert "domain2 . method2" in ortho_prompt       # along with adjacent's cell


def test_bridge_too_shallow_helper():
    # B unit: the free depth backstop -- long enough AND names both poles, else shallow.
    from matins.generate.pipeline import _bridge_too_shallow

    deep = ("finance 与 spectral radius 的结构对应：把后者的算子搬到前者的对象上，"
            "二者共享同一不动点结构，这一映射并不显然，却能迁移其收敛性定理。")
    assert _bridge_too_shallow(deep, "finance . spectral radius") is False
    # too short -> shallow regardless of content
    assert _bridge_too_shallow("把 X 套到 Y 上。", "finance . spectral radius") is True
    # long but names only ONE pole (no 'spectral radius' tokens) -> shallow
    assert _bridge_too_shallow("finance " * 20, "finance . spectral radius") is True
    # missing behavior tag -> degrade to a length-only check (poles vacuously named)
    assert _bridge_too_shallow(deep, "") is False


def test_bridge_persists_through_pipeline_and_store():
    # A: every idea now carries a 'bridge', and it round-trips through SQLite intact.
    class BridgeLLM:
        def __init__(self):
            self.n = 0

        def generate(self, prompt, *, temperature, json_schema=None):
            if "single JSON object" in prompt or "RANDOM-MUTATION" in prompt:
                self.n += 1
                return _idea_b(f"Idea {self.n}", f"domain{self.n} . method{self.n}")
            return _RANKS

    store = Store(":memory:")
    _b, ideas = run_batch(_cfg(), store, BridgeLLM(), None, date="2026-04-11")
    assert all(i.bridge for i in ideas)                  # every idea carries a bridge
    assert "结构对应" in ideas[0].bridge
    reread = store.ideas_for_batch(ideas[0].batch_id)
    assert reread[0].bridge == ideas[0].bridge           # survives the DB round-trip


def test_bridge_gate_regenerates_a_shallow_bridge():
    # B: a fusion slot whose first try has a stub (empty) bridge is rejected and retried
    # into one that actually articulates the collision -- the depth analogue of dedup.
    class ShallowThenDeepLLM:
        def __init__(self):
            self.ortho = 0

        def generate(self, prompt, *, temperature, json_schema=None):
            if "HIGH-FIT" in prompt:
                return _idea_b("Spectral markets", "finance . spectral radius")
            if "ADJACENT-STRETCH" in prompt:
                return _idea_b("Drift genetics", "biology . genetic drift")
            if "ORTHOGONAL" in prompt:
                self.ortho += 1
                return _idea_b("Linguistic topology", "linguistics . topology",
                               "" if self.ortho == 1 else None)   # stub first, deep on retry
            if "RANDOM-MUTATION" in prompt:
                return _idea_b("Random chaos", "physics . random matrix")
            return _RANKS

    llm = ShallowThenDeepLLM()
    _b, ideas = run_batch(_cfg(), Store(":memory:"), llm, None, date="2026-04-12")
    ortho = next(i for i in ideas if i.slot == "orthogonal")
    assert llm.ortho == 2                                 # the empty-bridge first try was rejected
    assert "结构对应" in ortho.bridge                     # the substantive bridge was kept
    assert len(ideas) == 4


def test_saturation_gate_regenerates_red_ocean_orthogonal():
    # The orthogonal slot's first try is a saturated "red ocean" idea. The grounded judge
    # (given a high literature count + closest works = B2) returns "regenerate" (B1), so the
    # slot is retried into a fresh idea. Proves B2 (density) + B1 (judge) wire into the loop.
    class GatedLLM:
        def __init__(self):
            self.judge_calls = 0
            self.ortho_gens = 0

        def generate(self, prompt, *, temperature, json_schema=None):
            if "GROUNDED EVIDENCE" in prompt:                  # the saturation judge (B1)
                self.judge_calls += 1
                verdict = "regenerate" if self.judge_calls == 1 else "keep"
                return '{"verdict": "%s", "saturation": "high", "reason": "r"}' % verdict
            if "ORTHOGONAL" in prompt:                         # checked AFTER the judge marker
                self.ortho_gens += 1
                # Carry a substantive bridge so the depth gate passes and the saturation
                # judge (the behaviour under test) is the one that decides regenerate/keep.
                br = ("把远域的算子搬到本域对象上，二者共享同一不动点结构，这一对应并不显然，"
                      "却可迁移其收敛性定理，正是本槽要的碰撞点。")
                return _idea_b("Red ocean GNN protein" if self.ortho_gens == 1
                               else "Fresh contrarian angle", "", br)
            if "HIGH-FIT" in prompt:
                return _idea_json("Spectral market stability")
            if "ADJACENT-STRETCH" in prompt:
                return _idea_json("Population genetics drift")
            if "RANDOM-MUTATION" in prompt:
                return _idea_json("Random matrix chaos")
            return _RANKS                                      # self-rank

    class CountingSearch:
        def __init__(self):
            self.counts = 0

        def search(self, query, *, k=5):
            return [{"title": "Existing flagship work", "url": "http://x"}]

        def count(self, query):                                # B2 density signal
            self.counts += 1
            return 99999                                       # a crowded field

    gate_search = CountingSearch()
    llm = GatedLLM()
    # search=None keeps novelty/retrieval offline; the gate runs on the injected gate_search.
    _b, ideas = run_batch(_cfg(), Store(":memory:"), llm, None,
                          date="2026-03-01", gate_search=gate_search)

    ortho = next(i for i in ideas if i.slot == "orthogonal")
    assert "Fresh" in ortho.title                              # red-ocean first try regenerated
    assert llm.judge_calls >= 2                                # judged twice: regenerate -> keep
    assert gate_search.counts >= 1                             # B2 density was actually queried
    assert ortho.prior_art.startswith("closest prior art")     # gate's search filled prior_art
    assert len(ideas) == 4


def test_saturation_gate_inactive_without_search():
    # With no search provider the gate cannot ground itself, so it must stay off: the
    # orthogonal slot is accepted on the first try and no judge call is made.
    class CountingJudge:
        def __init__(self):
            self.judge_calls = 0

        def generate(self, prompt, *, temperature, json_schema=None):
            if "GROUNDED EVIDENCE" in prompt:
                self.judge_calls += 1
                return '{"verdict": "regenerate"}'
            if "single JSON object" in prompt or "RANDOM-MUTATION" in prompt:
                return _VALID
            return _RANKS

    llm = CountingJudge()
    _b, ideas = run_batch(_cfg(), Store(":memory:"), llm, None, date="2026-03-02")
    assert llm.judge_calls == 0                                 # gate never ran (search=None)
    assert len(ideas) == 4


def test_saturation_gate_slots_empty_disables_gate():
    # The gated-slot set is configurable; an empty list turns the gate off entirely, even
    # with a search wired -- the orthogonal candidate is accepted on the first try.
    class CountingJudge:
        def __init__(self):
            self.judge_calls = 0

        def generate(self, prompt, *, temperature, json_schema=None):
            if "GROUNDED EVIDENCE" in prompt:
                self.judge_calls += 1
                return '{"verdict": "regenerate"}'
            if "single JSON object" in prompt or "RANDOM-MUTATION" in prompt:
                return _VALID
            return _RANKS

    class CountingSearch:
        def search(self, query, *, k=5):
            return [{"title": "w", "url": "http://x"}]

        def count(self, query):
            return 99999

    cfg = _cfg()
    cfg.novelty.saturation_gate_slots = []                      # gate off by config
    llm = CountingJudge()
    _b, ideas = run_batch(cfg, Store(":memory:"), llm, None,
                          date="2026-03-03", gate_search=CountingSearch())
    assert llm.judge_calls == 0                                 # no slot gated -> judge never ran
    assert len(ideas) == 4
