"""Generation pipeline: the heart of `matins run` (DESIGN.md section 3, 6, 7).

Flow per batch: assemble context (taste skill + fast memory + fresh retrieval +
interest seed) -> generate one idea per slot (tolerant-parse, one repair retry) ->
self-rank the four -> novelty check each -> persist. Idempotent on date: a second
run for the same date returns the existing batch unchanged.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

from ..config import Config
from ..providers.base import LLMProvider, SearchProvider
from ..store.db import Store, new_id, now_iso, today_iso
from ..store.models import SLOTS, Batch, Idea
from .explore import adaptive_temperature
from .saturation import gate_saturation
from .schema import IDEA_JSON_SCHEMA, IdeaParseError, parse_idea
from .slots import (
    build_generation_prompt,
    build_self_rank_prompt,
    load_genes,
    parse_self_ranks,
    sample_genes,
    slot_temperature,
)

_REPAIR_INSTRUCTION = (
    "Your previous reply was not valid JSON. Re-emit the SAME idea as a single "
    "JSON object only, no prose, no code fences. Keys: title, mechanism, why_now, "
    "math_structure, tractability, fit_to_program.\n\nPrevious reply:\n"
)

# How many times to (re)try a slot before giving up on it. The random slot resamples its
# genes each attempt -- a single awkward gene combo is the usual reason that pure-
# perturbation slot fails to yield JSON, and dropping it leaves <n_slots ideas that
# over-fit the batch to the user's existing work.
_SLOT_ATTEMPTS = 3


def _title_tokens(title: str) -> set:
    """Tokens of a title for crude similarity: CJK characters + English words (len>2)."""
    s = (title or "").lower()
    return set(re.findall(r"[一-鿿]", s)) | {w for w in re.findall(r"[a-z0-9]+", s) if len(w) > 2}


def _too_similar(title: str, siblings: list, *, threshold: float = 0.8) -> bool:
    """True if `title` overlaps any sibling's title at/above `threshold`.

    The prompt-level "distinct from above" constraint is not always honored (a weaker
    model can restate high-fit verbatim in the adjacent slot), so this is a deterministic
    backstop: a slot that merely echoes an idea already produced this batch is retried.
    """
    t = _title_tokens(title)
    if not t:
        return False
    for s in siblings:
        o = _title_tokens(s.get("title", ""))
        if o and len(t & o) / max(len(t), len(o)) >= threshold:
            return True
    return False


def _behavior_cell(behavior: str) -> tuple[str, str]:
    """Parse a '<domain> . <method>' behavior tag into normalized (domain, method).

    Every idea already emits this 2-4 word tag for the diversity archive (slots.py), so it
    is a free, slightly-semantic dedup key -- cheaper and less brittle than title text.
    Returns ("", "") when there's nothing usable, so the caller falls back to the title
    backstop instead of treating an absent tag as a match.
    """
    b = (behavior or "").strip().lower()
    if not b:
        return ("", "")
    parts = re.split(r"\s*[.·]\s*", b, maxsplit=1)      # the schema's ' . ' (or '·') separator
    domain = parts[0].strip()
    method = parts[1].strip() if len(parts) > 1 else ""
    return (domain, method)


def _component_match(a: str, b: str, *, threshold: float = 0.6) -> bool:
    """True if two behavior-cell components (a domain, or a method) are the same 'cell'.

    Token overlap (CJK chars + English words >2) so minor rephrasings of the SAME topic
    still collide ('protein folding' vs 'protein structure folding'); it deliberately does
    NOT catch synonyms in different words ('protein science' vs 'structural biology') -- that
    is semantic dedup, the job of the deferred embedding upgrade (A3), not this cheap gate.
    """
    ta, tb = _title_tokens(a), _title_tokens(b)
    if not ta or not tb:                                # tokenless (e.g. very short) -> exact
        return bool(a.strip()) and a.strip() == b.strip()
    return len(ta & tb) / max(len(ta), len(tb)) >= threshold


# Slots whose whole value is the fusion of two poles -- they must articulate the bridge.
# High-fit is exploit (aim at the core), so it is not depth-gated.
_FUSION_SLOTS = ("adjacent", "orthogonal", "random")


def _bridge_too_shallow(bridge: str, behavior: str, *, min_chars: int = 50) -> bool:
    """True if the `bridge` field is a stub: too short, or it fails to NAME both poles of
    the collision (the domain AND the method of the idea's own behavior cell).

    The free, deterministic depth backstop (A+B), the analogue of `_too_similar`: it cannot
    judge how *deep* the structural correspondence is -- that is the deferred LLM critic
    (tier C) -- only that the model wrote a two-sided connection instead of a one-line
    restatement. Degrades to a length-only check when the behavior tag is missing.
    """
    b = (bridge or "").strip()
    if len(b) < min_chars:
        return True
    dom, meth = _behavior_cell(behavior)
    btoks = _title_tokens(b)

    def names(component: str) -> bool:
        ctoks = _title_tokens(component)
        return (not ctoks) or bool(ctoks & btoks)   # nothing to require -> satisfied

    return not (names(dom) and names(meth))


def _behavior_conflict(slot: str, behavior: str, siblings: list) -> bool:
    """True if this idea's behavior cell collides with an earlier sibling's, under the
    slot's own distinctness rule (A2: semantic 'placeholder cell' dedup).

    Per-slot strictness mirrors each slot's design intent:
      - adjacent: a near-core extension MAY keep a taken domain, but then it must change the
        METHOD (and vice versa); only a same-domain AND same-method cell -- a collapsed
        restatement of high-fit -- is rejected.
      - orthogonal / random: must land in a fresh DOMAIN, so any shared domain is rejected
        (random simply resamples its genes on the next attempt).
    Returns False on an empty/unparseable tag so the title-overlap backstop still applies.
    """
    dom, meth = _behavior_cell(behavior)
    if not dom:
        return False
    for s in siblings:
        s_dom, s_meth = _behavior_cell(s.get("behavior", ""))
        if not s_dom:
            continue
        if slot == "adjacent":
            if _component_match(dom, s_dom) and _component_match(meth, s_meth):
                return True
        elif _component_match(dom, s_dom):             # orthogonal, random: new domain required
            return True
    return False


def _read_interest_seed(cfg: Config) -> str:
    path = cfg.interest_seed_path()
    if path.exists():
        return path.read_text(encoding="utf-8")
    # The real interest_seed.md is personal and gitignored; a fresh clone ships only
    # the .example template. Fall back to it so generation still has a seed out of the box.
    example = path.with_name(path.stem + ".example" + path.suffix)
    return example.read_text(encoding="utf-8") if example.exists() else ""


def _read_active_skill(cfg: Config, store: Store) -> tuple[str, int | None]:
    """Active approved skill text + version; fall back to the skills/taste.md mirror."""
    sv = store.active_skill()
    if sv is not None:
        return sv.content, sv.version
    mirror = cfg.skills_dir() / "taste.md"
    if mirror.exists():
        return mirror.read_text(encoding="utf-8"), None
    return "", None


def _collect_channel(provider, queries: list[str], quota: int,
                     seen: set[str], label: str) -> list[dict]:
    """One fresh (non-duplicate) hit per query, for topic spread, up to `quota`.

    `seen` is shared across channels (and seeded from recent batches) so the same
    item never appears twice. Each kept item is tagged with its source `label`.
    """
    bucket: list[dict] = []
    for q in queries:
        if len(bucket) >= quota:
            break
        try:
            results = provider.search(q, k=3) or []
        except Exception:
            continue
        for r in results:                         # take the first fresh hit for this query
            rid = (r.get("url") or r.get("title") or "").strip()
            if not rid or rid in seen:
                continue
            seen.add(rid)
            bucket.append({**r, "via": label})
            break
    return bucket


def _interleave(buckets: list[list[dict]], cap: int) -> list[dict]:
    """Round-robin merge channel buckets so the cap stays balanced, not front-loaded."""
    out: list[dict] = []
    depth = 0
    while len(out) < cap and any(depth < len(b) for b in buckets):
        for b in buckets:
            if depth < len(b):
                out.append(b[depth])
                if len(out) >= cap:
                    break
        depth += 1
    return out


def _fetch_retrieval(cfg: Config, store: Store, batch_id: str) -> list[dict]:
    """Blend a small, balanced set of fresh items across the configured sources.

    Each source (cfg.retrieval.blend) contributes at most its quota; the channels are
    interleaved and the whole feed is capped at cfg.retrieval.max_items, de-duped
    against recent batches. A deliberate blend, not a pile: scholarly sources lead,
    web/community add a minority of timeliness/breakout signal.

    Advisory only: any source failure is swallowed; returns [] when nothing is
    configured or available.
    """
    from ..providers.search_web import get_retrieval_searcher

    sources = cfg.retrieval.sources
    blend = cfg.retrieval.blend or {}
    if not sources or not blend:
        return []

    seen = store.recent_result_ids(cfg.retrieval.dedup_against_days)
    buckets: list[list[dict]] = []
    for name, quota in blend.items():
        if quota <= 0:
            continue
        provider = get_retrieval_searcher(name, cfg)
        if provider is None:                      # missing key / unknown source -> skip
            continue
        bucket = _collect_channel(provider, sources, quota, seen, name)
        if bucket:
            buckets.append(bucket)

    out = _interleave(buckets, cfg.retrieval.max_items)
    if out:
        store.log_retrieval(
            batch_id,
            query="; ".join(sources),
            source="blend:" + ",".join(blend.keys()),
            result_ids=[(s.get("url") or s.get("title") or "") for s in out],
        )
    return out


def _generate_one(llm: LLMProvider, prompt: str, temperature: float,
                  *, max_repairs: int = 2) -> dict:
    """Generate + tolerant-parse one idea, with up to `max_repairs` repair retries.

    Flash-tier models occasionally emit prose instead of JSON; each repair re-asks
    for JSON only. Raises IdeaParseError if still unrecoverable after the retries.
    """
    text = llm.generate(prompt, temperature=temperature, json_schema=IDEA_JSON_SCHEMA)
    for _ in range(max_repairs):
        try:
            return parse_idea(text)
        except IdeaParseError:
            text = llm.generate(_REPAIR_INSTRUCTION + text, temperature=0.0)
    return parse_idea(text)  # final attempt; raises IdeaParseError if still unrecoverable


def _apply_self_ranks(llm: LLMProvider, cfg: Config, ideas: list[Idea],
                      prompts_dir: Path) -> None:
    prompt = build_self_rank_prompt(ideas, prompts_dir, cfg.generation.output_language)
    try:
        text = llm.generate(prompt, temperature=0.0)
        ranks = parse_self_ranks(text, len(ideas))
    except Exception:
        ranks = []
    by_idx = {r["idx"]: r for r in ranks}
    for idea in ideas:
        r = by_idx.get(idea.idx)
        if r:
            idea.self_rank = r["rank"]
            idea.self_rationale = r["rationale"]


def run_batch(
    cfg: Config,
    store: Store,
    llm: LLMProvider,
    search: SearchProvider | None,
    *,
    date: str | None = None,
    gate_search: SearchProvider | None = None,
) -> tuple[Batch, list[Idea]]:
    """Run one daily batch. Idempotent per date.

    `gate_search` is the searcher the anti-red-ocean saturation gate uses to ground its
    judgment in literature density. It is deliberately OpenAlex (corpus-wide, citation-
    aware -- a reliable 'how crowded is this field' meter) rather than the novelty
    `search` provider, because arXiv's preprint counts time out on busy queries exactly
    when the gate needs them. Built here when online (search is not None) unless injected
    (offline tests inject a fake so the gate runs without touching the network).
    """
    date = date or today_iso()
    existing = store.batch_for_date(date)
    if existing is not None:
        return existing, store.ideas_for_batch(existing.batch_id)

    prompts_dir = cfg.prompts_dir()
    skill_text, skill_version = _read_active_skill(cfg, store)
    interest_seed = _read_interest_seed(cfg)
    genes_pool = load_genes(prompts_dir)

    # Adaptive exploration (algo-update.md #4): scale the explore temperature to how
    # volatile recent self-vs-user agreement has been -- explore more when uncertain
    # or drifting, exploit when stable. Cold start (no prior taus) -> base unchanged.
    explore_temp = adaptive_temperature(
        [b.self_user_tau for b in store.list_batches(limit=8)],
        cfg.generation.temperature,
    )

    # Fast memory is a *read* of the log (DESIGN.md section 5). Imported lazily so a
    # cold-start run with an empty log never requires the memory module to do work.
    from ..memory.kernels import compute_memory
    fast_kernel = cfg.fast_kernel
    fast_memory = ""
    if fast_kernel is not None:
        try:
            fast_memory = compute_memory(fast_kernel, store, llm, prompts_dir)
        except Exception:
            fast_memory = ""

    batch_id = new_id()
    batch = Batch(
        batch_id=batch_id,
        date=date,
        skill_version=skill_version,
        temperature=explore_temp,
        provider=cfg.provider.name,
        model=cfg.provider.model,
        created_at=now_iso(),
    )

    # Anti-repetition guard: show the model what it already proposed recently so it
    # does not re-surface near-duplicates when the fresh inputs barely move day to day.
    recent_ideas = store.recent_idea_titles(
        cfg.retrieval.dedup_against_days, exclude_batch_id=batch_id
    )

    # QD revival (algo-update.md #5): a few well-liked but dormant directions, offered
    # ONLY to the adjacent-stretch slot so the system can recover good ground it has
    # drifted away from, instead of collapsing onto current favourites.
    revival = store.archive_revival()

    # Fetch the blended fresh-literature feed ONCE; all slots share it (it is the
    # day's "what's new" context, not a per-slot input). Skipped when no search
    # provider is wired (search=None), which also keeps the offline tests network-free.
    retrieval = _fetch_retrieval(cfg, store, batch_id) if search is not None else []

    # Saturation gate searcher (anti red-ocean): OpenAlex, built once for the whole batch.
    # Only when online and at least one slot is gated; an injected gate_search (tests)
    # bypasses construction.
    if gate_search is None and search is not None and cfg.novelty.saturation_gate_slots:
        from ..providers.search_web import OpenAlexSearchProvider
        gate_search = OpenAlexSearchProvider(api_key=cfg.openalex_api_key())

    slots = SLOTS[: cfg.generation.n_slots]
    ideas: list[Idea] = []
    siblings: list[dict] = []          # ideas already produced THIS batch (within-batch distinctness)
    for slot in slots:
        # Per-slot resilience: try a few times before skipping, resampling the random
        # slot's genes each attempt. And show each slot the siblings already produced
        # today so it must differ -- this is what stops the adjacent slot from collapsing
        # into a near-duplicate of high-fit (every conditioned slot's prompt already
        # carries a hard "distinct from everything above" constraint).
        parsed = None
        used_genes = None
        gate_on = gate_search is not None and slot in cfg.novelty.saturation_gate_slots
        for attempt in range(_SLOT_ATTEMPTS):
            last = attempt == _SLOT_ATTEMPTS - 1
            genes_try = sample_genes(genes_pool) if slot == "random" else None
            context = {
                "skill": skill_text,
                "fast_memory": fast_memory,
                "retrieval": retrieval,
                "interest_seed": interest_seed,
                "recent_ideas": recent_ideas + siblings,
                "archive": revival,
                "occupied": siblings,   # B2: 'domain . method' cells already taken THIS batch
            }
            prompt = build_generation_prompt(
                slot, context, prompts_dir, cfg.generation.output_language, genes_try
            )
            try:
                candidate = _generate_one(llm, prompt, slot_temperature(explore_temp, slot))
            except (IdeaParseError, RuntimeError):
                # Bad JSON OR a transient provider/API error (e.g. a 429 rate limit): a
                # single slot must not sink the whole morning. Treat it as a failed attempt.
                continue
            parsed, used_genes = candidate, genes_try        # remember the latest good parse
            # Within-batch distinctness: for non-high-fit slots reject a near-duplicate of a
            # sibling and retry (high-fit SHOULD aim at the core). Keep the last try as fallback.
            # A2 behavior-cell dedup (semantic 'placeholder cell') is the primary check; the
            # title-token overlap stays as a backstop for when the behavior tag is missing.
            if slot != "highfit" and not last and (
                _behavior_conflict(slot, candidate.get("behavior", ""), siblings)
                or _too_similar(candidate.get("title", ""), siblings)
            ):
                continue
            # Bridge depth (A+B): the fusion slots must SHOW why the two colliding poles
            # actually connect (the explicit correspondence the user prizes), not restate the
            # title. Reject a stub bridge and retry; this free check runs before the API-cost
            # saturation gate, and the last attempt is kept (never drop below n_slots).
            if slot in _FUSION_SLOTS and not last and _bridge_too_shallow(
                candidate.get("bridge", ""), candidate.get("behavior", "")
            ):
                continue
            # Anti-red-ocean saturation gate (B2 literature density -> B1 grounded judge):
            # orthogonal only, only with a live search to ground it. Regenerate a saturated,
            # undifferentiated idea; the SAME search fills prior_art so novelty won't re-search
            # it. On the last attempt keep what we have (never drop below n_slots ideas).
            if gate_on:
                passed, prior_note = gate_saturation(
                    llm, gate_search, candidate, k=cfg.novelty.k,
                    prompts_dir=prompts_dir, output_language=cfg.generation.output_language,
                )
                candidate["prior_art"] = prior_note
                if not passed and not last:
                    continue
            break
        if parsed is None:
            # Resilience (DESIGN.md philosophy): skip this slot and keep what parsed.
            print(f"[warning] slot '{slot}' could not be generated after {_SLOT_ATTEMPTS} "
                  "attempts (bad JSON or a provider/API error); skipping it.", file=sys.stderr)
            continue
        idea = Idea(
            idea_id=new_id(),
            batch_id=batch_id,
            slot=slot,
            idx=len(ideas) + 1,
            title=parsed["title"],
            bridge=parsed.get("bridge", ""),
            mechanism=parsed["mechanism"],
            why_now=parsed["why_now"],
            math_structure=parsed["math_structure"],
            prior_art=parsed.get("prior_art", "[unchecked]"),
            tractability=parsed["tractability"],
            fit_to_program=parsed["fit_to_program"],
            behavior=parsed.get("behavior", ""),
            random_genes=json.dumps(used_genes, ensure_ascii=False) if used_genes else "",
            created_at=now_iso(),
        )
        ideas.append(idea)
        siblings.append(
            {"date": date, "slot": slot, "title": idea.title, "behavior": idea.behavior}
        )

    if not ideas:
        raise RuntimeError(
            "all slots failed to produce valid JSON after retries; "
            "check the model/provider settings."
        )

    _apply_self_ranks(llm, cfg, ideas, prompts_dir)

    store.insert_batch(batch)
    for idea in ideas:
        store.insert_idea(idea)

    # Novelty check (DESIGN.md section 7). Lazy import: leaf module.
    from .novelty import attach_prior_art
    try:
        attach_prior_art(ideas, search, store, k=cfg.novelty.k, batch_id=batch_id)
    except Exception:
        pass  # advisory step; never block a batch on search failure

    return batch, store.ideas_for_batch(batch_id)
