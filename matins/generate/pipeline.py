"""Generation pipeline: the heart of `matins run` (DESIGN.md section 3, 6, 7).

Flow per batch: assemble context (taste skill + fast memory + fresh retrieval +
interest seed) -> generate one idea per slot (tolerant-parse, one repair retry) ->
self-rank the four -> novelty check each -> persist. Idempotent on date: a second
run for the same date returns the existing batch unchanged.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

from ..config import Config
from ..providers.base import LLMProvider, SearchProvider
from ..store.db import Store, new_id, now_iso, today_iso
from ..store.models import SLOTS, Batch, Idea
from .explore import adaptive_temperature
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


def _read_interest_seed(cfg: Config) -> str:
    path = cfg.interest_seed_path()
    return path.read_text(encoding="utf-8") if path.exists() else ""


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
) -> tuple[Batch, list[Idea]]:
    """Run one daily batch. Idempotent per date."""
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

    slots = SLOTS[: cfg.generation.n_slots]
    ideas: list[Idea] = []
    for slot in slots:
        genes = sample_genes(genes_pool) if slot == "random" else None
        context = {
            "skill": skill_text,
            "fast_memory": fast_memory,
            "retrieval": retrieval,
            "interest_seed": interest_seed,
            "recent_ideas": recent_ideas,
            "archive": revival,
        }
        prompt = build_generation_prompt(
            slot, context, prompts_dir, cfg.generation.output_language, genes
        )
        try:
            parsed = _generate_one(llm, prompt, slot_temperature(explore_temp, slot))
        except IdeaParseError:
            # Resilience (DESIGN.md philosophy): a single slot that won't yield JSON
            # must not sink the whole morning. Skip it and keep what parsed.
            print(f"[warning] slot '{slot}' returned no valid JSON after retries; skipping it.",
                  file=sys.stderr)
            continue
        ideas.append(
            Idea(
                idea_id=new_id(),
                batch_id=batch_id,
                slot=slot,
                idx=len(ideas) + 1,
                title=parsed["title"],
                mechanism=parsed["mechanism"],
                why_now=parsed["why_now"],
                math_structure=parsed["math_structure"],
                prior_art=parsed.get("prior_art", "[unchecked]"),
                tractability=parsed["tractability"],
                fit_to_program=parsed["fit_to_program"],
                behavior=parsed.get("behavior", ""),
                random_genes=json.dumps(genes, ensure_ascii=False) if genes else "",
                created_at=now_iso(),
            )
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
