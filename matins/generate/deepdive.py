"""On-demand deep dive: a grounded, cited briefing for one flagged idea.

Triggered by 'dig #N' in a reply (handled during `matins collect`) or `matins dig N`.
Pipeline: decompose the idea into English search queries (stronger model) -> search
arXiv (real abstracts) + Tavily web -> synthesize a briefing grounded ONLY in the
fetched sources, with explicit citations and an 'unverified' section. Extends the
advisory novelty check (DESIGN.md section 7) for the cross-domain / blind-spot case.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from ..config import Config
from ..providers.base import get_llm_provider
from ..providers.search_web import ArxivSearchProvider, get_web_searcher
from ..store.db import Store
from ..store.models import Idea
from .schema import strip_code_fences
from .slots import load_prompt, output_language_instruction, render_template


def _format_idea(idea: Idea) -> str:
    parts = [f"Title: {idea.title}"]
    if idea.mechanism:
        parts.append(f"Mechanism: {idea.mechanism}")
    if idea.math_structure:
        parts.append(f"Math structure: {idea.math_structure}")
    if idea.fit_to_program:
        parts.append(f"Fit: {idea.fit_to_program}")
    return "\n".join(parts)


def propose_queries(llm, idea: Idea, prompts_dir, max_queries: int) -> list[str]:
    """Ask the model for up to `max_queries` English search queries; tolerant parse."""
    prompt = render_template(
        load_prompt(prompts_dir, "deepdive_queries.txt"),
        {"IDEA": _format_idea(idea), "N": str(max_queries)},
    )
    queries: list[str] = []
    try:
        raw = strip_code_fences(llm.generate(prompt, temperature=0.0))
        start, end = raw.find("["), raw.rfind("]")
        if start != -1 and end != -1:
            arr = json.loads(raw[start:end + 1])
            queries = [str(q).strip() for q in arr if str(q).strip()]
    except Exception:
        queries = []
    if not queries:  # fallback: ASCII keywords from the title
        words = re.findall(r"[A-Za-z][A-Za-z0-9\-]+", idea.title or "")
        queries = [" ".join(words[:4])] if words else [idea.title or ""]
    return queries[:max_queries]


def gather_sources(searchers, queries, k: int, max_sources: int = 10) -> list[dict]:
    """Run every query on every searcher; de-dupe by URL, tag origin, and return at
    most `max_sources` items.

    Sources are merged round-robin across searchers (arxiv[0], web[0], arxiv[1], ...)
    so the cap keeps a balance of source types rather than letting the first searcher
    fill all the slots. A briefing grounded in ~10 well-chosen sources beats one
    drowning in 40 (and avoids overlong, timeout-prone synthesis prompts).
    """
    seen: set[str] = set()
    buckets: list[list[dict]] = []
    for sp, label in searchers:
        bucket: list[dict] = []
        for q in queries:
            try:
                results = sp.search(q, k=k) or []
            except Exception:
                continue
            for r in results:
                url = (r.get("url") or "").strip()
                key = url or (r.get("title") or "").strip()
                if not key or key in seen:
                    continue
                seen.add(key)
                bucket.append({
                    "title": (r.get("title") or "").strip(),
                    "url": url,
                    "snippet": (r.get("snippet") or "").strip(),
                    "via": label,
                })
        buckets.append(bucket)

    out: list[dict] = []
    depth = 0
    while len(out) < max_sources and any(depth < len(b) for b in buckets):
        for b in buckets:
            if depth < len(b):
                out.append(b[depth])
                if len(out) >= max_sources:
                    break
        depth += 1
    return out


def _format_sources(sources: list[dict]) -> str:
    if not sources:
        return "(no sources found)"
    lines = []
    for i, s in enumerate(sources, start=1):
        lines.append(f"[{i}] ({s['via']}) {s['title']} — {s['url']}\n    {s['snippet']}")
    return "\n".join(lines)


def synthesize_brief(llm, idea: Idea, sources, prompts_dir, output_language: str) -> str:
    prompt = render_template(
        load_prompt(prompts_dir, "deepdive_brief.txt"),
        {
            "IDEA": _format_idea(idea),
            "SOURCES": _format_sources(sources),
            "OUTPUT_LANGUAGE": output_language_instruction(output_language),
        },
    )
    return llm.generate(prompt, temperature=0.2)


def run_deep_dive(cfg: Config, store: Store, idea: Idea) -> dict:
    """Full deep dive for one idea; persists to the store and returns the result."""
    llm = get_llm_provider(cfg, model=cfg.dig_model())
    prompts_dir = cfg.prompts_dir()

    searchers = [(ArxivSearchProvider(), "arxiv")]
    web = get_web_searcher(cfg)
    if web is not None:
        searchers.append((web, "web"))

    queries = propose_queries(llm, idea, prompts_dir, cfg.deep_dive.max_queries)
    sources = gather_sources(
        searchers, queries, cfg.deep_dive.k_per_query,
        max_sources=cfg.deep_dive.max_sources,
    )
    brief = synthesize_brief(llm, idea, sources, prompts_dir, cfg.generation.output_language)

    store.save_deep_dive(idea.idea_id, brief, json.dumps(sources, ensure_ascii=False))
    return {"brief": brief, "sources": sources, "queries": queries}


def write_brief_md(cfg: Config, idea: Idea, brief: str) -> Path:
    """Write a human-readable deep_dives/<date>-<slot>-<idx>.md mirror."""
    d = cfg.deep_dives_dir()
    d.mkdir(parents=True, exist_ok=True)
    slug = f"{(idea.created_at or '')[:10]}-{idea.slot}-{idea.idx}"
    path = d / f"{slug}.md"
    path.write_text(f"# Deep dive — {idea.title}\n\n{brief}\n", encoding="utf-8")
    return path
