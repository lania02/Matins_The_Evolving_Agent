"""Anti-red-ocean saturation gate for the orthogonal slot (post-generation B2 + B1).

The orthogonal slot, asked to jump to a distant domain, tends to reach for that domain's
most FAMOUS topic -- a saturated "red ocean" that does not inspire. This module grounds a
keep/regenerate decision in REAL literature density (B2: a search + a total-match count)
and then asks the model to judge WITH that evidence in hand (B1), rather than on its own
unanchored opinion. The same loop that already retries a slot for bad JSON / near-duplicate
titles also retries it here, so a rejected red-ocean idea is regenerated, not dropped.

Advisory and fail-open: any error, missing count, or unparseable verdict KEEPS the idea --
a flaky gate must never sink the morning's batch.
"""
from __future__ import annotations

import json
from pathlib import Path

from ..providers.base import LLMProvider, SearchProvider
from .novelty import build_query_from_fields, format_prior_art
from .schema import strip_code_fences
from .slots import load_prompt, output_language_instruction, render_template


def field_density(search: SearchProvider, query: str) -> int | None:
    """Total number of existing works matching `query` (B2), or None if unavailable.

    Providers expose this via an optional `count(query)` method (arXiv, OpenAlex);
    providers without it simply yield None and the judge runs on the closest works alone.
    """
    counter = getattr(search, "count", None)
    if not callable(counter):
        return None
    try:
        return counter(query)
    except Exception:
        return None


# Raw corpus counts are hard for a flash-tier judge to calibrate (it reads any 4-digit
# number as "a lot"). We bin the count into a band IN CODE and hand the judge the band +
# guidance, so the saturation call rests on a deterministic scale, not the model's number
# sense. Thresholds are tuned to OpenAlex loose-`search` counts for a ~4-keyword query: a
# saturated flagship pairing lands ~1e5; a genuine niche lands ~1e3. Heuristic, tunable.
_SATURATED_AT = 50_000
_BUSY_AT = 8_000


def _density_band(count: int | None) -> str:
    """A human-legible density band + guidance string for the judge prompt."""
    if count is None:
        return "unknown -- literature density could not be measured; judge on the works below"
    if count >= _SATURATED_AT:
        return f"~{count} works -- SATURATED (red-ocean scale: the area is flooded)"
    if count >= _BUSY_AT:
        return f"~{count} works -- BUSY (an active area, but not necessarily saturated)"
    return f"~{count} works -- SPARSE (niche or near-empty; this is NOT a red ocean)"


def _build_judge_prompt(
    candidate: dict, density: int | None, results: list[dict],
    prompts_dir: str | Path, output_language: str,
) -> str:
    closest = "\n".join(
        f"- {str(r.get('title', '')).strip()}" for r in results[:5] if r.get("title")
    ) or "(none found)"
    tokens = {
        "TITLE": candidate.get("title", ""),
        "MECHANISM": candidate.get("mechanism", ""),
        "WHY_NOW": candidate.get("why_now", ""),
        "DENSITY": _density_band(density),
        "CLOSEST": closest,
        "OUTPUT_LANGUAGE": output_language_instruction(output_language),
    }
    return render_template(load_prompt(prompts_dir, "saturation_judge.txt"), tokens)


def _parse_verdict(text: str) -> dict:
    """Tolerant-parse the judge reply into a dict; {} if unrecoverable (-> keep)."""
    raw = strip_code_fences(text)
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        start, end = raw.find("{"), raw.rfind("}")
        if start == -1 or end == -1:
            return {}
        try:
            obj = json.loads(raw[start : end + 1])
        except (ValueError, TypeError):
            return {}
    return obj if isinstance(obj, dict) else {}


def gate_saturation(
    llm: LLMProvider,
    search: SearchProvider,
    candidate: dict,
    *,
    k: int,
    prompts_dir: str | Path,
    output_language: str,
) -> tuple[bool, str]:
    """B2 (ground in literature) + B1 (judge with that evidence) for one candidate idea.

    Returns (passed, prior_art). `passed` is False only when the model, given the real
    density + closest works, judges the idea a saturated, undifferentiated red-ocean
    restatement. `prior_art` is the closest-work note from the SAME search, returned so
    the caller can reuse it and the novelty step need not search this idea again.
    Fail-open: a search/LLM failure returns (True, ...) so the batch is never blocked.
    """
    query = build_query_from_fields(
        candidate.get("title", ""),
        candidate.get("math_structure", ""),
        candidate.get("mechanism", ""),
    )
    try:
        results = search.search(query, k=k) or []
    except Exception:
        return True, "[unchecked]"

    prior_art = format_prior_art(results)
    density = field_density(search, query)
    try:
        text = llm.generate(
            _build_judge_prompt(candidate, density, results, prompts_dir, output_language),
            temperature=0.0,
        )
    except Exception:
        return True, prior_art

    verdict = _parse_verdict(text)
    passed = str(verdict.get("verdict", "keep")).strip().lower() != "regenerate"
    return passed, prior_art
