"""Grounding lenses (Idea Oracle port): inject a REAL external vantage into generation.

The recurring failure of a taste-only idea engine is abstraction -- ideas float in
method-space, restate the user's own work, and read as "these two could be combined."
A lens fixes that by handing a slot a concrete, externally-sourced vantage it MUST serve:
either an occupation + its real task-frictions (O*NET-style, for buildable/product ideas)
or a research domain + real data + open questions (for research ideas). The pool is
human-curated in prompts/lenses.yaml -- deliberately NOT model-invented, which is the whole
point ("doctors are busy" is worthless; a named task-friction is not).

Zero-dep, fail-open: a missing or empty lenses.yaml simply yields no lenses and generation
proceeds unchanged.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path

import yaml

# lens_mode -> which kinds are eligible. "off" is handled by the caller (no sampling at all).
_MODE_KINDS = {
    "research": ("research",),
    "product": ("occupation",),
    "mixed": ("occupation", "research"),
}


@dataclass
class Lens:
    kind: str            # "occupation" | "research"
    name: str            # short tag recorded on the idea (the "vantage") + dedup key
    detail: dict         # occupation: tasks/frictions/would_pay_for; research: phenomena/data/open_questions


def load_lenses(prompts_dir: str | Path) -> list[Lens]:
    """Load the curated vantage pool from prompts/lenses.yaml ([] if absent/empty)."""
    path = Path(prompts_dir, "lenses.yaml")
    if not path.exists():
        return []
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: list[Lens] = []
    for kind in ("occupation", "research"):
        for item in data.get(kind) or []:
            if not isinstance(item, dict):
                continue
            name = str(item.get("name", "")).strip()
            if not name:
                continue
            out.append(Lens(kind=kind, name=name,
                            detail={k: v for k, v in item.items() if k != "name"}))
    return out


def lenses_for_mode(pool: list[Lens], mode: str) -> list[Lens]:
    """Filter the pool to the kinds a given lens_mode is allowed to use."""
    kinds = _MODE_KINDS.get(mode, ())
    return [l for l in pool if l.kind in kinds]


def sample_lenses(pool: list[Lens], k: int, *, exclude_names=(), rng: random.Random | None = None) -> list[Lens]:
    """Pick `k` lenses, preferring ones not recently used and not repeating within the draw.

    Falls back to repeats only when the pool is smaller than k, so each lens slot still
    gets a vantage even with a tiny pool. Returns [] when the pool is empty.
    """
    if not pool or k <= 0:
        return []
    r = rng or random
    excl = set(exclude_names)
    fresh = [l for l in pool if l.name not in excl] or list(pool)
    r.shuffle(fresh)
    if len(fresh) >= k:
        return fresh[:k]
    out = list(fresh)                                   # take all fresh, then top up with repeats
    while len(out) < k:
        extra = list(pool)
        r.shuffle(extra)
        out.extend(extra[: k - len(out)])
    return out[:k]


def _line(label: str, items) -> str:
    vals = [str(x).strip() for x in (items or []) if str(x).strip()]
    return f"- {label}: " + "; ".join(vals) if vals else ""


def render_lens_block(lens: Lens | None) -> str:
    """Render one lens into a prompt block (header + how-to), or "" when there is none.

    The block is self-contained (its own header + instruction) so a slot template only
    needs a bare {{LENS}} placeholder: when no lens is assigned it renders to nothing.
    """
    if not lens:
        return ""
    d = lens.detail or {}
    if lens.kind == "occupation":
        body = "\n".join(filter(None, [
            f"PROFESSION: {lens.name}",
            _line("real tasks", d.get("tasks")),
            _line("real frictions / pain points", d.get("frictions")),
            _line("would pay to fix", d.get("would_pay_for")),
        ]))
        howto = (
            "Ground THIS idea in the profession above: target ONE of its listed, real "
            "task-frictions, and make 'tractability' a concrete weekend-buildable demo for "
            "that worker. Do NOT invent generic 'they are busy' pain -- use the frictions listed."
        )
    else:  # research
        body = "\n".join(filter(None, [
            f"DOMAIN / PHENOMENON: {lens.name}",
            _line("real phenomena", d.get("phenomena")),
            _line("available data", d.get("data")),
            _line("open questions", d.get("open_questions")),
        ]))
        howto = (
            "Ground THIS idea in the domain above: aim at ONE of its real phenomena or open "
            "questions, and prefer a falsifiable test on the listed data. Do NOT drift back "
            "into pure abstraction -- the vantage is the anchor, not a decoration."
        )
    return ("== GROUNDING VANTAGE (real, externally sourced -- do not invent it) ==\n"
            f"{body}\n{howto}")
