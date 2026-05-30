"""Memory kernels: time-series reads of the event log (DESIGN.md section 5).

The two memory tiers are *reads* of the append-only log, never separately
maintained state. A kernel is parameterised by a window, a stride (low-pass
sampling), and an aggregator that turns the raw events into either a fast
generation hint or a slow proposed skill diff. Both aggregators run the LLM;
the calling daily loop must never crash on an advisory step.
"""
from __future__ import annotations

from ..generate.slots import load_prompt, render_template


def format_events(events: list[dict]) -> str:
    """Render the event dicts as one compact line each, skipping None fields."""
    lines: list[str] = []
    for e in events:
        parts: list[str] = []
        date = e.get("date")
        if date is not None:
            parts.append(str(date))
        slot = e.get("slot")
        if slot is not None:
            parts.append(str(slot))
        idx = e.get("idx")
        if idx is not None:
            parts.append("#" + str(idx))
        title = e.get("title")
        if title is not None:
            parts.append(str(title))
        self_rank = e.get("self_rank")
        if self_rank is not None:
            parts.append("self_rank=" + str(self_rank))
        user_rank = e.get("user_rank")
        if user_rank is not None:
            parts.append("user_rank=" + str(user_rank))
        user_comment = e.get("user_comment")
        if user_comment is not None:
            parts.append("comment=" + str(user_comment))
        lines.append(" | ".join(parts))
    return "\n".join(lines)


def compute_memory(kernel_cfg, store, llm, prompts_dir) -> str:
    """Read recent events for a kernel and aggregate them via the LLM.

    Returns '' when there is no history or the aggregator is unknown, so the
    caller can treat an empty memory as a benign cold-start signal.
    """
    events = store.recent_events(kernel_cfg.window_days, kernel_cfg.stride)
    if not events:
        return ""
    ev = format_events(events)
    if kernel_cfg.aggregator == "llm_summarize_recent":
        prompt = render_template(load_prompt(prompts_dir, "summarize_recent.txt"), {"EVENTS": ev})
        return llm.generate(prompt, temperature=0.0)
    if kernel_cfg.aggregator == "llm_propose_skill_diff":
        cur = store.active_skill()
        cur_text = cur.content if cur else "(none yet)"
        prompt = render_template(
            load_prompt(prompts_dir, "propose_skill_diff.txt"),
            {"EVENTS": ev, "CURRENT_SKILL": cur_text},
        )
        return llm.generate(prompt, temperature=0.2)
    return ""
