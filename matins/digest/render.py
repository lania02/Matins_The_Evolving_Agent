"""Render a batch of ideas into plain-text messages for the channel.

DESIGN.md section 9.3. Output is PLAIN text: the messaging adapter is
responsible for any MarkdownV2 escaping. Each idea becomes its own message so
the user can reply per-idea, and every message is kept under the Telegram
4096-character limit.
"""
from __future__ import annotations

from matins.store.models import SLOT_LABELS

# Telegram hard limit per message; keep a margin so the ellipsis always fits.
_MAX_LEN = 4096
_TRUNCATE_AT = _MAX_LEN - 1
_ELLIPSIS = "…"


def _truncate(text: str) -> str:
    """Clamp a message to < 4096 chars, appending an ellipsis if cut."""
    if len(text) < _MAX_LEN:
        return text
    return text[:_TRUNCATE_AT - 1].rstrip() + _ELLIPSIS


def render_digest(batch, ideas, output_language) -> tuple[str, list[str]]:
    """Return (header, [one message per idea]) as plain text.

    `ideas` are rendered sorted by their `idx`. `output_language` is accepted
    for signature compatibility; idea content already carries the language the
    model was told to produce, and the structural labels stay in English.
    """
    skill_version = batch.skill_version if batch.skill_version is not None else "-"
    temperature = batch.temperature if batch.temperature is not None else "-"

    header_lines = [
        f"Matins morning brainstorm — {batch.date}",
        f"skill version: {skill_version}  |  temperature: {temperature}",
        "Reply best to worst by number, e.g. 3>1>4>2 . "
        "Add comments like: #3 your note",
    ]
    header = "\n".join(header_lines)

    messages: list[str] = []
    for idea in sorted(ideas, key=lambda i: i.idx):
        label = SLOT_LABELS.get(idea.slot, idea.slot)
        lines = [f"#{idea.idx} [{label}] {idea.title}"]

        def add(field_label: str, value: str) -> None:
            value = (value or "").strip()
            if value:
                lines.append(f"{field_label}: {value}")

        add("Mechanism", idea.mechanism)
        add("Why now", idea.why_now)
        add("Math structure", idea.math_structure)  # skipped when empty
        add("Tractability", idea.tractability)
        add("Fit to program", idea.fit_to_program)

        prior_art = (idea.prior_art or "").strip()
        if prior_art:
            if len(prior_art) > 600:
                prior_art = prior_art[:599].rstrip() + _ELLIPSIS
            lines.append(f"Prior art: {prior_art}")

        messages.append(_truncate("\n".join(lines)))

    return header, messages
