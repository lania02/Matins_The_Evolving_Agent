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

        add("Bridge", idea.bridge)                   # the collision, shown first as the headline
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


def render_overview(store, batches, *, db_path: str | None = None) -> str:
    """Read-only markdown view of the DB: batches, the fresh retrieval that seeded
    each one, ideas (with self/user ranks), feedback, and deep-dive presence.

    `store` is a Store; `batches` a list of Batch (newest first). Pulls everything
    else (ideas, feedback, retrieval log, deep dives) live from the store.
    """
    n_ideas = sum(len(store.ideas_for_batch(b.batch_id)) for b in batches)
    out = ["# 🔭 Matins — database view", ""]
    if db_path:
        out.append(f"_source: `{db_path}`_")
    out.append(f"_{len(batches)} batch(es) · {n_ideas} idea(s) · newest first._")
    out.append("")

    for b in batches:
        ideas = store.ideas_for_batch(b.batch_id)
        tau = b.self_user_tau
        tau_s = f"{tau:.2f}" if isinstance(tau, (int, float)) else "—"
        skill_v = b.skill_version if b.skill_version is not None else "-"
        out.append(f"## {b.date}")
        out.append(
            f"_provider: {b.provider} · model: {b.model} · temp: {b.temperature} · "
            f"skill v{skill_v} · self↔user τ: {tau_s}_"
        )
        out.append("")

        # The fresh-literature feed that seeded this batch (the 'blend:' log row).
        feed = [r for r in store.retrieval_for_batch(b.batch_id)
                if str(r.get("source", "")).startswith("blend")]
        if feed:
            ids = feed[-1].get("result_ids") or []
            out.append(f"**Fresh retrieval fed in** ({len(ids)} items, {feed[-1]['source']}):")
            for item in ids:
                out.append(f"- {item}")
            out.append("")

        # Compact ranking table.
        out.append("| # | slot | title | self-rank | your rank |")
        out.append("|---|------|-------|-----------|-----------|")
        for idea in ideas:
            fb = store.feedback_for_idea(idea.idea_id)
            ur = fb.user_rank if fb and fb.user_rank is not None else "—"
            sr = idea.self_rank if idea.self_rank is not None else "—"
            label = SLOT_LABELS.get(idea.slot, idea.slot)
            title = (idea.title or "").replace("|", "/")
            out.append(f"| {idea.idx} | {label} | {title} | {sr} | {ur} |")
        out.append("")

        # Full idea cards.
        for idea in ideas:
            label = SLOT_LABELS.get(idea.slot, idea.slot)
            badge = " · 🔬 deep-dived" if store.get_deep_dive(idea.idea_id) else ""
            out.append(f"### #{idea.idx} [{label}] {idea.title}{badge}")
            for field_label, value in (
                ("Bridge", idea.bridge),
                ("Mechanism", idea.mechanism),
                ("Why now", idea.why_now),
                ("Math structure", idea.math_structure),
                ("Tractability", idea.tractability),
                ("Fit to program", idea.fit_to_program),
                ("Prior art", idea.prior_art),
            ):
                value = (value or "").strip()
                if value:
                    out.append(f"- **{field_label}:** {value}")
            if (idea.random_genes or "").strip():
                out.append(f"- **Genes:** {idea.random_genes}")
            fb = store.feedback_for_idea(idea.idea_id)
            if fb and (fb.user_rank is not None or (fb.user_comment or "").strip()):
                rank = fb.user_rank if fb.user_rank is not None else "—"
                comment = (fb.user_comment or "").strip()
                line = f"- **Your feedback:** rank {rank}"
                out.append(line + (f" — {comment}" if comment else ""))
            out.append("")
    return "\n".join(out)


def render_favorites_md(favorites) -> str:
    """Render curated favorites -- a list of (Idea, note, favorited_at) tuples --
    as a human-readable markdown document (the favorites.md mirror)."""
    if not favorites:
        return (
            "# ⭐ Matins favorites\n\n"
            "(none yet — reply `must try #N` to a digest, then run `matins collect`.)\n"
        )

    out = ["# ⭐ Matins favorites", "", f"{len(favorites)} curated idea(s), newest first.", ""]
    for idea, note, fav_at in favorites:
        label = SLOT_LABELS.get(idea.slot, idea.slot)
        out.append(f"## {idea.title}")
        out.append(f"_[{label}] · generated {(idea.created_at or '')[:10]} · saved {fav_at[:10]}_")
        if note:
            out.append("")
            out.append(f"> {note}")
        out.append("")
        for field_label, value in (
            ("Bridge", idea.bridge),
            ("Mechanism", idea.mechanism),
            ("Why now", idea.why_now),
            ("Math structure", idea.math_structure),
            ("Tractability", idea.tractability),
            ("Fit to program", idea.fit_to_program),
            ("Prior art", idea.prior_art),
        ):
            value = (value or "").strip()
            if value:
                out.append(f"- **{field_label}:** {value}")
        out.append("")
    return "\n".join(out)
