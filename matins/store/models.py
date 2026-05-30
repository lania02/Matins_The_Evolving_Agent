"""Row models mirroring the SQLite schema (DESIGN.md section 4).

These are plain dataclasses. They are the in-memory shape of a row; the Store
(db.py) is responsible for (de)serialising them. Field names match column names
exactly so conversion stays mechanical.
"""
from __future__ import annotations

from dataclasses import dataclass

# Generation slots (DESIGN.md section 6). Order is canonical.
SLOTS = ["highfit", "adjacent", "orthogonal", "random"]

# Slot -> short human label used in digests / prompts.
SLOT_LABELS = {
    "highfit": "high-fit",
    "adjacent": "adjacent-stretch",
    "orthogonal": "orthogonal",
    "random": "random-mutation",
}


@dataclass
class Batch:
    batch_id: str
    date: str
    skill_version: int | None = None
    temperature: float | None = None
    provider: str = ""
    model: str = ""
    self_user_tau: float | None = None
    digest_msg_id: str | None = None
    created_at: str = ""


@dataclass
class Idea:
    idea_id: str
    batch_id: str
    slot: str
    idx: int
    title: str = ""
    mechanism: str = ""
    why_now: str = ""
    math_structure: str = ""
    prior_art: str = ""
    tractability: str = ""
    fit_to_program: str = ""
    random_genes: str = ""
    self_rank: int | None = None
    self_rationale: str = ""
    created_at: str = ""


@dataclass
class Feedback:
    idea_id: str
    user_rank: int | None = None
    user_comment: str = ""
    source: str = "telegram"          # telegram | cli | card
    created_at: str = ""


@dataclass
class TasteHypothesis:
    hyp_id: str
    text: str
    kind: str                          # topic | structure  (DESIGN.md section 10)
    evidence: str = "[]"               # JSON list of idea_ids
    confidence: float = 0.0
    occurrence: int = 0
    status: str = "open"               # open | confirmed | rejected | retired
    first_seen: str = ""
    last_seen: str = ""


@dataclass
class SkillVersion:
    version: int
    content: str
    parent_version: int | None = None
    diff_summary: str = ""
    approved: int = 0                  # requires human approval to activate
    created_at: str = ""
