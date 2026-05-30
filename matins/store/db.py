"""SQLite-backed append-only log + derived-view queries (DESIGN.md sections 4-5).

Design stance: rows are inserted, not destructively overwritten. The two memory
tiers are *reads* of this log (see memory/kernels.py), never separately maintained
state. So the only mutations here are: insert rows, fill prior_art / tau / digest
ids on a batch (single-write enrichment), advance the messaging offset, and flip a
skill version's approval flag.
"""
from __future__ import annotations

import json
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .models import Batch, Feedback, Idea, SkillVersion, TasteHypothesis

SCHEMA = """
CREATE TABLE IF NOT EXISTS batches (
  batch_id        TEXT PRIMARY KEY,
  date            TEXT,
  skill_version   INTEGER,
  temperature     REAL,
  provider        TEXT, model TEXT,
  self_user_tau   REAL,
  digest_msg_id   TEXT,
  created_at      TEXT
);

CREATE TABLE IF NOT EXISTS ideas (
  idea_id         TEXT PRIMARY KEY,
  batch_id        TEXT REFERENCES batches(batch_id),
  slot            TEXT,
  idx             INTEGER,
  title           TEXT,
  mechanism       TEXT,
  why_now         TEXT,
  math_structure  TEXT,
  prior_art       TEXT,
  tractability    TEXT,
  fit_to_program  TEXT,
  random_genes    TEXT,
  self_rank       INTEGER,
  self_rationale  TEXT,
  created_at      TEXT
);

CREATE TABLE IF NOT EXISTS feedback (
  idea_id         TEXT REFERENCES ideas(idea_id),
  user_rank       INTEGER,
  user_comment    TEXT,
  source          TEXT,
  created_at      TEXT
);

CREATE TABLE IF NOT EXISTS retrieval_log (
  batch_id TEXT, query TEXT, source TEXT,
  result_ids TEXT, created_at TEXT
);

CREATE TABLE IF NOT EXISTS taste_hypotheses (
  hyp_id TEXT PRIMARY KEY,
  text TEXT,
  kind TEXT,
  evidence TEXT,
  confidence REAL, occurrence INTEGER,
  status TEXT,
  first_seen TEXT, last_seen TEXT
);

CREATE TABLE IF NOT EXISTS skill_versions (
  version INTEGER PRIMARY KEY,
  content TEXT, parent_version INTEGER,
  diff_summary TEXT, approved INTEGER,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS messaging_state (
  channel TEXT PRIMARY KEY,
  last_update_id TEXT
);

CREATE TABLE IF NOT EXISTS favorites (
  idea_id    TEXT PRIMARY KEY REFERENCES ideas(idea_id),
  note       TEXT,
  created_at TEXT
);

CREATE TABLE IF NOT EXISTS deep_dives (
  idea_id    TEXT PRIMARY KEY REFERENCES ideas(idea_id),
  brief      TEXT,
  sources    TEXT,
  created_at TEXT
);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def today_iso() -> str:
    return datetime.now(timezone.utc).date().isoformat()


def new_id() -> str:
    return uuid.uuid4().hex


def _cutoff_date(window_days: int) -> str:
    return (datetime.now(timezone.utc) - timedelta(days=window_days)).date().isoformat()


class Store:
    """Thin typed wrapper over a single SQLite file."""

    def __init__(self, path: str | Path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(self.path)
        self.conn.row_factory = sqlite3.Row
        self.init_schema()

    def init_schema(self) -> None:
        self.conn.executescript(SCHEMA)
        self.conn.commit()

    def close(self) -> None:
        self.conn.close()

    # ---- batches ---------------------------------------------------------
    def insert_batch(self, b: Batch) -> None:
        self.conn.execute(
            """INSERT INTO batches (batch_id, date, skill_version, temperature,
               provider, model, self_user_tau, digest_msg_id, created_at)
               VALUES (?,?,?,?,?,?,?,?,?)""",
            (b.batch_id, b.date, b.skill_version, b.temperature, b.provider,
             b.model, b.self_user_tau, b.digest_msg_id, b.created_at or now_iso()),
        )
        self.conn.commit()

    def batch_for_date(self, date: str) -> Batch | None:
        """Idempotency guard for `matins run` (one batch per date)."""
        row = self.conn.execute(
            "SELECT * FROM batches WHERE date=? ORDER BY created_at DESC LIMIT 1", (date,)
        ).fetchone()
        return Batch(**dict(row)) if row else None

    def latest_batch(self) -> Batch | None:
        row = self.conn.execute(
            "SELECT * FROM batches ORDER BY created_at DESC LIMIT 1"
        ).fetchone()
        return Batch(**dict(row)) if row else None

    def set_batch_digest_msg_id(self, batch_id: str, digest_msg_id: str) -> None:
        self.conn.execute(
            "UPDATE batches SET digest_msg_id=? WHERE batch_id=?", (digest_msg_id, batch_id)
        )
        self.conn.commit()

    def set_batch_tau(self, batch_id: str, tau: float | None) -> None:
        self.conn.execute(
            "UPDATE batches SET self_user_tau=? WHERE batch_id=?", (tau, batch_id)
        )
        self.conn.commit()

    # ---- ideas -----------------------------------------------------------
    def insert_idea(self, idea: Idea) -> None:
        self.conn.execute(
            """INSERT INTO ideas (idea_id, batch_id, slot, idx, title, mechanism,
               why_now, math_structure, prior_art, tractability, fit_to_program,
               random_genes, self_rank, self_rationale, created_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)""",
            (idea.idea_id, idea.batch_id, idea.slot, idea.idx, idea.title,
             idea.mechanism, idea.why_now, idea.math_structure, idea.prior_art,
             idea.tractability, idea.fit_to_program, idea.random_genes,
             idea.self_rank, idea.self_rationale, idea.created_at or now_iso()),
        )
        self.conn.commit()

    def update_idea_prior_art(self, idea_id: str, prior_art: str) -> None:
        self.conn.execute(
            "UPDATE ideas SET prior_art=? WHERE idea_id=?", (prior_art, idea_id)
        )
        self.conn.commit()

    def ideas_for_batch(self, batch_id: str) -> list[Idea]:
        rows = self.conn.execute(
            "SELECT * FROM ideas WHERE batch_id=? ORDER BY idx ASC", (batch_id,)
        ).fetchall()
        return [Idea(**dict(r)) for r in rows]

    # ---- feedback --------------------------------------------------------
    def insert_feedback(self, fb: Feedback) -> None:
        self.conn.execute(
            """INSERT INTO feedback (idea_id, user_rank, user_comment, source, created_at)
               VALUES (?,?,?,?,?)""",
            (fb.idea_id, fb.user_rank, fb.user_comment, fb.source, fb.created_at or now_iso()),
        )
        self.conn.commit()

    def feedback_for_idea(self, idea_id: str) -> Feedback | None:
        row = self.conn.execute(
            "SELECT * FROM feedback WHERE idea_id=? ORDER BY created_at DESC LIMIT 1", (idea_id,)
        ).fetchone()
        return Feedback(**dict(row)) if row else None

    def batch_has_feedback(self, batch_id: str) -> bool:
        row = self.conn.execute(
            """SELECT 1 FROM feedback f JOIN ideas i ON f.idea_id=i.idea_id
               WHERE i.batch_id=? LIMIT 1""", (batch_id,)
        ).fetchone()
        return row is not None

    # ---- derived events (memory kernels) ---------------------------------
    def recent_events(self, window_days: int, stride: int = 1) -> list[dict]:
        """Ordered (idea + its feedback) events within a window, coarsely sampled.

        The log is treated as a time series of events. `window_days` bounds how far
        back we look; `stride` samples every Nth *batch* (oldest-first), acting as the
        low-pass filter described in DESIGN.md section 5. Returns flat event dicts for
        the selected batches, oldest first.
        """
        cutoff = _cutoff_date(window_days)
        rows = self.conn.execute(
            """SELECT b.batch_id AS batch_id, b.date AS date, b.self_user_tau AS tau,
                      i.idea_id AS idea_id, i.slot AS slot, i.idx AS idx,
                      i.title AS title, i.mechanism AS mechanism,
                      i.math_structure AS math_structure, i.tractability AS tractability,
                      i.self_rank AS self_rank, i.self_rationale AS self_rationale,
                      f.user_rank AS user_rank, f.user_comment AS user_comment
               FROM ideas i
               JOIN batches b ON i.batch_id = b.batch_id
               LEFT JOIN feedback f ON f.idea_id = i.idea_id
               WHERE b.date >= ?
               ORDER BY b.date ASC, i.idx ASC""",
            (cutoff,),
        ).fetchall()

        # Distinct batch order (oldest first), then keep every `stride`-th batch.
        seen: list[str] = []
        for r in rows:
            if r["batch_id"] not in seen:
                seen.append(r["batch_id"])
        stride = max(1, stride)
        keep = set(seen[::stride])
        return [dict(r) for r in rows if r["batch_id"] in keep]

    # ---- retrieval log ---------------------------------------------------
    def log_retrieval(self, batch_id: str, query: str, source: str, result_ids: list[str]) -> None:
        self.conn.execute(
            """INSERT INTO retrieval_log (batch_id, query, source, result_ids, created_at)
               VALUES (?,?,?,?,?)""",
            (batch_id, query, source, json.dumps(result_ids), now_iso()),
        )
        self.conn.commit()

    def recent_result_ids(self, days: int) -> set[str]:
        """Result ids seen recently, for de-duplicating future batches."""
        cutoff = (datetime.now(timezone.utc) - timedelta(days=days)).isoformat()
        rows = self.conn.execute(
            "SELECT result_ids FROM retrieval_log WHERE created_at >= ?", (cutoff,)
        ).fetchall()
        out: set[str] = set()
        for r in rows:
            try:
                out.update(json.loads(r["result_ids"] or "[]"))
            except (ValueError, TypeError):
                continue
        return out

    # ---- taste hypotheses (FAST memory) ----------------------------------
    def open_hypotheses(self) -> list[TasteHypothesis]:
        rows = self.conn.execute(
            "SELECT * FROM taste_hypotheses WHERE status IN ('open','confirmed') ORDER BY occurrence DESC"
        ).fetchall()
        return [TasteHypothesis(**dict(r)) for r in rows]

    def find_hypothesis(self, text: str) -> TasteHypothesis | None:
        row = self.conn.execute(
            "SELECT * FROM taste_hypotheses WHERE text=? LIMIT 1", (text,)
        ).fetchone()
        return TasteHypothesis(**dict(row)) if row else None

    def upsert_hypothesis(self, h: TasteHypothesis) -> None:
        existing = self.find_hypothesis(h.text)
        if existing is None:
            self.conn.execute(
                """INSERT INTO taste_hypotheses
                   (hyp_id, text, kind, evidence, confidence, occurrence, status, first_seen, last_seen)
                   VALUES (?,?,?,?,?,?,?,?,?)""",
                (h.hyp_id or new_id(), h.text, h.kind, h.evidence, h.confidence,
                 h.occurrence, h.status, h.first_seen or now_iso(), h.last_seen or now_iso()),
            )
        else:
            self.conn.execute(
                """UPDATE taste_hypotheses
                   SET evidence=?, confidence=?, occurrence=?, status=?, last_seen=?
                   WHERE hyp_id=?""",
                (h.evidence, h.confidence, h.occurrence, h.status, now_iso(), existing.hyp_id),
            )
        self.conn.commit()

    def hypotheses_over_threshold(self, threshold: int) -> list[TasteHypothesis]:
        rows = self.conn.execute(
            "SELECT * FROM taste_hypotheses WHERE occurrence >= ? AND status='open'", (threshold,)
        ).fetchall()
        return [TasteHypothesis(**dict(r)) for r in rows]

    # ---- skill versions (SLOW memory) ------------------------------------
    def active_skill(self) -> SkillVersion | None:
        """Highest approved skill version (the live taste skill)."""
        row = self.conn.execute(
            "SELECT * FROM skill_versions WHERE approved=1 ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return SkillVersion(**dict(row)) if row else None

    def latest_skill_version(self) -> int:
        row = self.conn.execute("SELECT MAX(version) AS v FROM skill_versions").fetchone()
        return int(row["v"]) if row and row["v"] is not None else 0

    def get_skill_version(self, version: int) -> SkillVersion | None:
        row = self.conn.execute(
            "SELECT * FROM skill_versions WHERE version=?", (version,)
        ).fetchone()
        return SkillVersion(**dict(row)) if row else None

    def insert_skill_version(self, content: str, parent_version: int | None,
                             diff_summary: str, approved: int = 0) -> int:
        version = self.latest_skill_version() + 1
        self.conn.execute(
            """INSERT INTO skill_versions
               (version, content, parent_version, diff_summary, approved, created_at)
               VALUES (?,?,?,?,?,?)""",
            (version, content, parent_version, diff_summary, approved, now_iso()),
        )
        self.conn.commit()
        return version

    def approve_skill(self, version: int) -> None:
        self.conn.execute("UPDATE skill_versions SET approved=1 WHERE version=?", (version,))
        self.conn.commit()

    def pending_skill_version(self) -> SkillVersion | None:
        """Most recent unapproved proposal awaiting the human."""
        row = self.conn.execute(
            "SELECT * FROM skill_versions WHERE approved=0 ORDER BY version DESC LIMIT 1"
        ).fetchone()
        return SkillVersion(**dict(row)) if row else None

    # ---- messaging offset ------------------------------------------------
    def get_offset(self, channel: str) -> str | None:
        row = self.conn.execute(
            "SELECT last_update_id FROM messaging_state WHERE channel=?", (channel,)
        ).fetchone()
        return row["last_update_id"] if row else None

    def set_offset(self, channel: str, last_update_id: str | None) -> None:
        self.conn.execute(
            """INSERT INTO messaging_state (channel, last_update_id) VALUES (?,?)
               ON CONFLICT(channel) DO UPDATE SET last_update_id=excluded.last_update_id""",
            (channel, last_update_id),
        )
        self.conn.commit()

    # ---- favorites (curated "must try" ideas) ----------------------------
    def add_favorite(self, idea_id: str, note: str = "") -> bool:
        """Copy an idea into the curated favorites. Returns True if newly added,
        False if it was already a favorite (idempotent)."""
        cur = self.conn.execute(
            "INSERT OR IGNORE INTO favorites (idea_id, note, created_at) VALUES (?,?,?)",
            (idea_id, note, now_iso()),
        )
        self.conn.commit()
        return cur.rowcount > 0

    def is_favorite(self, idea_id: str) -> bool:
        return self.conn.execute(
            "SELECT 1 FROM favorites WHERE idea_id=? LIMIT 1", (idea_id,)
        ).fetchone() is not None

    def list_favorites(self) -> list[tuple[Idea, str, str]]:
        """Return [(Idea, note, favorited_at)], newest first."""
        rows = self.conn.execute(
            """SELECT i.*, f.note AS fav_note, f.created_at AS fav_created_at
               FROM favorites f JOIN ideas i ON i.idea_id = f.idea_id
               ORDER BY f.created_at DESC"""
        ).fetchall()
        out: list[tuple[Idea, str, str]] = []
        for r in rows:
            d = dict(r)
            note = d.pop("fav_note", "") or ""
            fav_at = d.pop("fav_created_at", "") or ""
            idea = Idea(**{k: v for k, v in d.items() if k in Idea.__dataclass_fields__})
            out.append((idea, note, fav_at))
        return out

    # ---- deep dives (on-demand grounded briefings) -----------------------
    def save_deep_dive(self, idea_id: str, brief: str, sources_json: str = "[]") -> None:
        self.conn.execute(
            """INSERT INTO deep_dives (idea_id, brief, sources, created_at) VALUES (?,?,?,?)
               ON CONFLICT(idea_id) DO UPDATE SET
                 brief=excluded.brief, sources=excluded.sources, created_at=excluded.created_at""",
            (idea_id, brief, sources_json, now_iso()),
        )
        self.conn.commit()

    def get_deep_dive(self, idea_id: str) -> dict | None:
        row = self.conn.execute(
            "SELECT brief, sources, created_at FROM deep_dives WHERE idea_id=?", (idea_id,)
        ).fetchone()
        return dict(row) if row else None
