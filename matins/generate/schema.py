"""Idea JSON schema + portable tolerant parsing (DESIGN.md sections 6.2, 11).

The core asks the model for JSON in the prompt and tolerant-parses the reply:
strip code fences, json.loads, validate keys, coerce. Providers with native JSON
modes can pass `IDEA_JSON_SCHEMA` through, but correctness never depends on it.
"""
from __future__ import annotations

import json
import re

# Fields every idea carries (DESIGN.md section 6.2). prior_art is blank at
# generation and filled by the novelty check (section 7).
IDEA_FIELDS = [
    "title",
    "intuition",        # plain-language, jargon-free "what real thing is this + why care" (graspable in 10s)
    "bridge",           # the collision: the explicit structural correspondence between the two fused poles
    "mechanism",
    "elaboration",      # the deep walkthrough: construction, load-bearing argument, key assumption, first experiment
    "why_now",
    "math_structure",   # empty if none -- itself a signal
    "tractability",
    "fit_to_program",
    "behavior",         # 2-4 word "domain . method" tag; behavior coord for the diversity archive
    "prior_art",
]

# JSON-schema hint for adapters with native structured-output modes.
IDEA_JSON_SCHEMA = {
    "type": "object",
    "properties": {f: {"type": "string"} for f in IDEA_FIELDS if f != "prior_art"},
    "required": ["title", "mechanism", "why_now", "tractability", "fit_to_program"],
    "additionalProperties": True,
}

_FENCE_RE = re.compile(r"^\s*```(?:json)?\s*|\s*```\s*$", re.IGNORECASE)


class IdeaParseError(ValueError):
    """Raised when a model reply cannot be coerced into an idea dict."""


def strip_code_fences(text: str) -> str:
    """Remove a single surrounding ```json ... ``` (or ``` ... ```) fence."""
    t = text.strip()
    if t.startswith("```"):
        t = _FENCE_RE.sub("", t)
    return t.strip()


def _extract_first_json_object(text: str) -> str | None:
    """Best-effort: pull the first balanced {...} block out of free text."""
    start = text.find("{")
    if start == -1:
        return None
    depth = 0
    for i in range(start, len(text)):
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[start:i + 1]
    return None


def parse_idea(text: str) -> dict:
    """Tolerant-parse a model reply into a normalized idea dict.

    Strategy: strip fences -> json.loads -> fall back to first balanced object.
    Raises IdeaParseError if no JSON object can be recovered. The caller (slots /
    pipeline) is responsible for the one repair-retry described in DESIGN.md
    section 11.
    """
    raw = strip_code_fences(text)
    obj = None
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        candidate = _extract_first_json_object(raw)
        if candidate is not None:
            try:
                obj = json.loads(candidate)
            except (ValueError, TypeError):
                obj = None
    if not isinstance(obj, dict):
        raise IdeaParseError("no JSON object found in model reply")
    return normalize_idea(obj)


def normalize_idea(obj: dict) -> dict:
    """Ensure every IDEA_FIELD key exists as a string; default prior_art unchecked."""
    out: dict[str, str] = {}
    for f in IDEA_FIELDS:
        val = obj.get(f, "")
        if val is None:
            val = ""
        out[f] = val if isinstance(val, str) else json.dumps(val, ensure_ascii=False)
    if not out.get("prior_art"):
        out["prior_art"] = "[unchecked]"
    return out
