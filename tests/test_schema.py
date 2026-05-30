"""Offline tests for tolerant idea-JSON parsing (matins.generate.schema)."""
from __future__ import annotations

import pytest

from matins.generate.schema import (
    IDEA_FIELDS,
    IdeaParseError,
    normalize_idea,
    parse_idea,
)

_BARE = '{"title": "T", "mechanism": "M", "why_now": "now", "tractability": "easy", "fit_to_program": "fits"}'


def test_parse_fenced_json() -> None:
    text = "```json\n" + _BARE + "\n```"
    idea = parse_idea(text)
    assert idea["title"] == "T"
    assert idea["mechanism"] == "M"


def test_parse_bare_json() -> None:
    idea = parse_idea(_BARE)
    assert idea["why_now"] == "now"


def test_parse_json_embedded_in_prose() -> None:
    text = "Here is my idea for today:\n\n" + _BARE + "\n\nHope you like it!"
    idea = parse_idea(text)
    assert idea["fit_to_program"] == "fits"


def test_normalize_fills_all_fields_and_defaults_prior_art() -> None:
    obj = normalize_idea({"title": "only-title"})
    for f in IDEA_FIELDS:
        assert f in obj
        assert isinstance(obj[f], str)
    assert obj["prior_art"] == "[unchecked]"


def test_parse_idea_raises_when_no_json_object() -> None:
    with pytest.raises(IdeaParseError):
        parse_idea("no json here, just prose without braces")
