"""Offline tests for the Telegram identity check (no network / token needed).

The bot username is public, so getUpdates returns ANYONE's messages. _replies_from_updates
is the security boundary: only the configured owner chat's messages may become feedback.
"""
from __future__ import annotations

from matins.providers.messaging.telegram import (
    _bold_key_terms,
    _replies_from_updates,
    escape_markdown_v2,
)

OWNER = "7302099055"


def _update(uid, chat_id, text, *, reply_to=None):
    msg = {"chat": {"id": chat_id}, "text": text, "date": 1000 + uid}
    if reply_to is not None:
        msg["reply_to_message"] = {"message_id": reply_to}
    return {"update_id": uid, "message": msg}


def test_keeps_only_owner_chat() -> None:
    updates = [
        _update(1, int(OWNER), "3>1>4>2"),          # owner (int id) -> kept
        _update(2, 999999, "dig #1 #2 #3 #4"),      # stranger -> dropped (DoS / poisoning)
        _update(3, OWNER, "#1 nice"),               # owner (str id) -> kept
    ]
    out = _replies_from_updates(updates, OWNER)
    assert [r["text"] for r in out] == ["3>1>4>2", "#1 nice"]
    assert all("dig" not in r["text"] for r in out)   # stranger's dig never reaches collect


def test_extracts_reply_to_and_sorts() -> None:
    updates = [
        _update(5, int(OWNER), "later", reply_to=42),
        _update(4, int(OWNER), "earlier"),
    ]
    out = _replies_from_updates(updates, OWNER)
    assert [r["update_id"] for r in out] == ["4", "5"]            # sorted by update_id
    assert out[0]["reply_to_message_id"] is None
    assert out[1]["reply_to_message_id"] == "42"                  # reply target captured


def test_skips_non_message_and_textless_updates() -> None:
    updates = [
        {"update_id": 1},                                              # no message at all
        {"update_id": 2, "message": {"chat": {"id": int(OWNER)}, "date": 1}},  # no text
        _update(3, int(OWNER), "ok"),
    ]
    out = _replies_from_updates(updates, OWNER)
    assert [r["text"] for r in out] == ["ok"]


def test_unset_chat_id_trusts_nobody() -> None:
    updates = [_update(1, 123, "hello"), _update(2, 456, "world")]
    assert _replies_from_updates(updates, "") == []
    assert _replies_from_updates(updates, None) == []


def test_bold_key_terms_bolds_title_line_and_known_labels() -> None:
    raw = (
        "#2 [adjacent-stretch] Wikipedia编辑战中的极化前兆\n"
        "Intuition: some plain pitch (with parens).\n"
        "Bridge: the collision, with a formula a+b=1.\n"
        "Checks: useful 0.60 (conf 0.55)"
    )
    out = _bold_key_terms(escape_markdown_v2(raw))
    lines = out.split("\n")
    # first line (title) is wholly bold, still contains its escaped content
    assert lines[0].startswith("*") and lines[0].endswith("*")
    assert "adjacent\\-stretch" in lines[0]
    # known field labels become bold; their VALUES stay untouched/escaped, not re-bolded
    assert lines[1].startswith("*Intuition*: ")
    assert lines[2].startswith("*Bridge*: ")
    assert lines[3].startswith("*Checks*: ")
    # only the label's own asterisks are unescaped -- the rest of the message has none
    body_after_labels = "".join(l.split(": ", 1)[1] if ": " in l else l for l in lines[1:])
    assert "*" not in body_after_labels


def test_bold_key_terms_does_not_touch_non_label_lines() -> None:
    raw = "Title line\nSome LLM sentence that happens to start with Bridgework: not a label"
    out = _bold_key_terms(escape_markdown_v2(raw))
    assert "*Bridgework*" not in out                # not an exact label match -> untouched


def test_bold_key_terms_handles_empty_and_single_line() -> None:
    assert _bold_key_terms("") == ""
    assert _bold_key_terms("just one line") == "*just one line*"


def test_send_wraps_body_in_bold_key_terms(monkeypatch) -> None:
    from matins.config import Config
    from matins.providers.messaging.telegram import TelegramProvider

    captured = {}

    class FakeClient:
        def __init__(self, *a, **k):
            pass

        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def post(self, url, json):
            captured["json"] = json

            class R:
                status_code = 200

                def json(self):
                    return {"result": {"message_id": 1}}
            return R()

    monkeypatch.setattr("matins.providers.messaging.telegram.httpx.Client", FakeClient)
    cfg = Config()
    cfg.messaging.telegram.chat_id = "123"
    import os
    os.environ["MATINS_TELEGRAM_TOKEN"] = "tok"
    provider = TelegramProvider(cfg)
    provider.send("#1 [high-fit] Title\nIntuition: plain pitch")
    assert captured["json"]["text"].startswith("*")
    assert "*Intuition*: " in captured["json"]["text"]
