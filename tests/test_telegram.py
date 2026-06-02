"""Offline tests for the Telegram identity check (no network / token needed).

The bot username is public, so getUpdates returns ANYONE's messages. _replies_from_updates
is the security boundary: only the configured owner chat's messages may become feedback.
"""
from __future__ import annotations

from matins.providers.messaging.telegram import _replies_from_updates

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
