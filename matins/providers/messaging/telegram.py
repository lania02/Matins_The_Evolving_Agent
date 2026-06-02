"""Telegram messaging adapter (DESIGN.md sections 9.1, 9.2).

The default and only fully-supported channel in v1. Talks to the Bot API over
HTTPS via httpx. Advisory by nature: callers wrap sends so a messaging failure
never crashes the daily loop.
"""
from __future__ import annotations

import httpx

from ...config import Config
from .base import Reply

# MarkdownV2 reserved characters that must be backslash-escaped (Telegram Bot API).
_MD_V2_RESERVED = r"_*[]()~`>#+-=|{}.!"

_TIMEOUT = 90.0


def escape_markdown_v2(text: str) -> str:
    """Backslash-escape every MarkdownV2 reserved character in `text`."""
    out = []
    for ch in text:
        if ch in _MD_V2_RESERVED:
            out.append("\\")
        out.append(ch)
    return "".join(out)


def _replies_from_updates(updates: list, chat_id: str | None) -> list[Reply]:
    """Build Reply objects from raw getUpdates results, keeping ONLY the owner's messages.

    SECURITY (the bot username is public, so anyone can message it and Telegram returns
    their messages here): without an identity check a stranger's text would be ingested as
    the owner's feedback (taste-log poisoning) or trigger a `dig` -- an unauthenticated,
    remote LLM+web-search call billed to the owner's key. We therefore DROP every update
    whose `message.chat.id` does not equal the configured `chat_id`. When `chat_id` is
    unset there is no owner to trust, so nothing qualifies. Sorted by update_id.
    """
    want = str(chat_id) if chat_id else ""
    replies: list[Reply] = []
    for update in updates:
        msg = update.get("message")
        if not msg or "text" not in msg:
            continue
        if not want or str((msg.get("chat") or {}).get("id", "")) != want:
            continue                                      # not from the owner chat -> ignore
        reply_to = None
        if "reply_to_message" in msg:
            reply_to = str(msg["reply_to_message"]["message_id"])
        replies.append(
            Reply(
                text=msg["text"],
                ts=str(msg["date"]),
                reply_to_message_id=reply_to,
                update_id=str(update["update_id"]),
            )
        )
    replies.sort(key=lambda r: int(r["update_id"]))
    return replies


class TelegramProvider:
    """MessagingProvider over the Telegram Bot API."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.token = cfg.telegram_token()
        self.chat_id = cfg.messaging.telegram.chat_id
        self.base = "https://api.telegram.org/bot" + (self.token or "")

    def send(self, text: str, *, parse_mode: str = "MarkdownV2") -> str:
        if not self.token or not self.chat_id:
            raise RuntimeError(
                "Telegram send requires a bot token and chat_id "
                "(set MATINS_TELEGRAM_TOKEN and messaging.telegram.chat_id)."
            )

        body = escape_markdown_v2(text) if parse_mode == "MarkdownV2" else text
        payload = {"chat_id": self.chat_id, "text": body, "parse_mode": parse_mode}

        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.post(self.base + "/sendMessage", json=payload)
            if resp.status_code // 100 != 2:
                # Markdown can fail on malformed entities; retry once as plain text.
                resp = client.post(
                    self.base + "/sendMessage",
                    json={"chat_id": self.chat_id, "text": text},
                )
            if resp.status_code // 100 != 2:
                raise RuntimeError(
                    f"Telegram sendMessage failed: {resp.status_code} {resp.text[:300]}"
                )
            data = resp.json()

        return str(data["result"]["message_id"])

    def fetch_replies(self, since_update_id: str | None) -> list[Reply]:
        params: dict = {}
        if since_update_id:
            params["offset"] = int(since_update_id) + 1

        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(self.base + "/getUpdates", params=params)
            if resp.status_code // 100 != 2:
                raise RuntimeError(
                    f"Telegram getUpdates failed: {resp.status_code} {resp.text[:300]}"
                )
            data = resp.json()

        # Identity check happens here: only the configured owner chat's messages survive.
        return _replies_from_updates(data.get("result", []), self.chat_id)

    def discover_chat_ids(self) -> list[dict]:
        """List distinct chats that have messaged the bot (helper for setup).

        Returns dicts with keys chat_id, name, text. Useful for finding the
        chat_id to put in config without guessing.
        """
        with httpx.Client(timeout=_TIMEOUT) as client:
            resp = client.get(self.base + "/getUpdates")
            if resp.status_code // 100 != 2:
                raise RuntimeError(
                    f"Telegram getUpdates failed: {resp.status_code} {resp.text[:300]}"
                )
            data = resp.json()

        seen: set[str] = set()
        chats: list[dict] = []
        for update in data.get("result", []):
            msg = update.get("message")
            if not msg:
                continue
            chat = msg.get("chat", {})
            chat_id = str(chat.get("id", ""))
            if not chat_id or chat_id in seen:
                continue
            seen.add(chat_id)
            name = chat.get("first_name") or chat.get("title") or chat.get("username") or ""
            chats.append({"chat_id": chat_id, "name": name, "text": msg.get("text", "")})

        return chats
