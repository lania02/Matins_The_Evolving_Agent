"""Channel-agnostic messaging abstraction + factory (DESIGN.md section 9.2).

Telegram is the default and only fully-supported channel in v1. WhatsApp adapters
exist behind the same interface but are opt-in and carry ToS / fragility caveats
(DESIGN.md section 9.1, 9.6).
"""
from __future__ import annotations

from typing import Protocol, TypedDict, runtime_checkable

from ...config import Config


class Reply(TypedDict):
    text: str
    ts: str
    reply_to_message_id: str | None
    update_id: str


@runtime_checkable
class MessagingProvider(Protocol):
    def send(self, text: str, *, parse_mode: str = "MarkdownV2") -> str:
        """Send a message; return the channel's message id."""
        ...

    def fetch_replies(self, since_update_id: str | None) -> list[Reply]:
        """Return replies newer than `since_update_id` (None = from the start)."""
        ...


def get_messaging_provider(cfg: Config) -> MessagingProvider | None:
    """Instantiate the configured messaging adapter, or None when channel='none'."""
    channel = cfg.messaging.channel
    if channel in ("none", "", None):
        return None
    if channel == "telegram":
        from .telegram import TelegramProvider
        return TelegramProvider(cfg)
    if channel == "whatsapp_baileys":
        from .whatsapp_baileys import WhatsAppBaileysProvider
        return WhatsAppBaileysProvider(cfg)
    if channel == "whatsapp_cloud":
        from .whatsapp_cloud import WhatsAppCloudProvider
        return WhatsAppCloudProvider(cfg)
    raise ValueError(f"unknown messaging channel: {channel!r}")
