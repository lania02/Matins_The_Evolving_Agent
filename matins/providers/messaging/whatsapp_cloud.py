"""WhatsApp adapter via Meta's official Cloud API (DESIGN.md sections 9.1, 9.6).

Experimental, opt-in. The official route uses Meta's WhatsApp Business Cloud
API: a dedicated business number, an access token, and an inbound webhook to
receive replies. It is unimplemented in v1.
"""
from __future__ import annotations

from ...config import Config
from .base import Reply

_MSG = (
    "WhatsApp (Cloud) channel is not implemented in v1. It requires the Meta "
    "WhatsApp Business Cloud API (dedicated business number, access token) plus "
    "an inbound webhook to receive replies (DESIGN.md sections 9.1 / 9.6). "
    "Use channel=telegram."
)


class WhatsAppCloudProvider:
    """MessagingProvider stub for Meta's official WhatsApp Business Cloud API."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def send(self, text: str, *, parse_mode: str = "MarkdownV2") -> str:
        raise NotImplementedError(_MSG)

    def fetch_replies(self, since_update_id: str | None) -> list[Reply]:
        raise NotImplementedError(_MSG)
