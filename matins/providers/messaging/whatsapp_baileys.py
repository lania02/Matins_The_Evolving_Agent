"""WhatsApp adapter via the unofficial Baileys bridge (DESIGN.md section 9.6).

Experimental, opt-in. Baileys is a Node/TS library that drives the WhatsApp Web
protocol, so this adapter would talk to a small Node side-process over a local
socket. It is unimplemented in v1 and carries the ToS / ban / fragility caveats
described in DESIGN.md section 9.1 -- use a dedicated number.
"""
from __future__ import annotations

from ...config import Config
from .base import Reply

_MSG = (
    "WhatsApp (Baileys) channel is not implemented in v1. It requires a Node "
    "Baileys bridge exposing send/fetch over a local socket and is opt-in with "
    "ToS / ban / fragility caveats (DESIGN.md section 9.6). Use channel=telegram."
)


class WhatsAppBaileysProvider:
    """MessagingProvider stub for the unofficial Baileys route."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg

    def send(self, text: str, *, parse_mode: str = "MarkdownV2") -> str:
        raise NotImplementedError(_MSG)

    def fetch_replies(self, since_update_id: str | None) -> list[Reply]:
        raise NotImplementedError(_MSG)
