"""Matins command-line entrypoints (DESIGN.md sections 9.5, 12, 14).

Commands (all idempotent, designed to be triggered by cron / Task Scheduler):
  matins run           generate -> store -> push digest
  matins collect       fetch replies -> store feedback -> reflect (tau, hypotheses)
  matins consolidate   propose skill diff (weekly) / approve a pending proposal
  matins feedback      offline fallback: ingest a ranking from the CLI
  matins init-telegram print chat ids so you can fill messaging.telegram.chat_id

No always-on daemon: each command does one pass and exits.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from . import __version__
from .config import Config, load_config
from .providers.base import get_llm_provider, get_search_provider
from .providers.messaging.base import get_messaging_provider
from .store.db import Store


def _bootstrap(args) -> Config:
    cfg = load_config(args.config)
    return cfg


def _open_store(cfg: Config) -> Store:
    return Store(cfg.db_path())


# ---- matins run ----------------------------------------------------------
def cmd_run(args) -> int:
    from .digest.render import render_digest
    from .generate.pipeline import run_batch

    cfg = _bootstrap(args)
    store = _open_store(cfg)
    llm = get_llm_provider(cfg)
    search = get_search_provider(cfg)

    batch, ideas = run_batch(cfg, store, llm, search, date=args.date)
    header, idea_msgs = render_digest(batch, ideas, cfg.generation.output_language)

    # Always print to stdout so the tool is usable with messaging.channel = none.
    print(header)
    for m in idea_msgs:
        print("\n" + m)

    messaging = get_messaging_provider(cfg)
    if messaging is not None:
        digest_msg_id = messaging.send(header)
        for m in idea_msgs:
            messaging.send(m)
        if digest_msg_id:
            store.set_batch_digest_msg_id(batch.batch_id, digest_msg_id)
        print(f"\n[sent digest to {cfg.messaging.channel}]")
    return 0


# ---- matins collect ------------------------------------------------------
def cmd_collect(args) -> int:
    from .feedback.capture import ingest_replies
    from .feedback.diverge import reflect_on_batch

    cfg = _bootstrap(args)
    store = _open_store(cfg)
    messaging = get_messaging_provider(cfg)
    if messaging is None:
        print("messaging.channel is 'none'; use `matins feedback` for offline input.")
        return 1

    batch = store.latest_batch()
    if batch is None:
        print("no batch yet; run `matins run` first.")
        return 1

    offset = store.get_offset(cfg.messaging.channel)
    replies = messaging.fetch_replies(offset)
    n = ingest_replies(store, batch, replies, source=cfg.messaging.channel)
    if replies:
        store.set_offset(cfg.messaging.channel, replies[-1]["update_id"])

    llm = get_llm_provider(cfg)
    tau = reflect_on_batch(cfg, store, llm, batch)
    print(f"ingested {n} feedback row(s); self-vs-user tau = {tau}")
    return 0


# ---- matins consolidate --------------------------------------------------
def cmd_consolidate(args) -> int:
    from .memory.consolidate import run_consolidation

    cfg = _bootstrap(args)
    store = _open_store(cfg)
    llm = get_llm_provider(cfg)
    messaging = get_messaging_provider(cfg)
    result = run_consolidation(cfg, store, llm, messaging, approve_version=args.approve)
    print(result.get("message", "consolidation step complete."))
    return 0


# ---- matins feedback (offline fallback) ----------------------------------
def cmd_feedback(args) -> int:
    from .feedback.capture import ingest_cli_feedback
    from .feedback.diverge import reflect_on_batch

    cfg = _bootstrap(args)
    store = _open_store(cfg)
    batch = store.latest_batch()
    if batch is None:
        print("no batch yet; run `matins run` first.")
        return 1

    text = args.ranking
    if not text:
        print("paste ranking (e.g. '3>1>4>2' then optional '#n comment' lines), then Ctrl-D:")
        text = sys.stdin.read()
    n = ingest_cli_feedback(store, batch, text, source="cli")
    llm = get_llm_provider(cfg)
    tau = reflect_on_batch(cfg, store, llm, batch)
    print(f"ingested {n} feedback row(s); self-vs-user tau = {tau}")
    return 0


# ---- matins init-telegram ------------------------------------------------
def cmd_init_telegram(args) -> int:
    cfg = _bootstrap(args)
    if not cfg.telegram_token():
        print(f"set your bot token in ${cfg.messaging.telegram.bot_token_env} first.")
        return 1
    from .providers.messaging.telegram import TelegramProvider

    tp = TelegramProvider(cfg)
    chats = tp.discover_chat_ids()
    if not chats:
        print("no chats found. Message your bot once, then re-run `matins init-telegram`.")
        return 1
    print("Found chats (put the chat_id into config.yaml -> messaging.telegram.chat_id):")
    for c in chats:
        print(f"  chat_id={c.get('chat_id')}  name={c.get('name')}  last={c.get('text')!r}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="matins", description="A daily human-AI brainstorm loop.")
    p.add_argument("--version", action="version", version=f"matins {__version__}")
    p.add_argument("--config", default="config.yaml", help="path to config.yaml")
    sub = p.add_subparsers(dest="command", required=True)

    pr = sub.add_parser("run", help="generate today's batch and push the digest")
    pr.add_argument("--date", default=None, help="override the batch date (YYYY-MM-DD)")
    pr.set_defaults(func=cmd_run)

    pc = sub.add_parser("collect", help="fetch replies, store feedback, reflect")
    pc.set_defaults(func=cmd_collect)

    pco = sub.add_parser("consolidate", help="propose or approve a taste-skill update")
    pco.add_argument("--approve", type=int, default=None,
                     help="approve a pending skill version by number")
    pco.set_defaults(func=cmd_consolidate)

    pf = sub.add_parser("feedback", help="offline fallback: ingest a ranking from the CLI")
    pf.add_argument("ranking", nargs="?", default=None, help="e.g. '3>1>4>2'")
    pf.set_defaults(func=cmd_feedback)

    pi = sub.add_parser("init-telegram", help="discover your Telegram chat_id")
    pi.set_defaults(func=cmd_init_telegram)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
