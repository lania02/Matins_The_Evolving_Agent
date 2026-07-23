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


def _send_long(messaging, text: str, limit: int = 3500) -> None:
    """Send a long message as several chunks (Telegram caps each at 4096 chars)."""
    chunks: list[str] = []
    cur = ""
    for para in text.split("\n"):
        while len(para) > limit:
            if cur:
                chunks.append(cur)
                cur = ""
            chunks.append(para[:limit])
            para = para[limit:]
        if len(cur) + len(para) + 1 > limit:
            if cur:
                chunks.append(cur)
            cur = para
        else:
            cur = (cur + "\n" + para) if cur else para
    if cur:
        chunks.append(cur)
    for c in chunks:
        messaging.send(c)


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

    # Daily news radar, pushed as one extra message after the ideas. Fully advisory: the
    # batch is the product, so a radar failure must never cost the user their ideas.
    news_msg = ""
    if cfg.news.enabled:
        try:
            from .digest.render import render_news
            from .news import collect_news
            news_msg = render_news(collect_news(cfg, store), batch.date)
        except Exception as e:
            print(f"[warning] news radar failed (ideas unaffected): {e}")

    # Always print to stdout so the tool is usable with messaging.channel = none.
    print(header)
    for m in idea_msgs:
        print("\n" + m)
    if news_msg:
        print("\n" + news_msg)

    messaging = get_messaging_provider(cfg)
    if messaging is not None:
        try:
            digest_msg_id = messaging.send(header)
            for m in idea_msgs:
                messaging.send(m)
            if news_msg:
                messaging.send(news_msg)
            if digest_msg_id:
                store.set_batch_digest_msg_id(batch.batch_id, digest_msg_id)
            print(f"\n[sent digest to {cfg.messaging.channel}]")
        except Exception as e:
            # The batch is already generated, ranked, and stored above; a delivery
            # failure should not discard that work or dump a traceback on a cron run.
            print(f"\n[warning] could not send to {cfg.messaging.channel}: {e}")
            print("[hint] batch saved. Set messaging.telegram.chat_id (run "
                  "`matins init-telegram`), or set messaging.channel: none.")
    return 0


# ---- matins collect ------------------------------------------------------
def cmd_collect(args) -> int:
    from .feedback.capture import (
        classify_comment, ingest_must_try, ingest_replies, replies_for_batch,
    )
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

    llm = get_llm_provider(cfg)
    offset = store.get_offset(cfg.messaging.channel)
    replies = messaging.fetch_replies(offset)
    # fetch_replies already dropped anyone but the owner (chat-id check); bind the survivors
    # to THIS batch so a late reply to an older digest is not mis-logged onto today's.
    bound = replies_for_batch(replies, batch.digest_msg_id)
    n = ingest_replies(store, batch, bound, source=cfg.messaging.channel,
                       classify=lambda c: classify_comment(llm, c))
    if replies:
        store.set_offset(cfg.messaging.channel, replies[-1]["update_id"])

    tau = reflect_on_batch(cfg, store, llm, batch)
    print(f"ingested {n} feedback row(s); self-vs-user tau = {tau}")

    favs = ingest_must_try(store, batch, bound)
    if favs:
        from .digest.render import render_favorites_md
        cfg.favorites_path().write_text(
            render_favorites_md(store.list_favorites()), encoding="utf-8"
        )
        print(f"added {len(favs)} idea(s) to favorites -> {cfg.favorites_path()}")

    from .feedback.capture import parse_dig
    ideas_by_idx = {i.idx: i for i in store.ideas_for_batch(batch.batch_id)}
    dig_text = "\n".join(r.get("text", "") for r in bound)
    for idx in parse_dig(dig_text, len(ideas_by_idx)):
        idea = ideas_by_idx.get(idx)
        if idea is None:
            continue
        print(f"deep-diving #{idx} (this may take a moment)...")
        from .generate.deepdive import run_deep_dive, write_brief_md
        try:
            result = run_deep_dive(cfg, store, idea)
        except Exception as e:
            print(f"[warning] deep dive #{idx} failed: {e}")
            continue
        path = write_brief_md(cfg, idea, result["brief"])
        print(f"deep dive #{idx} -> {path} ({len(result['sources'])} sources)")
        if messaging is not None:
            try:
                _send_long(messaging, f"🔬 Deep dive #{idx}: {idea.title}\n\n{result['brief']}")
            except Exception as e:
                print(f"[warning] could not send deep dive #{idx}: {e}")
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

    # Phase 5: when enabled, also try to evolve a NEW taste dimension (propose -> held-out
    # backtest -> park for approval). Skipped while approving an existing proposal.
    if args.approve is None and getattr(cfg.consolidation, "evolve_dimensions", False):
        from .memory.evolve import evolve_dimension
        er = evolve_dimension(cfg, store, llm, messaging)
        print(er.get("message", "evolution step complete."))
    return 0


# ---- matins feedback (offline fallback) ----------------------------------
def cmd_feedback(args) -> int:
    from .feedback.capture import classify_comment, ingest_cli_feedback
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
    llm = get_llm_provider(cfg)
    n = ingest_cli_feedback(store, batch, text, source="cli",
                            classify=lambda c: classify_comment(llm, c))
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


# ---- matins favorites ----------------------------------------------------
def cmd_favorites(args) -> int:
    from .digest.render import render_favorites_md

    cfg = _bootstrap(args)
    store = _open_store(cfg)
    favs = store.list_favorites()
    # Always refresh the human-readable mirror so it never goes stale.
    cfg.favorites_path().write_text(render_favorites_md(favs), encoding="utf-8")
    if not favs:
        print("no favorites yet. Reply 'must try #N' to a digest, then run `matins collect`.")
        return 0
    print(f"{len(favs)} favorite(s)  (full text in {cfg.favorites_path()}):\n")
    for idea, _note, fav_at in favs:
        print(f"  * [{idea.slot}] {idea.title}   (saved {fav_at[:10]})")
    return 0


# ---- matins view ---------------------------------------------------------
def cmd_view(args) -> int:
    from .digest.render import render_overview

    cfg = _bootstrap(args)
    store = _open_store(cfg)
    batches = store.list_batches(limit=args.limit)
    if not batches:
        print("no batches yet; run `matins run` first.")
        return 1
    md = render_overview(store, batches, db_path=str(cfg.db_path()))
    out_path = Path(args.out) if args.out else cfg.view_path()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(md, encoding="utf-8")
    print(f"wrote DB view ({len(batches)} batch(es)) -> {out_path}")
    return 0


# ---- matins dig ----------------------------------------------------------
def cmd_dig(args) -> int:
    from .generate.deepdive import run_deep_dive, write_brief_md

    cfg = _bootstrap(args)
    store = _open_store(cfg)
    batch = store.batch_for_date(args.date) if args.date else store.latest_batch()
    if batch is None:
        print("no batch found; run `matins run` first.")
        return 1
    ideas = {i.idx: i for i in store.ideas_for_batch(batch.batch_id)}
    idea = ideas.get(args.n)
    if idea is None:
        print(f"no idea #{args.n} in batch {batch.date}.")
        return 1
    print(f"deep-diving #{args.n}: {idea.title}")
    print(f"(arxiv + web search, synthesizing with {cfg.dig_model()} — this may take a moment)\n")
    result = run_deep_dive(cfg, store, idea)
    print(result["brief"])
    path = write_brief_md(cfg, idea, result["brief"])
    print(f"\n[saved {path}; {len(result['sources'])} sources]")
    if args.send:
        messaging = get_messaging_provider(cfg)
        if messaging is not None:
            try:
                _send_long(messaging, f"🔬 Deep dive #{args.n}: {idea.title}\n\n{result['brief']}")
                print("[sent to telegram]")
            except Exception as e:
                print(f"[warning] could not send: {e}")
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

    pfav = sub.add_parser("favorites", help="list curated 'must try' ideas (refreshes favorites.md)")
    pfav.set_defaults(func=cmd_favorites)

    pview = sub.add_parser("view", help="render the DB to a markdown overview (-> state_dir/view.md)")
    pview.add_argument("--limit", type=int, default=None, help="only the N most recent batches")
    pview.add_argument("--out", default=None, help="output path (default: <state_dir>/view.md)")
    pview.set_defaults(func=cmd_view)

    pd = sub.add_parser("dig", help="deep-dive briefing for one idea (arxiv + web, cited)")
    pd.add_argument("n", type=int, help="idea number to deep-dive")
    pd.add_argument("--date", default=None, help="batch date (default: latest)")
    pd.add_argument("--send", action="store_true", help="also push the brief to the channel")
    pd.set_defaults(func=cmd_dig)

    return p


def _force_utf8_stdout() -> None:
    """Make stdout/stderr UTF-8 so bilingual (Chinese) and math (e.g. rho) output
    prints on a Windows cp1252 console instead of raising UnicodeEncodeError."""
    for stream in (sys.stdout, sys.stderr):
        reconfigure = getattr(stream, "reconfigure", None)
        if reconfigure is not None:
            try:
                reconfigure(encoding="utf-8")
            except Exception:
                pass


def main(argv: list[str] | None = None) -> int:
    _force_utf8_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
