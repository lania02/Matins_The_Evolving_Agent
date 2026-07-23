"""Daily tech-news radar: what actually broke today, pushed alongside the idea digest.

Tiers 1+2 of the radar design: a KEYWORD-FREE firehose (attention-sorted lists, so genuinely
unprecedented news is not missed by a stale keyword list -- you cannot keyword-search for an
event whose vocabulary does not exist yet) plus DETERMINISTIC anomaly scoring (velocity +
cross-source corroboration). No LLM call and no extra paid search: the subreddit pull reuses
the same 'trending' provider the generation feed already builds, so a run gains ~2 free HTTP
calls.

Deliberately NOT here yet: LLM significance judgement and targeted confirmation. Those only
earn their cost once the raw anomaly stream proves it surfaces things worth waking up for.
Until then this reports "what is hot", never "this is big" -- an honest label, since a news
feed without a skepticism layer is a hype amplifier.
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone

import httpx

logger = logging.getLogger("matins.news")

_TIMEOUT = 30.0


@dataclass
class NewsItem:
    title: str
    url: str
    source: str            # "HN" | "GitHub"
    metric: int            # points (HN) or stars (GitHub)
    age_hours: float
    velocity: float        # metric per hour -- the anomaly signal, not raw volume
    corroborated: bool = False   # also being discussed in a watched subreddit
    score: float = 0.0

    def key(self) -> str:
        return (self.url or self.title).strip()


def _tokens(text: str) -> set[str]:
    """Words for crude title matching (CJK chars + English words >2 chars)."""
    s = (text or "").lower()
    return set(re.findall(r"[一-鿿]", s)) | {
        w for w in re.findall(r"[a-z0-9]+", s) if len(w) > 2
    }


def _overlaps(a: str, b: str, *, threshold: float = 0.35) -> bool:
    ta, tb = _tokens(a), _tokens(b)
    if not ta or not tb:
        return False
    return len(ta & tb) / min(len(ta), len(tb)) >= threshold


def fetch_hn_top(*, window_hours: int, min_points: int, k: int = 30) -> list[NewsItem]:
    """Top HN stories from the last `window_hours` (keyword-free: the whole front of the site)."""
    cutoff = int((datetime.now(timezone.utc) - timedelta(hours=window_hours)).timestamp())
    try:
        resp = httpx.get(
            "https://hn.algolia.com/api/v1/search",
            params={"tags": "story",
                    "numericFilters": f"created_at_i>{cutoff},points>{min_points}",
                    "hitsPerPage": k},
            timeout=_TIMEOUT, follow_redirects=True,
        )
        if not (200 <= resp.status_code < 300):
            logger.warning("hn radar HTTP %s", resp.status_code)
            return []
        now = datetime.now(timezone.utc).timestamp()
        out: list[NewsItem] = []
        for h in (resp.json().get("hits") or []):
            title = (h.get("title") or "").strip()
            if not title:
                continue
            created = h.get("created_at_i") or now
            age = max((now - float(created)) / 3600.0, 1.0)
            points = int(h.get("points") or 0)
            url = (h.get("url") or "").strip() or \
                f"https://news.ycombinator.com/item?id={h.get('objectID', '')}"
            out.append(NewsItem(title=title, url=url, source="HN", metric=points,
                                age_hours=age, velocity=points / age))
        return out
    except Exception as exc:
        logger.warning("hn radar failed: %s", exc)
        return []


def fetch_github_new(*, days: int, min_stars: int, k: int = 10) -> list[NewsItem]:
    """Recently-created repos that already have many stars -- a launch/breakout signal."""
    since = (datetime.now(timezone.utc) - timedelta(days=days)).strftime("%Y-%m-%d")
    try:
        resp = httpx.get(
            "https://api.github.com/search/repositories",
            params={"q": f"stars:>{min_stars} created:>{since}",
                    "sort": "stars", "order": "desc", "per_page": k},
            headers={"Accept": "application/vnd.github+json"},
            timeout=_TIMEOUT, follow_redirects=True,
        )
        if not (200 <= resp.status_code < 300):
            logger.warning("github radar HTTP %s", resp.status_code)
            return []
        now = datetime.now(timezone.utc)
        out: list[NewsItem] = []
        for r in (resp.json().get("items") or []):
            name = (r.get("full_name") or "").strip()
            if not name:
                continue
            desc = (r.get("description") or "").strip()
            stars = int(r.get("stargazers_count") or 0)
            try:
                created = datetime.fromisoformat(
                    (r.get("created_at") or "").replace("Z", "+00:00"))
                age = max((now - created).total_seconds() / 3600.0, 1.0)
            except ValueError:
                age = float(days * 24)
            title = f"{name} — {desc}" if desc else name
            out.append(NewsItem(title=title, url=(r.get("html_url") or "").strip(),
                                source="GitHub", metric=stars, age_hours=age,
                                velocity=stars / age))
        return out
    except Exception as exc:
        logger.warning("github radar failed: %s", exc)
        return []


def _subreddit_titles(cfg) -> list[str]:
    """Watched-subreddit thread titles, reusing the 'trending' provider the feed already
    builds (so this costs no extra SerpAPI call beyond what generation already spends)."""
    try:
        from .providers.search_web import get_retrieval_searcher
        provider = get_retrieval_searcher("trending", cfg)
        if provider is None:
            return []
        return [str(h.get("title", "")) for h in (provider.search("", k=16) or [])
                if str(h.get("title", "")).startswith("[r/")]
    except Exception as exc:
        logger.warning("radar corroboration source failed: %s", exc)
        return []


def score_and_rank(candidates: list[NewsItem], sub_titles: list[str]) -> list[NewsItem]:
    """Rank by velocity AND magnitude, normalized per source, boosted by corroboration.

    Three deliberate choices, each fixing a way a naive ranking misleads:

    * Per-SOURCE normalization: HN points/hour and GitHub stars/hour are different units
      entirely, so comparing them raw is meaningless -- each source competes with itself.
    * Velocity AND magnitude: velocity alone catches what is breaking but buries a landmark
      that has been climbing all day under a fast-spiking news-cycle piece (observed live: a
      1011-point Terence Tao thread ranked below a 176-point news item). Magnitude alone does
      the reverse. Half weight each.
    * Corroboration -- the same story ALSO surfacing in a watched subreddit -- is the cheapest
      hard evidence that something is genuinely breaking rather than one site's quirk.
    """
    if not candidates:
        return []
    peaks: dict[str, tuple[float, float]] = {}
    for c in candidates:
        peak_vel, peak_mag = peaks.get(c.source, (0.0, 0.0))
        peaks[c.source] = (max(peak_vel, c.velocity), max(peak_mag, float(c.metric)))
    for c in candidates:
        peak_vel, peak_mag = peaks[c.source]
        vel = c.velocity / peak_vel if peak_vel else 0.0
        mag = float(c.metric) / peak_mag if peak_mag else 0.0
        c.corroborated = any(_overlaps(c.title, t) for t in sub_titles)
        c.score = round(0.5 * vel + 0.5 * mag + (0.5 if c.corroborated else 0.0), 3)
    return sorted(candidates, key=lambda c: (-c.score, -c.velocity))


def collect_news(cfg, store) -> list[NewsItem]:
    """Fetch, score, de-duplicate against recent days, and return the top `news.count`.

    Fail-open: any source failure degrades the list rather than the run. Everything returned
    is recorded so tomorrow does not re-push the same story.
    """
    news = cfg.news
    candidates = fetch_hn_top(window_hours=news.window_hours, min_points=news.min_points)
    candidates += fetch_github_new(days=news.github_days, min_stars=news.github_min_stars)
    if not candidates:
        return []

    ranked = score_and_rank(candidates, _subreddit_titles(cfg))

    seen_urls, seen_titles = store.recent_news(news.dedup_days)
    picked: list[NewsItem] = []
    for c in ranked:
        if c.key() in seen_urls:
            continue
        # cluster: the same story reported under a different URL is not new either
        if any(_overlaps(c.title, t) for t in seen_titles):
            continue
        if any(_overlaps(c.title, p.title) for p in picked):
            continue
        picked.append(c)
        seen_titles.append(c.title)
        if len(picked) >= news.count:
            break

    for c in picked:
        try:
            store.insert_news_event(c.key(), c.title, c.source, c.score)
        except Exception as exc:                       # logging must never sink the run
            logger.warning("could not record news event: %s", exc)
    return picked
