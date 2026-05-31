"""Search providers for the novelty / prior-art step (DESIGN.md section 7).

Two best-effort adapters implement the SearchProvider protocol from base.py:

  * ArxivSearchProvider  -- queries the arXiv Atom API (stable, no key needed).
  * WebSearchProvider    -- scrapes DuckDuckGo HTML as a zero-config default.

Both are advisory: the daily loop must never crash because a search failed, so
the web adapter swallows all exceptions and returns []. A real, paid search API
can be slotted in here behind the same `search(query, *, k=5)` signature.
"""
from __future__ import annotations

import re
import time
import xml.etree.ElementTree as ET
from html import unescape
from html.parser import HTMLParser

import httpx

from ..config import Config
from .base import SearchProvider

_ATOM = "{http://www.w3.org/2005/Atom}"
_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"
)


def get_search_provider(cfg: Config) -> SearchProvider:
    """Return the concrete search adapter named by cfg.novelty.search_provider.

    Anything other than 'arxiv' falls back to the web adapter (the default).
    """
    if cfg.novelty.search_provider == "arxiv":
        return ArxivSearchProvider()
    return WebSearchProvider()


def _collapse(text: str, *, limit: int = 300) -> str:
    """Collapse whitespace and trim to roughly `limit` characters."""
    out = re.sub(r"\s+", " ", text or "").strip()
    return out[:limit]


class ArxivSearchProvider:
    """Search arXiv via its public Atom API."""

    def search(self, query: str, *, k: int = 5) -> list[dict]:
        # AND the terms (all:t1 AND all:t2 ...) so every keyword must appear.
        # arXiv treats a bare space-separated all: query as OR, which lets a paper
        # matching just one generic word (e.g. "spectral") dominate; ANDing instead
        # surfaces a genuinely related paper or honestly returns nothing.
        terms = query.split()
        search_query = " AND ".join(f"all:{t}" for t in terms) if terms else f"all:{query}"
        params = {
            "search_query": search_query,
            "start": 0,
            "max_results": k,
            "sortBy": "relevance",
            "sortOrder": "descending",
        }
        # arXiv asks for ~1 request / 3s; back off and retry on 429 rate limiting.
        resp = None
        for attempt in range(3):
            resp = httpx.get(
                "https://export.arxiv.org/api/query",
                params=params,
                timeout=60,
                follow_redirects=True,
            )
            if resp.status_code != 429:
                break
            time.sleep(3 * (attempt + 1))
        if resp is None or not (200 <= resp.status_code < 300):
            code = resp.status_code if resp is not None else "no-response"
            body = resp.text[:200] if resp is not None else ""
            raise RuntimeError(f"arxiv search HTTP {code}: {body}")
        root = ET.fromstring(resp.text)
        results: list[dict] = []
        for entry in root.findall(f"{_ATOM}entry"):
            title_el = entry.find(f"{_ATOM}title")
            title = _collapse(title_el.text or "", limit=300) if title_el is not None else ""

            id_el = entry.find(f"{_ATOM}id")
            url = (id_el.text or "").strip() if id_el is not None else ""
            if not url:
                link = entry.find(f"{_ATOM}link")
                if link is not None:
                    url = (link.get("href") or "").strip()

            summary_el = entry.find(f"{_ATOM}summary")
            snippet = _collapse(summary_el.text or "") if summary_el is not None else ""

            results.append({"title": title, "url": url, "snippet": snippet})
        return results


class _DDGParser(HTMLParser):
    """Tolerant parser for DuckDuckGo HTML result anchors."""

    def __init__(self) -> None:
        super().__init__()
        self.results: list[dict] = []
        self._in_result_link = False
        self._current_url = ""
        self._current_title_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag != "a":
            return
        attr = dict(attrs)
        cls = attr.get("class") or ""
        if "result__a" in cls:
            self._in_result_link = True
            self._current_url = attr.get("href") or ""
            self._current_title_parts = []

    def handle_data(self, data: str) -> None:
        if self._in_result_link:
            self._current_title_parts.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "a" and self._in_result_link:
            self._in_result_link = False
            title = _collapse("".join(self._current_title_parts), limit=300)
            url = self._normalize_url(self._current_url)
            if title and url:
                self.results.append({"title": title, "url": url, "snippet": ""})

    @staticmethod
    def _normalize_url(url: str) -> str:
        url = unescape(url or "").strip()
        # DuckDuckGo wraps targets in a redirect: //duckduckgo.com/l/?uddg=<enc>
        m = re.search(r"[?&]uddg=([^&]+)", url)
        if m:
            from urllib.parse import unquote

            return unquote(m.group(1))
        return url


class WebSearchProvider:
    """Best-effort web search by scraping DuckDuckGo HTML.

    This is a zero-config default; a real search API can be slotted in here
    behind the same SearchProvider.search signature. On ANY exception this
    returns [] so the advisory novelty step never breaks the daily loop.
    """

    def search(self, query: str, *, k: int = 5) -> list[dict]:
        try:
            resp = httpx.post(
                "https://html.duckduckgo.com/html/",
                data={"q": query},
                headers={"User-Agent": _USER_AGENT},
                timeout=60,
                follow_redirects=True,
            )
            if not (200 <= resp.status_code < 300):
                return []
            parser = _DDGParser()
            parser.feed(resp.text)
            return parser.results[:k]
        except Exception:
            return []


class TavilySearchProvider:
    """Web search via the Tavily API (purpose-built for LLM grounding).

    Returns clean content snippets ready to cite. On any error returns [] so the
    deep dive degrades gracefully to arXiv-only.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key

    def search(self, query: str, *, k: int = 5) -> list[dict]:
        if not self.api_key:
            return []
        try:
            resp = httpx.post(
                "https://api.tavily.com/search",
                headers={"Authorization": f"Bearer {self.api_key}"},
                json={
                    "api_key": self.api_key,  # also accepted in body for older keys
                    "query": query,
                    "max_results": k,
                    "search_depth": "advanced",
                    "include_answer": False,
                },
                timeout=60,
            )
            if not (200 <= resp.status_code < 300):
                return []
            results = resp.json().get("results", []) or []
            out = []
            for r in results:
                out.append({
                    "title": (r.get("title") or "").strip(),
                    "url": (r.get("url") or "").strip(),
                    "snippet": _collapse(r.get("content") or "", limit=500),
                })
            return out
        except Exception:
            return []


def _openalex_abstract(inv: dict | None) -> str:
    """Reconstruct an abstract from OpenAlex's inverted index ({word: [positions]})."""
    if not inv:
        return ""
    pos: dict[int, str] = {}
    for word, idxs in inv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


class OpenAlexSearchProvider:
    """Scholarly search via OpenAlex (open corpus, cross-domain, citation-aware).

    Relevance-ranked full-text search; abstracts are reconstructed from the inverted
    index. The API key (optional) just raises the rate limit. Best-effort: any error
    returns [] so the daily loop never breaks on a search hiccup.
    """

    def __init__(self, api_key: str | None = None, mailto: str | None = None):
        self.api_key = api_key
        self.mailto = mailto

    def search(self, query: str, *, k: int = 5) -> list[dict]:
        try:
            params: dict = {"search": query, "per_page": max(1, k)}
            if self.api_key:
                params["api_key"] = self.api_key
            if self.mailto:
                params["mailto"] = self.mailto
            resp = httpx.get("https://api.openalex.org/works", params=params,
                             timeout=60, follow_redirects=True)
            if not (200 <= resp.status_code < 300):
                return []
            out: list[dict] = []
            for w in (resp.json().get("results") or []):
                title = _collapse(w.get("display_name") or "", limit=300)
                url = (w.get("doi") or "").strip() or (w.get("id") or "").strip()
                snippet = _collapse(_openalex_abstract(w.get("abstract_inverted_index")), limit=400)
                if title:
                    out.append({"title": title, "url": url, "snippet": snippet})
            return out
        except Exception:
            return []


class HackerNewsSearchProvider:
    """Community-signal search via the Hacker News Algolia API (free, no key).

    Surfaces what practitioners are currently discussing -- a timeliness / applied-angle
    signal, not a scholarly one. Best-effort: any error returns [].
    """

    def search(self, query: str, *, k: int = 5) -> list[dict]:
        try:
            resp = httpx.get("https://hn.algolia.com/api/v1/search",
                             params={"query": query, "tags": "story", "hitsPerPage": max(1, k)},
                             timeout=60, follow_redirects=True)
            if not (200 <= resp.status_code < 300):
                return []
            out: list[dict] = []
            for h in (resp.json().get("hits") or []):
                title = _collapse(h.get("title") or h.get("story_title") or "", limit=300)
                url = (h.get("url") or "").strip()
                if not url and h.get("objectID"):
                    url = f"https://news.ycombinator.com/item?id={h['objectID']}"
                if title:
                    out.append({"title": title, "url": url, "snippet": ""})
            return out
        except Exception:
            return []


def get_web_searcher(cfg) -> SearchProvider | None:
    """Web searcher for the deep dive (Tavily), or None when unconfigured/keyless."""
    if cfg.deep_dive.web_search == "tavily":
        key = cfg.deep_dive_web_key()
        if key:
            return TavilySearchProvider(key)
    return None


def get_retrieval_searcher(name: str, cfg) -> SearchProvider | None:
    """Resolve a named source for the daily generation feed, or None if unavailable.

    'tavily' needs a key; the others are keyless (openalex's key is optional and only
    lifts the rate limit). Unknown names return None and are simply skipped.
    """
    name = (name or "").lower()
    if name == "arxiv":
        return ArxivSearchProvider()
    if name == "openalex":
        return OpenAlexSearchProvider(api_key=cfg.openalex_api_key())
    if name == "tavily":
        key = cfg.deep_dive_web_key()
        return TavilySearchProvider(key) if key else None
    if name == "hackernews":
        return HackerNewsSearchProvider()
    if name == "web":
        return WebSearchProvider()
    return None
