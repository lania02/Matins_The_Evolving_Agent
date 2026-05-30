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
