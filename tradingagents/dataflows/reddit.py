"""Reddit search fetcher for ticker-specific discussion posts.

Uses Reddit's public RSS search feed only. Requests are cached per ticker and
subreddit for the lifetime of the Python process so multiple agents analyzing
the same asset do not repeatedly hit Reddit. Requests are paced globally and a
429 is treated as a soft failure after a single Retry-After-aware wait.
"""

from __future__ import annotations

import html
import http.client
import logging
import re
import time
import xml.etree.ElementTree as ET
from collections.abc import Iterable
from datetime import datetime
from functools import lru_cache
from urllib.error import HTTPError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from .symbol_utils import crypto_base

logger = logging.getLogger(__name__)

_RSS = "https://www.reddit.com/r/{sub}/search.rss?{qs}"
_UA = "tradingagents/0.2 (+https://github.com/TauricResearch/TradingAgents)"
_ATOM_NS = {"atom": "http://www.w3.org/2005/Atom"}
DEFAULT_SUBREDDITS = ("wallstreetbets", "stocks", "investing")
_MIN_REQUEST_INTERVAL = 2.0
_last_request_at = 0.0


def _search_qs(ticker: str, limit: int) -> str:
    return urlencode({"q": ticker, "restrict_sr": "on", "sort": "new", "t": "week", "limit": limit})


def _pace_request() -> None:
    global _last_request_at
    wait = _MIN_REQUEST_INTERVAL - (time.monotonic() - _last_request_at)
    if wait > 0:
        time.sleep(wait)
    _last_request_at = time.monotonic()


def _iso_to_timestamp(iso_str: str | None) -> float | None:
    if not iso_str:
        return None
    try:
        normalized = iso_str[:-1] + "+00:00" if iso_str.endswith("Z") else iso_str
        return datetime.fromisoformat(normalized).timestamp()
    except (ValueError, TypeError):
        return None


def _strip_html(content: str) -> str:
    if not content:
        return ""
    if "<!-- SC_OFF -->" in content and "<!-- SC_ON -->" in content:
        content = content.split("<!-- SC_OFF -->")[1].split("<!-- SC_ON -->")[0]
    return " ".join(html.unescape(re.sub(r"<[^>]+>", " ", content)).split())


def _retry_after_seconds(exc: HTTPError) -> float:
    try:
        val = exc.headers.get("Retry-After") if getattr(exc, "headers", None) else None
        return min(float(val), 30.0) if val else 10.0
    except (ValueError, TypeError, AttributeError):
        return 10.0


@lru_cache(maxsize=128)
def _fetch_subreddit_rss_cached(ticker: str, sub: str, limit: int, timeout: float) -> tuple[dict, ...]:
    """Fetch once per ticker/subreddit per process and return immutable cached data."""
    url = _RSS.format(sub=sub, qs=_search_qs(ticker, limit))
    for attempt in range(2):
        _pace_request()
        req = Request(url, headers={"User-Agent": _UA})
        try:
            with urlopen(req, timeout=timeout) as resp:
                root = ET.fromstring(resp.read())
            posts = []
            for entry in root.findall("atom:entry", _ATOM_NS)[:limit]:
                title_el = entry.find("atom:title", _ATOM_NS)
                published_el = entry.find("atom:published", _ATOM_NS)
                content_el = entry.find("atom:content", _ATOM_NS)
                posts.append({
                    "title": (title_el.text if title_el is not None else "") or "",
                    "score": None,
                    "num_comments": None,
                    "created_utc": _iso_to_timestamp(published_el.text if published_el is not None else None),
                    "selftext": _strip_html(content_el.text if content_el is not None else ""),
                    "source": "rss",
                })
            return tuple(posts)
        except HTTPError as exc:
            if exc.code == 429 and attempt == 0:
                wait = _retry_after_seconds(exc)
                logger.warning("Reddit RSS 429 for r/%s · %s — waiting %.1fs before one retry", sub, ticker, wait)
                time.sleep(wait)
                continue
            logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
            return ()
        except (OSError, http.client.HTTPException, ET.ParseError) as exc:
            logger.warning("Reddit RSS fetch failed for r/%s · %s: %s", sub, ticker, exc)
            return ()
    return ()


def fetch_reddit_posts(
    ticker: str,
    subreddits: Iterable[str] = DEFAULT_SUBREDDITS,
    limit_per_sub: int = 5,
    timeout: float = 10.0,
    inter_request_delay: float = 2.0,
) -> str:
    """Fetch recent Reddit posts once per subreddit and reuse them for all agents."""
    ticker = crypto_base(ticker) or ticker
    blocks = []
    total_posts = 0
    for sub in subreddits:
        posts = list(_fetch_subreddit_rss_cached(ticker.upper(), sub, limit_per_sub, timeout))
        total_posts += len(posts)
        if not posts:
            blocks.append(f"r/{sub}: <no posts found mentioning {ticker.upper()} in the past 7 days>")
            continue
        lines = [f"r/{sub} — {len(posts)} recent posts mentioning {ticker.upper()} (via RSS; scores/comments unavailable):"]
        for p in posts:
            title = (p.get("title") or "").replace("\n", " ").strip()
            created = p.get("created_utc")
            created_str = time.strftime("%Y-%m-%d", time.gmtime(created)) if created else "?"
            selftext = (p.get("selftext") or "").replace("\n", " ").strip()
            if len(selftext) > 240:
                selftext = selftext[:240] + "…"
            lines.append(f"  [{created_str}] {title}" + (f"\n    body excerpt: {selftext}" if selftext else ""))
        blocks.append("\n".join(lines))
    if total_posts == 0:
        return f"<no Reddit posts found mentioning {ticker.upper()} across {', '.join(f'r/{s}' for s in subreddits)} in the past 7 days>"
    return "\n\n".join(blocks)
