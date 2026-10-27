from __future__ import annotations

import asyncio
import html
import os
import logging
import urllib.parse
import xml.etree.ElementTree as ET

from typing import Any

from concurrent.futures import ThreadPoolExecutor

from ddgs import DDGS
from scrapling_shim import AsyncFetcher

from config import GNEWS_RSS

logger = logging.getLogger("ddg")

DDGS.threads = 8  # parallelize across 8 metasearch engines

# ddgs is a blocking Rust client: every call has to leave the event loop. It used
# to leave via asyncio.to_thread, i.e. the loop's ONE shared default executor,
# which every other blocking call in the server also uses (PDF work, reranker
# spawn, file IO). Under a swarm that pool fills with 10-15 s network calls and
# then everything queues behind them — one of the ways a search that normally
# returns in 3 s ended up crossing the client's 300 s timeout. A dedicated pool
# keeps flaky-network latency contained to search.
_NET_EXEC = ThreadPoolExecutor(max_workers=8, thread_name_prefix="ddg")
_NET_TIMEOUT = float(os.environ.get("WEBSEARCH_DDG_TIMEOUT", "0") or 0)


async def _off(fn, *args):
    """Run a blocking ddgs call on the dedicated network pool."""
    loop = asyncio.get_running_loop()
    fut = loop.run_in_executor(_NET_EXEC, fn, *args)
    if _NET_TIMEOUT <= 0:
        return await fut
    try:
        return await asyncio.wait_for(fut, timeout=_NET_TIMEOUT)
    except asyncio.TimeoutError:
        # The thread keeps running (Python cannot cancel it), but the request
        # stops waiting on it. Bounded by WEBSEARCH_DDG_TIMEOUT.
        logger.warning("ddg call exceeded %.0fs budget; abandoning wait", _NET_TIMEOUT)
        return []


async def search_ddg(
    query: str, count: int = 10, search_type: str = "auto", backend: str = "auto",
    timelimit: str = "", page: int = 1, region: str = "us-en", proxy: str = "",
    timeout: int = 10, safesearch: str = "moderate",
    size: str = "", color: str = "", type_image: str = "", layout: str = "",
    license_image: str = "", resolution: str = "", duration: str = "",
    license_videos: str = "",
) -> list[dict]:
    if region == "wt-wt":
        region = "us-en"
    try:
        ddgs = DDGS(proxy=proxy or None, timeout=timeout)
        if search_type == "news":
            kw = dict(query=query, max_results=count, region=region, safesearch=safesearch,
                      timelimit=timelimit or None, backend=backend, page=page)
            raw = await _off(lambda: list(ddgs.news(**kw)))
            return [{"title": r.get("title", ""), "url": r.get("url", ""),
                     "snippet": r.get("body", ""),
                     "source": urllib.parse.urlparse(r.get("url", "")).netloc or "",
                     "date": r.get("date", ""), "image": r.get("image", "")}
                    for r in raw if r.get("url")]
        elif search_type == "images":
            kw = dict(query=query, max_results=count, region=region, safesearch=safesearch,
                      timelimit=timelimit or None, backend=backend, page=page)
            for k, v in [("size", size), ("color", color), ("type_image", type_image),
                         ("layout", layout), ("license_image", license_image)]:
                if v:
                    kw[k] = v
            raw = await _off(lambda: list(ddgs.images(**kw)))
            return [{"title": r.get("title", ""), "url": r.get("image", ""),
                     "snippet": r.get("source", ""),
                     "source": urllib.parse.urlparse(r.get("image", "")).netloc or "",
                     "thumbnail": r.get("thumbnail", ""), "height": r.get("height", 0),
                     "width": r.get("width", 0)}
                    for r in raw if r.get("image")]
        elif search_type == "videos":
            kw = dict(query=query, max_results=count, region=region, safesearch=safesearch,
                      timelimit=timelimit or None, backend=backend, page=page)
            for k, v in [("resolution", resolution), ("duration", duration),
                         ("license_videos", license_videos)]:
                if v:
                    kw[k] = v
            raw = await _off(lambda: list(ddgs.videos(**kw)))
            return [{"title": r.get("title", ""),
                     "url": r.get("content", "") or r.get("embed_url", ""),
                     "snippet": r.get("description", ""),
                     "source": urllib.parse.urlparse(
                         r.get("content", "") or r.get("embed_url", "")).netloc or "",
                     "duration": r.get("duration", ""), "publisher": r.get("publisher", ""),
                     "published": r.get("published", ""),
                     "statistics": r.get("statistics", {}),
                     "images": r.get("images", {})}
                    for r in raw if r.get("content") or r.get("embed_url")]
        elif search_type == "books":
            raw = await _off(
                lambda: list(ddgs.books(query=query, max_results=count, backend=backend, page=page)))
            return [{"title": r.get("title", ""), "url": r.get("url", ""),
                     "snippet": r.get("info", "") or r.get("author", ""),
                     "author": r.get("author", ""), "publisher": r.get("publisher", ""),
                     "source": "books", "thumbnail": r.get("thumbnail", "")}
                    for r in raw if r.get("url")]
        else:
            kw = dict(query=query, max_results=count, region=region, safesearch=safesearch,
                      timelimit=timelimit or None, backend=backend, page=page)
            raw = await _off(lambda: list(ddgs.text(**kw)))
            return [{"title": r.get("title", ""), "url": r.get("href", ""),
                     "snippet": r.get("body", ""),
                     "source": urllib.parse.urlparse(r.get("href", "")).netloc or ""}
                    for r in raw if r.get("href")]
    except Exception as e:
        logger.warning("DDG search failed: %s", e)
        return []





async def ddgs_extract(url: str, extract_type: str = "markdown") -> dict | None:
    """Lightweight URL content extraction using ddgs library (no trafilatura needed)."""
    try:
        ddgs = DDGS(timeout=10)
        fmt_map = {"markdown": "text_markdown", "text_plain": "text", "raw": "text"}
        fmt = fmt_map.get(extract_type, "text_markdown")
        result = await _off(lambda: ddgs.extract(url, fmt=fmt))
        if result:
            return {"url": url, "content": result.get("content", ""),
                    "title": result.get("title", ""), "description": result.get("description", ""),
                    "extract_type": extract_type}
        return None
    except Exception:
        return None


async def search_google_rss(query: str, count: int = 10, region: str = "en",
                            retries: int = 2, timelimit: str = "") -> list[dict]:
    # Parse region ("us-en", "de-de", "in-en", "fr-fr", etc.) into hl/gl/ceid
    # Google News expects hl=<language>, gl=<COUNTRY>, ceid=<COUNTRY>:<language>
    # Single-token ("en") → language only; "wt-wt" (worldwide) → no locale params.
    parts = region.split("-")
    if len(parts) == 2 and parts[0] != "wt":
        hl, gl = parts[1], parts[0].upper()
        locale = f"&hl={hl}&gl={gl}&ceid={gl}:{hl}"
    elif len(parts) == 1:
        locale = f"&hl={parts[0]}"
    else:
        locale = ""
    rss_url = f"{GNEWS_RSS}?q={urllib.parse.quote_plus(query)}{locale}"
    rss_time_map = {"d": "1d", "w": "7d", "m": "1m", "y": "1y"}
    when = rss_time_map.get(timelimit, "")
    if when:
        rss_url += f"&when={when}"
    last_err = None
    for attempt in range(retries + 1):
        try:
            p = await AsyncFetcher.get(rss_url, timeout=10, stealthy_headers=True)
            raw = p.body if isinstance(p.body, str) else p.body.decode("utf-8", errors="replace")
            root = ET.fromstring(raw)
            return [{"title": item.findtext("title", ""),
                     "url": item.findtext("link", ""),
                     "snippet": html.unescape(item.findtext("description", "")),
                     "source": urllib.parse.urlparse(item.findtext("link", "")).netloc or "",
                     "published": item.findtext("pubDate", "")}
                    for item in root.findall(".//item")[:count]
                    if item.findtext("link", "")]
        except Exception as e:
            last_err = e
            if attempt < retries:
                await asyncio.sleep(1 * (attempt + 1))
    return []
