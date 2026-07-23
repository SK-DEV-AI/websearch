from __future__ import annotations

import asyncio
import html
import urllib.parse
import xml.etree.ElementTree as ET

from typing import Any

from ddgs import DDGS
from scrapling.fetchers import AsyncFetcher

from config import GNEWS_RSS


async def search_ddg(
    query: str, count: int = 10, search_type: str = "auto", backend: str = "auto",
    timelimit: str = "", page: int = 1, region: str = "us-en", proxy: str = "",
    timeout: int = 10, safesearch: str = "moderate",
    size: str = "", color: str = "", type_image: str = "", layout: str = "",
    license_image: str = "", resolution: str = "", duration: str = "",
    license_videos: str = "",
) -> list[dict]:
    try:
        DDGS.threads = 8  # parallelize across 8 metasearch engines
        ddgs = DDGS(proxy=proxy or None, timeout=timeout)
        if search_type == "news":
            kw = dict(query=query, max_results=count, region=region, safesearch=safesearch,
                      timelimit=timelimit or None, backend=backend, page=page)
            raw = await asyncio.to_thread(lambda: list(ddgs.news(**kw)))
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
            raw = await asyncio.to_thread(lambda: list(ddgs.images(**kw)))
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
            raw = await asyncio.to_thread(lambda: list(ddgs.videos(**kw)))
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
            raw = await asyncio.to_thread(
                lambda: list(ddgs.books(query=query, max_results=count, backend=backend, page=page)))
            return [{"title": r.get("title", ""), "url": r.get("url", ""),
                     "snippet": r.get("info", "") or r.get("author", ""),
                     "author": r.get("author", ""), "publisher": r.get("publisher", ""),
                     "source": "books", "thumbnail": r.get("thumbnail", "")}
                    for r in raw if r.get("url")]
        else:
            kw = dict(query=query, max_results=count, region=region, safesearch=safesearch,
                      timelimit=timelimit or None, backend=backend, page=page)
            raw = await asyncio.to_thread(lambda: list(ddgs.text(**kw)))
            return [{"title": r.get("title", ""), "url": r.get("href", ""),
                     "snippet": r.get("body", ""),
                     "source": urllib.parse.urlparse(r.get("href", "")).netloc or ""}
                    for r in raw if r.get("href")]
    except Exception as e:
        return [{"error": str(e)}]





async def ddgs_extract(url: str, extract_type: str = "markdown") -> dict | None:
    """Lightweight URL content extraction using ddgs library (no trafilatura needed)."""
    try:
        ddgs = DDGS(timeout=10)
        fmt_map = {"markdown": "text_markdown", "text_plain": "text", "raw": "text"}
        fmt = fmt_map.get(extract_type, "text_markdown")
        result = await asyncio.to_thread(lambda: ddgs.extract(url, fmt=fmt))
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
    parts = region.split("-")
    hl = parts[0] if len(parts) > 0 else "en"
    gl = parts[1].upper() if len(parts) > 1 else parts[0].upper()
    ceid = f"{gl}:{hl}"
    rss_url = f"{GNEWS_RSS}?q={urllib.parse.quote_plus(query)}&hl={hl}&gl={gl}&ceid={ceid}"
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
