"""Bing RSS lanes — keyless general-web + news search (Brave replacement).

Brave Search is 429-dead from this egress (measured 2026-10-20), so this lane
covers the same gap using Bing's own RSS endpoints:

* general web: ``https://www.bing.com/search?q=<q>&format=rss``
* news:        ``https://www.bing.com/news/search?q=<q>&format=rss``

Both return RSS 2.0 with ``<item>`` elements (title, link, description,
pubDate). The news variant wraps the publisher URL in a Bing ``apiclick.aspx``
redirect — the real URL is recovered from its ``url=`` query parameter.

Results use the standard engine shape
``{"title", "url", "snippet", "published", "rank"}`` — identical to
``verticals._entry`` so the merge stage needs no special casing.

Advisory only: any failure returns ``[]`` and never raises.
"""

from __future__ import annotations

import email.utils
import html as _html
import logging
import urllib.parse
import xml.etree.ElementTree as ET

from config import get_http_client

logger = logging.getLogger("bing_rss")

BING_WEB_URL = "https://www.bing.com/search"
BING_NEWS_URL = "https://www.bing.com/news/search"

# Bing 403s requests without a browser-like User-Agent.
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36")

_TIMEOUT = 12.0
_MAX_ITEMS = 10

# Bing news links are redirects of the form
# http://www.bing.com/news/apiclick.aspx?...&url=<urlencoded real url>&...
_APICLICK = "apiclick.aspx"


def _rss_date_to_iso(date: str) -> str | None:
    """RSS pubDate -> ``YYYY-MM-DD``. Falls back to None."""
    raw = (date or "").strip()
    if not raw:
        return None
    try:
        dt = email.utils.parsedate_to_datetime(raw)
    except (TypeError, ValueError):
        return None
    if dt is None:
        return None
    return dt.date().isoformat()


def _real_url(link: str) -> str:
    """Unwrap Bing's apiclick redirect to the publisher URL.

    Non-redirect links (general web RSS) are returned unchanged.
    """
    link = (link or "").strip()
    if not link or _APICLICK not in link:
        return link
    try:
        qs = urllib.parse.parse_qs(urllib.parse.urlsplit(link).query)
        real = (qs.get("url") or [""])[0]
        if real.startswith("http"):
            return real
    except ValueError:
        pass
    return link


def _clean(text: str) -> str:
    return " ".join(_html.unescape(text or "").split())


def _parse_rss(body: str, count: int, *, news: bool) -> list[dict]:
    """Parse a Bing RSS document into the standard engine shape."""
    try:
        root = ET.fromstring(body)
    except ET.ParseError as e:
        logger.debug("bing rss parse failed: %s", e)
        return []
    out: list[dict] = []
    for item in root.findall(".//item"):
        if len(out) >= count:
            break
        title = _clean(item.findtext("title", ""))
        url = _real_url(_clean(item.findtext("link", "")))
        if not title or not url.startswith("http"):
            continue
        snippet = _clean(item.findtext("description", ""))
        if news:
            source = ""
            for child in item:
                tag = child.tag.rsplit("}", 1)[-1]
                if tag == "Source":
                    source = _clean(child.text or "")
                    break
            if source:
                snippet = f"{source}: {snippet}" if snippet else source
        pub = _clean(item.findtext("pubDate", ""))
        out.append({
            "title": title,
            "url": url,
            "snippet": snippet,
            "published": _rss_date_to_iso(pub),
            "rank": len(out),
        })
    return out


async def _fetch_rss(url: str, query: str, count: int, *, news: bool) -> list[dict]:
    try:
        c = get_http_client()
        r = await c.get(url, params={"q": query, "format": "rss"},
                        headers={"User-Agent": _UA,
                                 "Accept": "application/rss+xml, application/xml;q=0.9, */*;q=0.8"},
                        timeout=_TIMEOUT)
        if r.status_code != 200:
            logger.debug("bing rss %s -> HTTP %s", url, r.status_code)
            return []
        return _parse_rss(r.text, count, news=news)
    except Exception as e:
        logger.debug("bing rss failed: %s", e)
        return []


async def search_bing_web(query: str, count: int = 10, timeout: float = _TIMEOUT) -> list[dict]:
    """General web results via Bing's RSS endpoint (Brave replacement)."""
    if not (query or "").strip():
        return []
    n = max(1, min(int(count or _MAX_ITEMS), _MAX_ITEMS))
    return await _fetch_rss(BING_WEB_URL, query, n, news=False)


async def search_bing_news(query: str, count: int = 10, timeout: float = _TIMEOUT) -> list[dict]:
    """News results via Bing's news RSS endpoint."""
    if not (query or "").strip():
        return []
    n = max(1, min(int(count or _MAX_ITEMS), _MAX_ITEMS))
    return await _fetch_rss(BING_NEWS_URL, query, n, news=True)


if __name__ == "__main__":
    import asyncio

    async def check():
        for label, fn, q in (("web", search_bing_web, "python httpx timeout"),
                             ("news", search_bing_news, "python httpx")):
            hits = await fn(q, 10)
            print(f"bing {label}: {len(hits)} hits")
            for h in hits[:2]:
                print("  -", h["title"][:80], "|", h["url"][:90], "|", h["published"])

    asyncio.run(check())
    print("BING-RSS-OK")
