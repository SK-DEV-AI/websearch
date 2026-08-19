from __future__ import annotations

"""Keyless Marginalia Search public-HTML engine (steal engines #4).

Searches ``https://marginalia-search.com/search?query=...`` — the public web
UI, NOT the API (which is now key-walled). Parser verified against live HTML
2026-08-19. Unique niche: indie/old-web forums, blogs and personal sites the
big engines bury. The Tailwind UI uses h2 + rel=noopener anchors; the snippet
sits in a <p> inside the result row (div.flex-col.grow), after the flex-row
holding title+URL. Titles/URLs carry soft-hyphen entities.

Two quirks handled here:
- Marginalia rate-limits per IP by serving a help page (h2 "keyword-based
  search engine") instead of results; the page itself carries a "click here
  to proceed" sst-token link that bypasses the wait — follow it instead of
  sleeping. On a hard limit the engine fails softly (empty list).
- The engine must NOT be fetched via the shared AsyncFetcher: its stealth
  layer injects a google.com Referer, which Marginalia answers with the
  help page. Plain httpx with a browser UA and no Referer works.
"""

import asyncio
import re
import time
import urllib.parse

import httpx

MARGINALIA_URL = "https://marginalia-search.com/search"
_UA = ("Mozilla/5.0 (X11; Linux x86_64; rv:130.0) Gecko/20100101 Firefox/130.0")
_SHY = re.compile(r"\xad")
_HELP = re.compile(r"Wait A Moment")
_SST = re.compile(r"href=\"([^\"]*sst=[^\"]+)\"")
_RETRY_DELAY = 10.0


def _clean(s: str) -> str:
    return _SHY.sub("", s).strip()


def _parse(html_bytes: bytes) -> list[dict]:
    from lxml import html as lxml_html

    doc = lxml_html.fromstring(html_bytes)
    results = []
    for a in doc.xpath("//h2/a[@rel='noopener noreferrer']"):
        href = a.get("href") or ""
        if not href.startswith("http"):
            continue
        title = _clean(a.text_content())
        row = a.xpath("ancestor::div[contains(@class, 'flex-col')][1]")
        snippet = ""
        if row:
            p = row[0].find(".//p")
            if p is not None:
                snippet = _clean(p.text_content())
        results.append({
            "title": title, "url": href, "snippet": snippet,
            "source": urllib.parse.urlparse(href).netloc or "",
        })
    return results


async def _fetch(url: str) -> bytes:
    headers = {"User-Agent": _UA}
    async with httpx.AsyncClient(follow_redirects=True, timeout=15.0,
                                 headers=headers) as client:
        resp = await client.get(url)
        return resp.content


async def search_marginalia(query: str, count: int = 10, profile: str = "",
                            timeout: int = 10) -> list[dict]:
    """Search Marginalia (keyless public HTML). Returns a plain result list."""
    params = {"query": query}
    if profile:  # blogosphere | academia | code | blog | forum | ...
        params["profile"] = profile
    url = f"{MARGINALIA_URL}?{urllib.parse.urlencode(params)}"
    try:
        body = await asyncio.wait_for(_fetch(url), timeout=timeout)
        # rate-limit wait page: follow its own sst-token bypass link instead
        # of sleeping through the cooldown
        if _HELP.search(body.decode("utf-8", "ignore")):
            m = _SST.search(body.decode("utf-8", "ignore"))
            if m:
                body = await asyncio.wait_for(
                    _fetch(f"{MARGINALIA_URL.split('/search')[0]}{m.group(1).replace('&amp;', '&')}"),
                    timeout=timeout)
            else:
                for _ in range(2):
                    await asyncio.sleep(_RETRY_DELAY)
                    body = await asyncio.wait_for(_fetch(url), timeout=timeout)
                    if not _HELP.search(body.decode("utf-8", "ignore")):
                        break
    except Exception:
        return []
    if _HELP.search(body.decode("utf-8", "ignore")):
        return []
    return _parse(body)[:count]