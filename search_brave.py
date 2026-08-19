from __future__ import annotations

"""Keyless Brave Search HTML engine (steal B2 — searxng approach, own parser).

Searches ``https://search.brave.com/search?q=...&source=web`` with a desktop
Chrome UA and parses the server-rendered SERP (SvelteKit — results are in the
static DOM, no JS needed). Parser verified against live Brave HTML 2026-08
(sample: /tmp/brave_sample.html, query "rust async await").

Rate-limit conscious: Brave flags rapid repeat requests as bot activity, so
this engine is used sparingly (one query variant per search, not the multi-
variant fan-out). Time range and paging params follow searxng's brave.py.
"""

import urllib.parse

from scrapling.fetchers import AsyncFetcher

BRAVE_URL = "https://search.brave.com/search"
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 "
       "(KHTML, like Gecko) Chrome/151.0.0.0 Safari/537.36")
# searxng time_range_map: pd=day, pw=week, pm=month, py=year
_TR = {"d": "pd", "w": "pw", "m": "pm", "y": "py"}

# Build-hash-independent selectors (the svelte-XXXXXXX suffixes change per
# deploy). Container is the inner result-wrapper (its parent `snippet` class
# also matches divider/show-more chrome — the wrapper is the clean anchor).
_RESULT_BLOCK = (
    "//*[contains(concat(' ', normalize-space(@class), ' '), ' result-wrapper ')"
    " and not(contains(@class, 'result-wrapper-ad'))]"
)
_TITLE = ".//*[contains(concat(' ', normalize-space(@class), ' '), ' search-snippet-title ')]"
_HREF = _TITLE + "/ancestor::a[1]/@href"
_SNIPPET_A = (
    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' generic-snippet ')]"
    "//*[contains(concat(' ', normalize-space(@class), ' '), ' content ')]"
)
_SNIPPET_B = (
    ".//*[contains(concat(' ', normalize-space(@class), ' '), ' snippet ')"
    " and contains(@class, 'svelte-jmfu5f')]"
)


def _parse(html_bytes: bytes) -> list[dict]:
    from lxml import html as lxml_html

    doc = lxml_html.fromstring(html_bytes)
    results = []
    for block in doc.xpath(_RESULT_BLOCK):
        hrefs = block.xpath(_HREF)
        href = hrefs[0] if hrefs else None
        if not href or not href.startswith("http"):
            continue  # ad or malformed partial url
        titles = block.xpath(_TITLE)
        title = titles[0].text_content().strip() if titles else ""
        snippet = ""
        for sel in (_SNIPPET_A, _SNIPPET_B):
            for cand in block.xpath(sel):
                txt = cand.text_content().strip()
                if txt:
                    snippet = txt
                    break
            if snippet:
                break
        results.append({
            "title": title, "url": href, "snippet": snippet,
            "source": urllib.parse.urlparse(href).netloc or "",
        })
    return results


async def search_brave(query: str, count: int = 10, timelimit: str = "",
                       page: int = 1, timeout: int = 10) -> list[dict]:
    """Search Brave (keyless HTML). Returns a plain result list."""
    params = {"q": query, "source": "web"}
    tf = _TR.get(timelimit)
    if tf:
        params["tf"] = tf
    if page > 1:
        params["offset"] = str(page - 1)
    url = f"{BRAVE_URL}?{urllib.parse.urlencode(params)}"
    try:
        resp = await AsyncFetcher.get(url, timeout=timeout, stealthy_headers=True)
        if not resp or resp.status != 200:
            return []
        return _parse(bytes(resp.body))[:count]
    except Exception:
        return []
