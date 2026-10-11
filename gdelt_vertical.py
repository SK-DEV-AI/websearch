"""GDELT 2.0 Doc API news supplement (proposal 9 in task list).

Keyless, unlimited, 15-min updates, 100+ langs. Supplements the Google News
RSS vertical on news intent. Caveat per research: event-level data, not clean
article JSON — parse articles defensively and treat GDELT fields as leads.
Advisory-only: failures return [], never raise.
"""

from __future__ import annotations

import logging
import urllib.parse

logger = logging.getLogger("gdelt_vertical")

_GDELT_DOC = "https://api.gdeltproject.org/api/v2/doc/doc"
_TIMEOUT = 12.0
_SPACING_S = 6.0  # GDELT asks max 1 req / 5s; guard in-process, best-effort
_last_gdelt_ts = 0.0


def _shape(rank: int, art: dict) -> dict | None:
    url = (art.get("url") or "").strip()
    if not url.startswith("http"):
        return None
    title = (art.get("title") or "").strip()
    domain = (art.get("domain") or "").strip()
    date = (art.get("seendate") or "")[:10] or None
    snippet = " ".join(
        b for b in [domain, (art.get("socialimage") or "").strip()]
        if b
    )
    lang = (art.get("language") or "").strip()
    if lang and lang.lower() != "english":
        snippet = (snippet + f" [{lang}]").strip()
    return {"title": title or url, "url": url,
            "snippet": snippet or "GDELT news hit",
            "published": date, "rank": rank}


async def search_gdelt(query: str, count: int = 5,
                       mode: str = "artlist") -> list[dict]:
    """GDELT Doc API artlist for query. mode in {artlist, timelinevol}."""
    global _last_gdelt_ts
    try:
        import asyncio as _aio
        import time as _t
        wait = _SPACING_S - (_t.monotonic() - _last_gdelt_ts)
        if wait > 0:
            await _aio.sleep(wait)
        _last_gdelt_ts = _t.monotonic()
    except Exception:
        pass
    try:
        from config import get_http_client
        c = get_http_client()
        r = await c.get(
            _GDELT_DOC,
            params={"query": query, "mode": mode, "format": "json",
                    "maxrecords": min(max(count * 4, 10), 250)},
            timeout=_TIMEOUT,
            headers={"User-Agent": "websearch-mcp/1.0",
                     "Accept": "application/json"},
        )
        if r.status_code == 429:
            # Stamp the guard so a burst of news verticals does not hammer
            # the 1-req/5s limit in a tight loop.
            try:
                import time as _t2
                _last_gdelt_ts = _t2.monotonic()
            except Exception:
                pass
            return []
        if r.status_code != 200:
            return []
        ct = (r.headers.get("content-type", "") or "").lower()
        if "json" not in ct:
            return []
        try:
            v = r.json()
        except Exception:
            return []
        arts = (v.get("articles") or []) if isinstance(v, dict) else []
        out: list[dict] = []
        seen: set[str] = set()
        for art in arts:
            if not isinstance(art, dict):
                continue
            shaped = _shape(len(out), art)
            if not shaped or shaped["url"] in seen:
                continue
            seen.add(shaped["url"])
            out.append(shaped)
            if len(out) >= count:
                break
        return out
    except Exception as e:
        logger.debug("gdelt search failed: %s", e)
        return []


def gdelt_query_url(query: str) -> str:
    return (_GDELT_DOC + "?" + urllib.parse.urlencode(
        {"query": query, "mode": "artlist", "format": "html"}))


if __name__ == "__main__":
    import asyncio

    async def check():
        hits = await search_gdelt("semiconductor export controls")
        print(f"gdelt: {len(hits)} hits")
        for h in hits[:2]:
            print(" -", h["title"][:90], "|", h["url"][:110])

    asyncio.run(check())
