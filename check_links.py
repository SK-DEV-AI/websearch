"""E3b: web_check_links — dead-link detection for a batch of URLs.

Ported from corporatepiyush/mcp-web-search's `web_check_links`
(SSRF-guard discipline: every probe URL passes security.py validate_url,
same choke point as the rest of the server). Bounded concurrency,
HEAD first with GET fallback on 405/403, status classification.
"""

from __future__ import annotations

import asyncio
import re
from urllib.parse import urlparse

from security import SecurityError, validate_url

_BLOCKED_EXT = re.compile(
    r"\.(jpg|jpeg|png|gif|webp|svg|ico|mp4|mp3|zip|tar|gz|7z|exe|dmg|iso)$",
    re.I,
)
_OK_STATUSES = {200, 201, 202, 203, 204, 205, 206, 301, 302, 303, 307, 308}


def _classify(status: int, exc: str | None) -> str:
    if exc:
        if "timeout" in exc.lower():
            return "timeout"
        if "dns" in exc.lower() or "resolve" in exc.lower():
            return "dns"
        if "ssl" in exc.lower():
            return "ssl"
        return "error"
    if status in _OK_STATUSES:
        return "ok"
    if status == 404:
        return "gone"
    if status == 429:
        return "rate_limited"
    if status in (401, 403):
        return "blocked"
    if 500 <= status <= 599:
        return "server_error"
    return "broken"


async def check_links(urls: list[str], concurrency: int = 10,
                      timeout: float = 8.0) -> dict:
    """Probe a batch of URLs; classify each as ok/broken/blocked/...

    Skips media/archive extensions (no HTML to link to) and any URL
    that fails SSRF validation (reported as skipped_internal).
    """
    sem = asyncio.Semaphore(max(1, min(int(concurrency), 25)))
    import httpx

    async with httpx.AsyncClient(follow_redirects=True,
                                 timeout=timeout,
                                 limits=httpx.Limits(max_connections=25)) as client:
        async def probe(url: str) -> dict:
            async with sem:
                entry: dict = {"url": url}
                try:
                    entry["final_url"] = await validate_url(url)
                except SecurityError as e:
                    entry["status"] = "skipped_internal"
                    entry["reason"] = str(e)
                    return entry
                entry["url"] = entry["final_url"]
                path = urlparse(entry["final_url"]).path
                if _BLOCKED_EXT.search(path):
                    entry["status"] = "skipped_media"
                    return entry
                exc = None
                status = 0
                try:
                    r = await client.head(entry["final_url"])
                    status = r.status_code
                    if status in (405, 403) or not r.headers.get("content-length"):
                        r = await client.get(entry["final_url"])
                        status = r.status_code
                except Exception as e:  # noqa: BLE001
                    exc = f"{type(e).__name__}: {e}"
                entry["status"] = _classify(status, exc)
                entry["http_status"] = status
                if exc:
                    entry["error"] = exc
                return entry

        results = await asyncio.gather(*(probe(u) for u in urls), return_exceptions=True)
    out: list[dict] = []
    for r in results:
        if isinstance(r, BaseException):
            out.append({"url": "", "status": "error", "error": str(r)})
        else:
            out.append(r)
    counts: dict[str, int] = {}
    for e in out:
        counts[e["status"]] = counts.get(e["status"], 0) + 1
    return {"checked": len(out), "by_status": counts, "links": out}
