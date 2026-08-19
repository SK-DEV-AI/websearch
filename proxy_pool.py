"""Health-scored free-proxy fallback tier (steal-antibots #4).

Proxifly feed -> validate-on-use -> score -> serve best. Fallback only:
free proxies are ban-prone on engines, never primary.
"""

from __future__ import annotations

import asyncio
import random
import time
from urllib.parse import urlparse

import httpx

FEED = "https://raw.githubusercontent.com/proxifly/free-proxy-list/main/proxies/all/data.txt"
FEED_TTL = 600  # feed refreshed every 5 min upstream; re-pull every 10
VALIDATE_URL = "https://example.com"
MAX_SCORE_AGE = 300  # success scores decay after 5 min idle

_proxies: list[str] = []
_fetched_at = 0.0
_scores: dict[str, float] = {}  # proxy -> success rate
_hits: dict[str, list[float]] = {}  # proxy -> recent timestamps of successes
_lock = asyncio.Lock()


async def _refresh() -> None:
    global _proxies, _fetched_at
    async with _lock:
        if time.monotonic() - _fetched_at < FEED_TTL:
            return
        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as c:
                r = await c.get(FEED)
                r.raise_for_status()
            parsed = [
                line.strip()
                for line in r.text.splitlines()
                if line.strip().startswith(("http://", "socks5://"))
            ]
            if parsed:
                _proxies = parsed
                _fetched_at = time.monotonic()
        except Exception:
            pass  # keep last known list


async def _fetch_via(proxy: str, url: str) -> httpx.Response:
    """GET through one proxy; raises on failure."""
    async with httpx.AsyncClient(
        proxy=proxy, timeout=12, follow_redirects=True,
        headers={"User-Agent": "Mozilla/5.0 (X11; Linux x86_64) research-bot/1.0"},
    ) as c:
        return await c.get(url)


def _pick() -> str:
    """Highest-scored proxy, random tiebreak among top quartile."""
    if not _proxies:
        return ""
    now = time.monotonic()
    scored = []
    for p in _proxies:
        s = _scores.get(p, 0.5)
        hits = [t for t in _hits.get(p, []) if now - t < MAX_SCORE_AGE]
        if hits:
            s = min(s + 0.1 * len(hits), 0.95)
        scored.append((s, p))
    scored.sort(key=lambda x: -x[0])
    top = max(1, len(scored) // 4)
    return random.choice(scored[:top])[1]


def _score(proxy: str, ok: bool) -> None:
    s = _scores.get(proxy, 0.5)
    _scores[proxy] = s * 0.8 + (0.1 if ok else -0.15)
    if ok:
        _hits.setdefault(proxy, []).append(time.monotonic())


async def proxy_fetch(url: str) -> str | None:
    """GET url through the best healthy proxy; None if all fail."""
    await _refresh()
    seen: set[str] = set()
    for _ in range(min(3, len(_proxies))):
        p = _pick()
        if not p or p in seen:
            break
        seen.add(p)
        try:
            r = await _fetch_via(p, url)
            if r.status_code < 500:
                _score(p, True)
                return r.text
        except Exception:
            pass
        _score(p, False)
    return None


async def _validate(proxy: str) -> bool:
    try:
        r = await _fetch_via(proxy, VALIDATE_URL)
        return r.status_code == 200
    except Exception:
        return False


async def healthy_count() -> int:
    """Number of currently responsive proxies (background check)."""
    await _refresh()
    sample = random.sample(_proxies, min(10, len(_proxies)))
    results = await asyncio.gather(*(_validate(p) for p in sample), return_exceptions=True)
    return sum(1 for r in results if r is True)


if __name__ == "__main__":
    async def demo():
        ok = await healthy_count()
        body = await proxy_fetch(VALIDATE_URL)
        print(f"healthy-ish {ok}/10  proxy fetch ok={body is not None} len={len(body) if body else 0}")
    asyncio.run(demo())