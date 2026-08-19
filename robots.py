"""robots.txt politeness layer (steal-antibots #5).

Per-domain robots.txt parse cache (24h TTL) + tenacity jittered backoff.
"""

import asyncio
import random
import time
from urllib.parse import urlparse

from protego import Protego

from scrapling.fetchers import AsyncFetcher

UA = "research-bot/1.0"

_cache: dict[str, tuple[float, Protego | None]] = {}
_TTL = 24 * 3600
_lock = asyncio.Lock()


async def _fetch_robots(netloc: str) -> Protego | None:
    url = f"https://{netloc}/robots.txt"
    try:
        resp = await AsyncFetcher.get(url, timeout=8, headers={"User-Agent": UA})
    except Exception:
        return None
    if resp.status != 200:
        return None
    body = resp.body
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    return Protego.parse(body)


async def allowed(url: str) -> bool:
    """True if robots.txt permits fetching this URL (or robots is unreachable)."""
    p = urlparse(url)
    netloc = p.netloc.lower()
    now = time.monotonic()
    async with _lock:
        entry = _cache.get(netloc)
        if entry is None or now - entry[0] > _TTL:
            entry = (now, await _fetch_robots(netloc))
            _cache[netloc] = entry
    parser = entry[1]
    if parser is None:
        return True
    return parser.can_fetch(url, UA)


def backoff(retries: int = 3, base: float = 1.0, max_wait: float = 8.0):
    """Retry decorator with exponential backoff + jitter (tenacity-style).

    Retries on transient network errors (status 429/5xx, connection errors).
    """
    def deco(fn):
        async def wrapper(*args, **kwargs):
            for attempt in range(retries + 1):
                try:
                    return await fn(*args, **kwargs)
                except Exception:
                    if attempt >= retries:
                        raise
                    wait = min(max_wait, base * (2 ** attempt)) * random.uniform(0.5, 1.5)
                    await asyncio.sleep(wait)
            return None
        return wrapper
    return deco