from __future__ import annotations

import asyncio
import functools
import hashlib
import json
import os
import time
from typing import Any

import httpx

CACHE_TTL = 120
MAX_RESULTS = 20
GOOGLE_AI_URL = "https://www.google.com/search"
GNEWS_RSS = "https://news.google.com/rss/search"
HELIUM_CDP = "http://127.0.0.1:9222"

NV_KEY = os.environ.get("NV_KEY", "")
NV_BASE = "https://integrate.api.nvidia.com/v1"
NV_EMBED_MODEL = "nvidia/llama-nemotron-embed-1b-v2"


class _KeyRotator:
    """Thread-safe round-robin key rotator for API keys."""

    def __init__(self, env_var: str, fallback_var: str = ""):
        raw = os.environ.get(env_var) or os.environ.get(fallback_var, "")
        self._keys: list[str] = [k.strip() for k in raw.split(",") if k.strip()] if raw else []
        self._idx = 0
        self._lock = asyncio.Lock()

    async def next(self) -> str | None:
        if not self._keys:
            return None
        async with self._lock:
            k = self._keys[self._idx % len(self._keys)]
            self._idx = (self._idx + 1) % len(self._keys)
            return k

    @property
    def first(self) -> str:
        return self._keys[0] if self._keys else ""

    @property
    def has_keys(self) -> bool:
        return bool(self._keys)


TAVILY_KEYS = [k.strip() for k in os.environ.get("TAVILY_KEYS", "").split(",") if k.strip()]
_tavily_idx = 0

def _next_tavily_key() -> str:
    global _tavily_idx
    if not TAVILY_KEYS:
        return ""
    key = TAVILY_KEYS[_tavily_idx % len(TAVILY_KEYS)]
    _tavily_idx += 1
    return key


TINYFISH_KEYS = [k.strip() for k in os.environ.get("TINYFISH_KEYS", "").split(",") if k.strip()]

GROQ_API_KEYS = [k.strip() for k in os.environ.get("GROQ_API_KEYS", "").split(",") if k.strip()]
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "Alibaba-NLP/gte-reranker-modernbert-base")


_cache: dict[str, tuple[float, Any]] = {}
_MAX_CACHE = 500
_CACHE_TTL: dict[str, int] = {"emb": 600, "search": 90, "fetch": 120}
_cache_lock = asyncio.Lock()


async def _cached(key: str) -> Any | None:
    async with _cache_lock:
        entry = _cache.get(key)
        if entry and time.monotonic() - entry[0] < _CACHE_TTL.get(key.split(":")[0], 300):
            return entry[1]
    return None


async def _set_cache(key: str, val: Any):
    async with _cache_lock:
        _cache[key] = (time.monotonic(), val)
        if len(_cache) > _MAX_CACHE:
            sorted_keys = sorted(_cache, key=lambda k: _cache[k][0])
            evict_count = max(1, len(sorted_keys) // 4)
            for k in sorted_keys[:evict_count]:
                _cache.pop(k, None)


def cached(ttl: int = CACHE_TTL):
    def deco(fn):
        @functools.wraps(fn)
        async def wrapper(*args, **kw):
            raw = json.dumps([args, kw], sort_keys=True, default=str)
            key = f"{fn.__name__}:{hashlib.md5(raw.encode()).hexdigest()}"
            now = time.monotonic()
            async with _cache_lock:
                entry = _cache.get(key)
                if entry and now - entry[0] < ttl:
                    return entry[1]
            result = await fn(*args, **kw)
            async with _cache_lock:
                _cache[key] = (now, result)
                if len(_cache) > _MAX_CACHE:
                    cutoff = now - 300
                    stale = [k for k, (t, _) in _cache.items()
                             if now - t > _CACHE_TTL.get(k.split(":")[0], 300)]
                    for k in stale:
                        del _cache[k]
            return result
        return wrapper
    return deco


# ── Shared HTTP client pool ───────────────────────────────────────

_http_client: httpx.AsyncClient | None = None


def get_http_client() -> httpx.AsyncClient:
    """Return a shared httpx.AsyncClient with connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        _http_client = httpx.AsyncClient(
            timeout=30.0,
            follow_redirects=True,
            limits=httpx.Limits(max_keepalive_connections=10, max_connections=20),
        )
    return _http_client


async def close_http_client():
    """Close the shared httpx client on server shutdown."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None
