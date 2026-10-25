from __future__ import annotations

import asyncio
import copy
import functools
import hashlib
import json
import os
import threading
import time
from typing import Any

import httpx

CACHE_TTL = 120
MAX_RESULTS = 20
GOOGLE_AI_URL = "https://www.google.com/search"
GNEWS_RSS = "https://news.google.com/rss/search"
# CDP endpoint for the browser-dependent lanes (Google AI Mode, browser_action,
# CDP fetch fallback). Point this at ANY Chromium-based browser started with
# --remote-debugging-port. On a headless server the browser lanes simply stay
# unavailable; everything else works over plain HTTP with no keys.
HELIUM_CDP = os.environ.get("WEBSEARCH_CDP_URL", "http://127.0.0.1:9222").strip()

NV_KEY = os.environ.get("NV_KEY", "")
NV_BASE = "https://integrate.api.nvidia.com/v1"
NV_EMBED_MODEL = "nvidia/nemotron-3-embed-1b"
NV_EMBED_DIMS = 2048  # nemotron-3-embed-1b output dims (fixed 2048)


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

# ── Groq model split (proposal 1): scarce gpt-oss quota (1k RPD) is for
# synthesis/grounding/extract-llm; cheap high-volume ops (rewrite/classify/
# expand/decompose) use openai/gpt-oss-20b (llama-3.1-8b retired 404). Plain constants —
# env keys belong to the infra worker.
GROQ_SMART_MODEL = "openai/gpt-oss-120b"
GROQ_CHEAP_MODEL = "openai/gpt-oss-20b"
RERANKER_MODEL = os.environ.get("RERANKER_MODEL", "Alibaba-NLP/gte-reranker-modernbert-base")

# ── steal-theme config (C2/E2/B1/A2/ghost-tier/C3) ─────────────
QUALITY_FLOOR = float(os.environ.get("QUALITY_FLOOR", "0.35"))
SIX_SIGNAL_ENABLED = os.environ.get("SIX_SIGNAL_ENABLED", "true") == "true"
STEALTH_INJECT_ENABLED = os.environ.get("STEALTH_INJECT_ENABLED", "false") == "true"
STEALTH_HEADERS_ENABLED = os.environ.get("STEALTH_HEADERS_ENABLED", "false") == "true"
TIER_SKIP_THRESHOLD = float(os.environ.get("TIER_SKIP_THRESHOLD", "0.30"))
TIER_MIN_TRIES = int(os.environ.get("TIER_MIN_TRIES", "10"))
MINERU_DEVICE = os.environ.get("MINERU_DEVICE", "cpu")
MINERU_ENABLED = os.environ.get("MINERU_ENABLED", "false") == "true"
FAST_PATHS_ENABLED = os.environ.get("FAST_PATHS_ENABLED", "true") == "true"
ROBOTS_POLITENESS = os.environ.get("ROBOTS_POLITENESS", "true") == "true"
EXTRACT_METADATA_ENABLED = os.environ.get("EXTRACT_METADATA_ENABLED", "true") == "true"

def _clamp_int(raw: Any, lo: int, hi: int, default: int) -> int:
    try:
        return max(lo, min(hi, int(raw)))
    except (TypeError, ValueError):
        return default


# ── search effort defaults ───────────────────────────────────────
# depth is how much work a search does: 1 = snippets only, 2+ = fetch the top
# pages and score the full text. The right default depends on whether a model
# is configured downstream, because fetching pages only pays off when something
# reads them: with synthesis available, depth 2 measurably improves the answer;
# without it the caller gets snippets and a longer wait for nothing.
# Override with WEBSEARCH_DEFAULT_DEPTH for a fixed value.
DEPTH_MIN = 1
DEPTH_MAX = 5
DEPTH_DEFAULT_PLAIN = 1
# With a model available, fetching more pages is no longer wasted work: the
# model reads all of them and can reconcile contradictions between them. The
# value is set from measurement (exec/websearch-depth-bench), not taste.
DEPTH_DEFAULT_WITH_MODEL = 3

# Page fetches in the depth>=2 enrichment step. At least ENRICH_MIN_PAGES so
# depth 2 still gets a real pool; grows with depth up to the ceiling, which
# exists because 8 x 32K plus reranking overran a 60s client timeout.
ENRICH_MIN_PAGES = 5
ENRICH_MAX_PAGES = _clamp_int(os.environ.get("WEBSEARCH_ENRICH_MAX_PAGES"), 3, 12, 8)


def model_lanes_configured() -> bool:
    """True when any lane that consumes a model has credentials.

    Env-only and cheap on purpose: this runs on the search path, and reading a
    config file per call would be a filesystem hit inside a tool. `doctor`
    reports the authoritative answer (it calls distill.available(), which also
    knows about ~/.jcode/config.toml); this is the fast approximation used to
    pick a default.
    """
    return bool(
        os.environ.get("GROQ_API_KEYS", "").strip()
        or os.environ.get("DISTILL_API_KEY", "").strip()
    )


def default_depth(model_available: bool | None = None) -> int:
    """Resolve the depth a call should use when the caller did not specify one.

    WEBSEARCH_DEFAULT_DEPTH always wins when set, so behaviour is pinnable.
    Otherwise: 2 when a model lane is available, 1 when not.
    """
    raw = os.environ.get("WEBSEARCH_DEFAULT_DEPTH", "").strip()
    baseline = DEPTH_DEFAULT_WITH_MODEL if (
        model_lanes_configured() if model_available is None else model_available
    ) else DEPTH_DEFAULT_PLAIN
    return _clamp_int(raw, DEPTH_MIN, DEPTH_MAX, baseline)


def synthesis_settings() -> dict:
    """Context-shaping knobs for the synthesis step.

    Env                          default  meaning
    WEBSEARCH_SYNTH_TOKENS          1200  max_tokens for the answer (a ceiling;
                                          the sentence cap is the real control)
    WEBSEARCH_DOMAIN_CAP               2  max sources from one registrable domain
    WEBSEARCH_MIN_SOURCES              3  floor before the domain cap relaxes
    WEBSEARCH_SYNTH_MAX_SOURCES        8  ceiling on sources in the context
    WEBSEARCH_SYNTH_SNIPPET_CHARS   1500  chars of each source put in the context
    """
    return {
        "max_tokens": _clamp_int(
            os.environ.get("WEBSEARCH_SYNTH_TOKENS"), 200, 8000, 1200),
        "domain_cap": _clamp_int(
            os.environ.get("WEBSEARCH_DOMAIN_CAP"), 1, 10, 2),
        "min_sources": _clamp_int(
            os.environ.get("WEBSEARCH_MIN_SOURCES"), 1, 10, 3),
        "max_sources": _clamp_int(
            os.environ.get("WEBSEARCH_SYNTH_MAX_SOURCES"), 1, 20, 8),
        "snippet_chars": _clamp_int(
            os.environ.get("WEBSEARCH_SYNTH_SNIPPET_CHARS"), 200, 8000, 1500),
    }


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
                    return copy.deepcopy(entry[1])
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
_http_client_lock = threading.Lock()


class _PoolStream(httpx.AsyncByteStream):
    """Adapt an httpcore response stream to httpx's stream interface."""
    def __init__(self, stream) -> None:
        self._stream = stream

    async def __aiter__(self):
        async for chunk in self._stream:
            yield chunk

    async def aclose(self) -> None:
        await self._stream.aclose()


class _PinningTransport(httpx.AsyncBaseTransport):
    """httpx transport with DNS-rebinding pinning, built on public APIs only.
    Owns an httpcore pool with our backend instead of poking
    ``transport._pool`` (private — silently breaks on httpx upgrades,
    taking the DNS pin with it)."""
    def __init__(self, limits: httpx.Limits) -> None:
        from security import PinningNetworkBackend
        import httpcore
        self._pool = httpcore.AsyncConnectionPool(
            network_backend=PinningNetworkBackend(),
            max_keepalive_connections=limits.max_keepalive_connections,
            max_connections=limits.max_connections)

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        import httpcore
        req = httpcore.Request(
            method=request.method,
            url=str(request.url),
            headers=request.headers.multi_items(),
            content=request.stream,
            extensions=request.extensions)
        resp = await self._pool.handle_async_request(req)
        return httpx.Response(
            status_code=resp.status,
            headers=resp.headers,
            stream=_PoolStream(resp.stream),
            extensions=resp.extensions)

    async def aclose(self) -> None:
        await self._pool.aclose()


def get_http_client() -> httpx.AsyncClient:
    """Return a shared httpx.AsyncClient with connection pooling."""
    global _http_client
    if _http_client is None or _http_client.is_closed:
        with _http_client_lock:
            if _http_client is None or _http_client.is_closed:
                limits = httpx.Limits(
                    max_keepalive_connections=10, max_connections=20)
                _http_client = httpx.AsyncClient(
                    timeout=30.0,
                    follow_redirects=True,
                    transport=_PinningTransport(limits),
                )
    return _http_client


async def close_http_client():
    """Close the shared httpx client on server shutdown."""
    global _http_client
    if _http_client and not _http_client.is_closed:
        await _http_client.aclose()
        _http_client = None
