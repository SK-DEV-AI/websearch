from __future__ import annotations

import logging
from typing import Any

import httpx

from config import TINYFISH_KEYS as _TINYFISH_KEYS, get_http_client

logger = logging.getLogger("tinyfish")

_TINYFISH_IDX = 0


def _next_key() -> str:
    global _TINYFISH_IDX
    keys = _TINYFISH_KEYS
    if not keys:
        return ""
    key = keys[_TINYFISH_IDX % len(keys)]
    _TINYFISH_IDX += 1
    return key


async def tinyfish_search(
    query: str,
    count: int = 10,
    domain_type: str = "web",
    location: str = "",
    language: str = "",
    recency_minutes: int = 0,
    after_date: str = "",
    before_date: str = "",
    page: int = 0,
    goal: str = "",
) -> list[dict]:
    """Search via TinyFish API. Returns list of {title, url, snippet, source}."""
    key = _next_key()
    if not key:
        return []
    headers = {"X-API-Key": key}
    params: dict[str, Any] = {"query": query}
    if domain_type in ("web", "news", "research_paper"):
        params["domain_type"] = domain_type
    if goal:
        params["purpose"] = goal[:2000]
    if location:
        params["location"] = location
    if language:
        params["language"] = language
    if recency_minutes > 0:
        params["recency_minutes"] = recency_minutes
    if after_date:
        params["after_date"] = after_date
    if before_date:
        params["before_date"] = before_date
    if page > 0:
        params["page"] = min(page, 10)
    try:
        c = get_http_client()
        resp = await c.get("https://api.search.tinyfish.ai", headers=headers, params=params)
        if resp.status_code != 200:
            return []
        data = resp.json()
        results = []
        for item in (data.get("results") or data.get("data") or []):
            url = item.get("url", "")
            if not url:
                continue
            r: dict[str, Any] = {
                "title": item.get("title", ""),
                "url": url,
                "snippet": item.get("snippet", item.get("description", "")),
            }
            if domain_type == "research_paper":
                r["authors"] = item.get("authors", [])
                r["venue"] = item.get("venue", "")
                r["year"] = item.get("year", "")
                r["citations"] = item.get("cited_by_count", 0)
                r["pdf_url"] = item.get("pdf_url", "")
            results.append(r)
        return results[:count]
    except Exception as e:
        logger.warning("Tinyfish search failed: %s", e)
        return []
