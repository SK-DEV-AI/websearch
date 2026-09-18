from __future__ import annotations

import logging
from typing import Any

import httpx

from config import _next_tavily_key, get_http_client

logger = logging.getLogger("tavily")

TAVILY_SEARCH = "https://api.tavily.com/search"


async def search_tavily(query: str, n: int = 10, topic: str = "general",
                        time_range: str = "", include_raw_content: bool = True,
                        search_depth: str = "advanced", include_answer: bool = True,
                        include_images: bool = True, auto_parameters: bool = True,
                        include_domains: list | None = None,
                        exclude_domains: list | None = None,
                        chunks_per_source: int = 0,
                        start_date: str = "", end_date: str = "") -> dict:
    """Tavily search — AI-optimized with 1K free reqs/month per key. Returns answer + results."""
    key = _next_tavily_key()
    if not key:
        return {"results": [], "answer": ""}
    body: dict[str, Any] = {"query": query, "search_depth": search_depth,
                            "max_results": min(n, 20), "include_answer": include_answer,
                            "include_raw_content": include_raw_content,
                            "topic": topic, "auto_parameters": auto_parameters}
    if include_images:
        body["include_images"] = True
    if start_date:
        body["start_date"] = start_date
    if end_date:
        body["end_date"] = end_date
    if time_range and not start_date and not end_date:
        body["time_range"] = time_range
    if include_domains:
        body["include_domains"] = include_domains
    if exclude_domains:
        body["exclude_domains"] = exclude_domains
    if chunks_per_source > 0:
        body["chunks_per_source"] = chunks_per_source
    try:
        c = get_http_client()
        r = await c.post(TAVILY_SEARCH, json=body,
                         headers={"Authorization": f"Bearer {key}",
                                  "Content-Type": "application/json"},
                         timeout=15)
        if r.status_code != 200:
            logger.warning("Tavily non-200: %s %s", r.status_code, r.text[:200])
            return {"results": [], "answer": ""}
        data = r.json()
        results: list[dict] = []
        # Tavily's synthesized answer is surfaced as a first-class field,
        # NOT a result entry: url-less entries are dropped by the consume
        # loop (research.py requires r.get("url")), so an entry here paid
        # quota + tokens and never reached the model.
        answer = data.get("answer", "")
        for item in (data.get("results", []) or []):
            url = item.get("url", "")
            if not url:
                continue
            entry: dict[str, Any] = {"title": item.get("title", ""), "url": url,
                           "snippet": (item.get("content", "") or "")[:500],
                           "engine": "tavily"}
            if item.get("score"):
                entry["score"] = item["score"]
            results.append(entry)
        return {"results": results, "answer": answer}
    except httpx.HTTPError as e:
        logger.warning("Tavily search failed: %s", e)
        return {"results": [], "answer": ""}



