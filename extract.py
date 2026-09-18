"""Structured extraction — LLM-powered + CSS-driven + regex extraction.

Runs on page HTML fetched through our own pipeline (fetch_url: httpx +
trafilatura, CDP fallback for walled pages) — NOT a second Chromium.
crawl4ai's AsyncWebCrawler used to launch its own Playwright Chromium here
(~1 GB) alongside our Helium session; the strategies only need HTML, so we
feed them directly. crawl4ai stays a dependency for the strategy classes.
"""

from __future__ import annotations

import json
import os

from crawl4ai.extraction_strategy import (
    LLMExtractionStrategy,
    JsonCssExtractionStrategy,
    RegexExtractionStrategy,
)
from crawl4ai import LLMConfig

_DEFAULT_LLM = "groq/openai/gpt-oss-120b"


def _groq_key() -> str:
    raw = os.environ.get("GROQ_API_KEYS", "")
    if raw:
        return raw.split(",")[0].strip()
    return ""


async def extract_content(
    url: str,
    instruction: str | None = None,
    strategy: str = "llm",
    fields: list[str] | None = None,
    provider: str = _DEFAULT_LLM,
    chunk_threshold: int = 2000,
    max_pages: int = 1,
    css_selector: str | None = None,
) -> dict:
    from security import SecurityError, validate_url as _validate_url
    try:
        url = await _validate_url(url)
    except SecurityError as e:
        return {"success": False, "url": url, "error": str(e)}
    try:
        extraction_strategy = None

        if strategy == "llm":
            groq_key = _groq_key()
            if not groq_key:
                return {"success": False, "error": "No GROQ_API_KEYS configured for LLM extraction"}
            instruction_text = instruction or "Extract all important content from this page as structured JSON"
            extraction_strategy = LLMExtractionStrategy(
                llm_config=LLMConfig(provider=provider, api_token=groq_key),
                instruction=instruction_text,
                extraction_type="block",
                chunk_token_threshold=chunk_threshold,
                apply_chunking=True,
                input_format="markdown",
            )

        elif strategy == "css":
            if fields:
                # Build simple field schema for common extraction patterns
                schema = {
                    "baseSelector": css_selector or "body",
                    "fields": [
                        {"name": f, "selector": _guess_selector(f), "type": "text"}
                        for f in fields
                    ],
                }
            else:
                schema = {"baseSelector": css_selector or "body", "fields": []}
            extraction_strategy = JsonCssExtractionStrategy(schema=schema)

        elif strategy == "regex":
            extraction_strategy = RegexExtractionStrategy()

        else:
            return {"success": False, "error": f"Unknown strategy: {strategy}"}

        # Page HTML via our own pipeline (shares Helium CDP + cache +
        # SSRF gates with fetch) instead of a second Chromium.
        from fetch import fetch_url
        fetched = await fetch_url(url, max_chars=100000, output_format="html",
                                  include_tables=True, cache_ttl=3600)
        if not fetched.get("success"):
            return {"success": False, "url": url,
                    "error": fetched.get("error", "page fetch failed")}
        page_html = fetched.get("content", "") or ""
        if not page_html.strip():
            return {"success": False, "url": url, "error": "empty page HTML"}

        # Strategy .run() is SYNC (verified: not a coroutine) — call
        # directly, never await (awaiting a list = "can't be awaited").
        if strategy == "llm":
            # LLM strategy still needs markdown input for chunking — derive
            # it from the same fetch (no second request).
            md_fetch = await fetch_url(url, max_chars=100000,
                                       output_format="markdown",
                                       include_tables=True, cache_ttl=3600)
            page_md = md_fetch.get("content", "") if md_fetch.get("success") else ""
            sections = [page_md[i:i + chunk_threshold * 4]
                        for i in range(0, len(page_md), chunk_threshold * 4)] or [page_md]
            raw = extraction_strategy.run(url, sections)
        else:
            raw = extraction_strategy.run(url, [page_html])

        output: dict[str, object] = {
            "url": url,
            "success": True,
        }

        # Extract the structured data
        if raw:
            try:
                data = raw[0] if isinstance(raw, list) and len(raw) == 1 else raw
                data = json.loads(data) if isinstance(data, str) else data
                output["data"] = data
            except (json.JSONDecodeError, TypeError):
                s = str(raw)
                output["data"] = s[:10000] + ("\n\n[... truncated ...]" if len(s) > 10000 else "")

        # Include markdown if short (reuse the fetched HTML->text; no crawler)
        if strategy != "llm":
            text_fetch = await fetch_url(url, max_chars=5000,
                                         output_format="markdown",
                                         cache_ttl=3600)
            if text_fetch.get("success") and text_fetch.get("content"):
                text = text_fetch["content"]
                output["markdown"] = text[:5000] + ("\n\n[... truncated ...]" if len(text) > 5000 else "")

        return output

    except Exception as e:
        return {"success": False, "error": str(e)}


def _guess_selector(field: str) -> str:
    """Guess a reasonable CSS selector from a field name."""
    name = field.lower().replace(" ", "-").replace("_", "-")
    return (
        f"[class*='{name}'], [id*='{name}'], .{name}, #{name}, "
        f"[class*='{field.lower()}']"
    )
