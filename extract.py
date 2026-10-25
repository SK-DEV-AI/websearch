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

# crawl4ai is an OPTIONAL dependency (pip install websearch-mcp[extract]); it
# pulls a browser stack. Import it lazily so the server starts without it and
# only the extraction strategies that need it fail with a clear message.
def _crawl4ai() -> tuple:
    """Import and return the crawl4ai strategy classes on first use."""
    try:
        from crawl4ai.extraction_strategy import (
            LLMExtractionStrategy,
            JsonCssExtractionStrategy,
            RegexExtractionStrategy,
        )
        from crawl4ai import LLMConfig
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError(
            "crawl4ai is not installed. It is an optional dependency of this "
            "MCP server. Install it with:  pip install websearch-mcp[extract]  "
            "(or websearch-mcp[all] for every extra)."
        ) from e
    return LLMExtractionStrategy, JsonCssExtractionStrategy, RegexExtractionStrategy, LLMConfig

from config import _KeyRotator

_groq_rotator = _KeyRotator("GROQ_API_KEYS")

_DEFAULT_LLM = "groq/openai/gpt-oss-120b"


async def _groq_key() -> str | None:
    return await _groq_rotator.next()


async def extract_content(
    url: str,
    instruction: str | None = None,
    strategy: str = "css",
    fields: list[str] | None = None,
    provider: str | None = _DEFAULT_LLM,
    chunk_threshold: int = 2000,
    max_pages: int = 1,
    css_selector: str | None = None,
) -> dict:
    from security import SecurityError, validate_url as _validate_url
    # A caller that omits `provider` sends None explicitly, which would
    # override the default and reach crawl4ai as provider=None ->
    # "'NoneType' object has no attribute 'startswith'". Fall back here.
    provider = (provider or "").strip() or _DEFAULT_LLM
    try:
        url = await _validate_url(url)
    except SecurityError as e:
        return {"success": False, "url": url, "error": str(e)}
    try:
        extraction_strategy = None

        (LLMExtractionStrategy, JsonCssExtractionStrategy,
         RegexExtractionStrategy, LLMConfig) = _crawl4ai()

        if strategy == "llm":
            groq_key = await _groq_key()
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
                if len(s) > 10000:
                    # A marker inside the string is easy to miss when the caller
                    # parses the field; a flag is not.
                    output["truncated"] = True
                    output["truncated_fields"] = ["data"]
                    output["truncated_at_chars"] = 10000
                output["data"] = s[:10000] + ("\n\n[... truncated ...]" if len(s) > 10000 else "")

        # Include markdown if short (reuse the fetched HTML->text; no crawler)
        if strategy != "llm":
            text_fetch = await fetch_url(url, max_chars=5000,
                                         output_format="markdown",
                                         cache_ttl=3600)
            if text_fetch.get("success") and text_fetch.get("content"):
                text = text_fetch["content"]
                if len(text) > 5000:
                    output["truncated"] = True
                    output.setdefault("truncated_fields", []).append("markdown")
                    output["truncated_at_chars"] = 5000
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
