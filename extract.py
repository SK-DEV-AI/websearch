"""Structured extraction using crawl4ai — LLM-powered + CSS-driven + regex extraction."""

from __future__ import annotations

import json
import os

from crawl4ai import AsyncWebCrawler, CrawlerRunConfig, LLMConfig
from crawl4ai.extraction_strategy import (
    LLMExtractionStrategy,
    JsonCssExtractionStrategy,
    RegexExtractionStrategy,
)

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

        config = CrawlerRunConfig(
            extraction_strategy=extraction_strategy,
            cache_mode=None,
            verbose=False,
            page_timeout=30000,
        )

        async with AsyncWebCrawler(verbose=False) as crawler:
            result = await crawler.arun(url=url, config=config)

        if not result.success:
            return {"success": False, "error": str(result.error_message or "crawl failed")}

        output: dict[str, object] = {
            "url": result.url,
            "success": True,
        }

        # Extract the structured data
        raw = result.extracted_content
        if raw:
            try:
                data = json.loads(raw) if isinstance(raw, str) else raw
                output["data"] = data
            except (json.JSONDecodeError, TypeError):
                output["data"] = str(raw)[:10000]

        # Include markdown if short
        if hasattr(result, "markdown") and result.markdown:
            md = result.markdown
            if hasattr(md, "raw_markdown"):
                text = md.raw_markdown
            else:
                text = str(md)
            output["markdown"] = text[:5000]

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
