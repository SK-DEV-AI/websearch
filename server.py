"""Unified web MCP server — multi-engine search + stealth scraping + content extraction."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from typing import Any

logger = logging.getLogger("websearch")

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from config import MAX_RESULTS, HELIUM_CDP, get_http_client, _KeyRotator
from search_ddg import ddgs_extract
from fetch import fetch_url, scrapling_stealthy_fetch
from crawl import crawl_url
from pdf_extract import extract_pdf
from screenshot import cdpa11y_snapshot, screenshot_cdp
from wikipedia import (search_wikipedia, fetch_wikipedia_summary, fetch_wikipedia_summary_rest,
                       fetch_wikipedia_categories, fetch_wikipedia_links, fetch_wikipedia_extlinks,
                       fetch_wikipedia_pageviews, fetch_wikipedia_revisions,
                       search_wikipedia_category, search_wikipedia_backlinks,
                       search_wikipedia_geosearch, search_wikipedia_random,
                       search_wikipedia_recentchanges, fetch_wikipedia_langlinks,
                       search_wikipedia_allpages)
from arxiv import search_arxiv
from site_mapper import map_site
from research import search_multi, enrich
from extract import extract_content

INSTRUCTIONS = """# WebSearch MCP

Multi-engine search + content extraction.

## When to use what

- **search** for finding information (9 engines, dedup+rerank+synthesis). **synthesize=false** when you only need raw results. Set `domain` for vertical search (code, academic, finance, health, travel, legal).
- **fetch** to read a specific page (auto-CDP fallback on blocked pages. PDF/EPUB/DOCX support). **ddgs_extract** for a fast lightweight skim.
- **screenshot type=snapshot** for LLM-readable page text. **type=screenshot** for visual capture.
- **extract** for structured JSON: **strategy=css** fastest+free (simple fields), **strategy=llm** works on any content (costs quota).
- **crawl** for site exploration (BFS/DFS). **map_site** for sitemap discovery.
- **pdf_extract** for PDFs (OCR+formulas+tables). **wikipedia/arxiv** for those dedicated sources.

## Pipeline (search)

search → multiple engines → dedup → reranker → synthesis (Groq for top 3).
depth≥2: fetches full pages for better scoring.

## Params

- `language`: Wikipedia/arXiv/DDG content language
- `domain`/`anysearch_tag`/`anysearch_zone`: AnySearch vertical routing
- `start_date`/`end_date`: Tavily time range
- `upload_urls`: GAI file (.avif.bmp.heic.heif.jpeg.pdf.png.webp, 10MB, 1/call)
- `raw=true`: fetch raw file URLs
- `safesearch`: DDG content filter
- `cache_ttl=0`: bypass cache
"""

server = Server("websearch", instructions=INSTRUCTIONS)

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(name="ping",
            description="Lightweight connectivity check — verifies internet and key search endpoints are reachable. Use before expensive calls when connectivity is uncertain. No params needed. e.g. ping()",
            inputSchema={"type": "object", "properties": {}},
        ),
        Tool(name="search",
            description="Multi-engine web search with dedup, reranking, and synthesis. depth=1 returns snippets with relevance_score+fetch_relevance per result. depth>=2 fetches full pages + re-ranks. synthesize=True (default) returns Groq answer with [N] citations. google_ai_only skips all engines for Google AI Mode. e.g. search(query='latest AI models', depth=1)",
            inputSchema={"type": "object", "properties": {
                "query": {"type": "string"}, "count": {"type": "integer", "default": 10},
                "depth": {"type": "integer", "default": 1, "description": "1=snippets, 2+=fetch full pages + rerank"},
                "search_type": {"type": "string", "enum": ["auto","text","news","images","videos","books"]},
                "timelimit": {"type": "string", "description": "Time filter: d/w/m/y"},
                "safesearch": {"type": "string", "enum": ["off","moderate","strict"], "default": "moderate"},
                "language": {"type": "string", "default": "en", "description": "Content language for Wikipedia/summaries"},
                "upload_urls": {"type": "array", "items": {"type": "string"}, "description": "GAI file upload: supported formats .avif .bmp .heic .heif .jpeg .pdf .png .webp. 10MB max. Only one file per call (GAI drops all but the last). Local: file:///path or remote URL."},
                "start_date": {"type": "string", "description": "Tavily date filter start (YYYY-MM-DD)"},
                "end_date": {"type": "string", "description": "Tavily date filter end (YYYY-MM-DD)"},
                "synthesize": {"type": "boolean", "default": True, "description": "Groq-synthesize top results into a concise answer with citations"},
                "domain": {"type": "string", "description": "AnySearch vertical: finance, code, academic, health, travel, legal, security. Guessed from query via simple heuristic — may be wrong, omit for general search."},
                "anysearch_tag": {"type": "string", "description": "AnySearch precise sub-domain tag in {domain}.{sub_domain} format (e.g. code.doc, finance.us_stock). Overrides domain."},
                "anysearch_zone": {"type": "string", "enum": ["", "cn", "intl"], "description": "AnySearch geo zone (cn or intl)"},
                "anysearch_language": {"type": "string", "description": "AnySearch content language (e.g. en, zh-CN)"},
                "google_ai_only": {"type": "boolean", "description": "Skip all other search engines, only use Google AI Mode for an AI-generated answer"}},
                "required": ["query"]}),
         Tool(name="fetch",
            description="URL to markdown/text. Auto-fallback: direct fetch → CDP for blocked/JS pages. Supports PDF, EPUB, DOCX. SSRF-protected. Use focus=\"query\" to filter content by relevance. Paginated via offset (response: next_offset). Results cached 1h; cache_ttl=0 fresh. For structured JSON extraction use `extract` instead. e.g. fetch(url='https://example.com')",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string"}, "max_chars": {"type": "integer", "default": 5000, "description": "Chars to return per call (for pagination)"},
                "offset": {"type": "integer", "default": 0, "description": "Char offset for paginated reads (0 = start). Response includes is_truncated, next_offset, total_extracted_chars (full page size)."},
                "focus": {"type": "string", "description": "BM25 relevance filter — extract only content blocks relevant to this query. Runs on cached content too."},
                "cache_ttl": {"type": "integer", "default": 3600, "description": "Cache TTL in seconds (0 = force fresh fetch). Cache keyed by URL+extraction_type+css_selector, not focus/offset."},
                "css_selector": {"type": "string"},
                "extraction_type": {"type": "string", "enum": ["markdown","text","html"]},
                "target_language": {"type": "string"},
                "output_format": {"type": "string", "enum": ["markdown","txt","json","xml","csv"], "default": "markdown"},
                "fast": {"type": "boolean"},
                "raw": {"type": "boolean", "description": "Skip CDP/trafilatura, return raw text directly (use for GitHub raw files, pastebin, etc.)"},
                "network_idle": {"type": "boolean", "default": True, "description": "Wait for network idle before extracting (slower but captures JS-rendered content)"},
                   "include_images": {"type": "boolean", "default": True, "description": "Include image captions/alt text"},
                   "include_links": {"type": "boolean", "default": True, "description": "Include hyperlinks in output"},
                   "include_formatting": {"type": "boolean", "default": True, "description": "Preserve text formatting (bold, italic, etc)"},
                   "include_tables": {"type": "boolean", "default": True, "description": "Extract tables from HTML"},
                   "start_line": {"type": "integer", "description": "1-based start line for reading a range (slices content by newline)"},
                   "end_line": {"type": "integer", "description": "1-based end line (inclusive). Use with start_line for targeted reading."},
                   },
                "required": ["url"]}),
        Tool(name="crawl",
            description="BFS/DFS deep crawl. Returns per-page markdown + combined text. e.g. crawl(url='https://example.com', max_depth=2)",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string"}, "max_depth": {"type": "integer", "default": 1},
                "max_pages": {"type": "integer", "default": 10},
                "extract_links": {"type": "boolean", "default": True},
                "strategy": {"type": "string", "enum": ["bfs","dfs"], "default": "bfs"},
                "exclude_domains": {"type": "array", "items": {"type": "string"}},
                "content_filter": {"type": "string", "enum": ["","pruning","bm25","bm25_hq","cosine"]},
                "filter_query": {"type": "string"},
                "page_timeout": {"type": "integer", "default": 60000, "description": "Page load timeout in ms"},
                "check_robots_txt": {"type": "boolean", "default": False},
                "capture_console_messages": {"type": "boolean", "default": False, "description": "Capture console.log output"},
                "capture_network_requests": {"type": "boolean", "default": False},
                "css_selector": {"type": "string", "description": "CSS selector to target specific content"},
                "bypass_cache": {"type": "boolean", "default": False, "description": "Force fresh crawl, skip cache"},
                "exclude_all_images": {"type": "boolean", "default": False},
                "exclude_external_images": {"type": "boolean", "default": False}},
                "required": ["url"]}),
         Tool(name="screenshot",
            description="Screenshot or ARIA accessibility snapshot (AI-optimized for LLMs). Use type=snapshot for LLM-readable text, type=screenshot for visual capture. start_line/end_line for range reads. e.g. screenshot(url='https://example.com', type='snapshot')",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string"}, "full_page": {"type": "boolean", "default": True},
                "type": {"type": "string", "enum": ["screenshot","snapshot","both"]},
                "max_chars": {"type": "integer", "default": 10000},
                "quality": {"type": "integer", "description": "JPEG quality 1-100"},
                "image_type": {"type": "string", "enum": ["png", "jpeg"], "default": "png", "description": "Screenshot image format"},
                "clip_x": {"type": "number", "description": "Clip region X offset for screenshot"},
                "clip_y": {"type": "number", "description": "Clip region Y offset for screenshot"},
                "clip_width": {"type": "number", "description": "Clip region width for screenshot"},
                "clip_height": {"type": "number", "description": "Clip region height for screenshot"},
                "scale": {"type": "string", "enum": ["css", "device"], "default": "css", "description": "Screenshot scale: css (default DPR) or device (device DPR)"},
                "animations": {"type": "string", "enum": ["allow", "disabled"], "default": "allow", "description": "Whether to animate elements in screenshot"},
                "omit_background": {"type": "boolean", "default": False, "description": "Transparent background (PNG only)"},
                "caret": {"type": "string", "enum": ["hide", "initial"], "default": "initial", "description": "Whether to hide the caret before screenshot"},
                "depth": {"type": "integer", "description": "ARIA snapshot tree depth limit"},
                "boxes": {"type": "boolean", "default": False, "description": "Include bounding boxes in ARIA snapshot"},
                "verbose": {"type": "boolean", "default": False, "description": "Show all ARIA roles (not just interactive)"},
                "start_line": {"type": "integer", "description": "1-based start line for snapshot text range"},
                "end_line": {"type": "integer", "description": "1-based end line (inclusive) for snapshot text range"}},
                "required": ["url"]}),
         Tool(name="wikipedia",
            description="Search Wikipedia: articles, summaries, geosearch, random. Actions: search, summary (REST API v1 fast), summary_action (Action API with images/sections), categories, links, extlinks, categorymembers, pageviews, revisions, backlinks, recentchanges. e.g. wikipedia(query='Python', action='summary')",
            inputSchema={"type": "object", "properties": {
                "action": {"type": "string", "enum": ["search","summary","summary_action","geosearch","random","categories","links","extlinks","categorymembers","pageviews","revisions","backlinks","recentchanges","langlinks","allpages"], "default": "search", "description": "search=find articles, summary=REST fast extract, summary_action=Action API+images, categories=list page cats, links=page links, extlinks=external links, categorymembers=pages in cat, pageviews=traffic stats, revisions=edit history, backlinks=what links here, recentchanges=recent edits, geosearch=near coordinates, random=random pages, langlinks=cross-lang links, allpages=list all pages"},
                "query": {"type": "string"},
                "count": {"type": "integer", "default": 3},
                "language": {"type": "string", "default": "en"},
                "lat": {"type": "number"}, "lon": {"type": "number"},
                "category": {"type": "string", "description": "Category name for categorymembers action (without Category: prefix)"},
                "namespace": {"type": "integer", "default": 0, "description": "Namespace filter (0=articles, 14=categories). Applies to search/links/allpages/random actions."},
                "days": {"type": "integer", "default": 30, "description": "Days of pageview history (pageviews action)"},
                "type_filter": {"type": "string", "description": "Change type filter for recentchanges: edit/new/move/log/categorize"},
                "include_images": {"type": "boolean", "default": False, "description": "Include thumbnail images in summary_action"},
                "distance": {"type": "integer", "default": 1000, "description": "Search radius in meters for geosearch"}},
                "required": []}),
        Tool(name="arxiv",
            description="Search arXiv academic papers. Use raw_query for boolean operators (AND, OR, ANDNOT), phrase search (ti:\"exact phrase\"), wildcards (au:smith*). e.g. arxiv(query='cs.AI transformer', count=5)",
            inputSchema={"type": "object", "properties": {
                "query": {"type": "string"}, "count": {"type": "integer", "default": 3},
                "search_field": {"type": "string", "enum": ["all","ti","au","abs","cat","co","jr","id"], "default": "all"},
                "sort_by": {"type": "string", "enum": ["relevance","lastUpdatedDate","submittedDate"], "default": "relevance"},
                "sort_order": {"type": "string", "enum": ["ascending","descending"], "default": "descending"},
                "start": {"type": "integer", "default": 0},
                "id_list": {"type": "string", "description": "Comma-delimited arXiv IDs"},
                "category": {"type": "string", "description": "arXiv category filter (e.g. cs.AI, math.CO)"},
                "raw_query": {"type": "string", "description": "Raw arXiv search_query syntax with boolean operators (AND/OR/ANDNOT), phrase, wildcards. Overrides query+search_field."}},
                "required": ["query"]}),
        Tool(name="map_site",
            description="Discover all pages on a website via sitemap XML (primary) and HTML link extraction (fallback). Returns the base domain, source type, total count, and an array of discovered URLs with metadata (last_modified, priority, changefreq when available). e.g. map_site(url='https://example.com')",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string", "description": "Full URL of the site to map (e.g. https://example.com)"},
                "max_urls": {"type": "integer", "default": 1000, "description": "Cap on returned URLs"},
                "max_depth": {"type": "integer", "default": 0, "description": "Recursive link extraction depth (0=homepage only). Only used when no sitemap exists."},
                "include_sitemap": {"type": "boolean", "default": True, "description": "Try sitemap discovery first"},
                "include_links": {"type": "boolean", "default": True, "description": "Fall back to HTML link extraction when no sitemap"},
                "same_domain": {"type": "boolean", "default": True, "description": "Only include URLs from the same domain"},
                "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "Regex patterns to exclude matching URLs"}},
                "required": ["url"]}),
         Tool(name="ddgs_extract",
            description="Lightweight URL content extraction via DuckDuckGo's extract endpoint. Faster than fetch for simple pages — markdown or plain text. Best for search snippets and quick page reads where trafilatura is overkill. e.g. ddgs_extract(url='https://example.com')",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string"}, "extract_type": {"type": "string", "enum": ["markdown","text_plain","raw"], "default": "markdown"}},
                "required": ["url"]}),
         Tool(name="extract",
            description="Extract structured JSON from a webpage using LLM, CSS, or regex strategies. Persistent cache — repeat calls free. For LLM strategy, describe what you want and get clean JSON back. For CSS strategy, provide field names (fastest, free). For raw page content, use `fetch` instead. Examples: extract(url='...', instruction='extract product name and price') or extract(url='...', fields=['name','price'], strategy='css')",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string", "description": "Target URL to extract data from"},
                "instruction": {"type": "string", "description": "Natural language extraction instruction (used with strategy=llm). Example: 'extract all product names, prices, and ratings from this page'"},
                "strategy": {"type": "string", "enum": ["llm", "css", "regex"], "default": "css", "description": "css=CSS-selector-based extraction (fast, free), llm=AI-powered (costs quota), regex=pattern-based (emails, phones, URLs)"},
                "fields": {"type": "array", "items": {"type": "string"}, "description": "List of field names for extraction (e.g. ['name', 'price', 'rating']). Used with strategy=css or strategy=llm"},
                "chunk_threshold": {"type": "integer", "default": 2000, "description": "Max tokens per chunk for LLM extraction (lower = cheaper, higher = more context)"},
                "css_selector": {"type": "string", "description": "CSS selector for the container element (used with strategy=css). Defaults to 'body'"},
                "provider": {"type": "string", "default": "groq/openai/gpt-oss-120b", "description": "LLM provider string in LiteLLM format (e.g. groq/openai/gpt-oss-120b, openai/gpt-4o, ollama/llama2)"}},
                "required": ["url"]}),
         Tool(name="pdf_extract",
            description="PDF to structured data (text, tables, formulas, images with bounding boxes). Supports scanned PDFs (OCR), complex tables. e.g. pdf_extract(input_path='/path/to/doc.pdf', format='markdown')",
            inputSchema={"type": "object", "properties": {
                "input_path": {"type": "array", "items": {"type": "string"}, "description": "PDF file paths or URLs (local files, http/https, file://)"},
                "format": {"type": "string", "enum": ["markdown","json","html","tagged-pdf","markdown,json","markdown,json,html"], "default": "markdown"},
                "password": {"type": "string", "description": "PDF password for protected files"},
                "pages": {"type": "string", "description": "Page range e.g. 1-5,8,10-12"},
                "hybrid": {"type": "string", "enum": ["", "docling-fast", "docling-enterprise", "marker"], "description": "AI hybrid mode for complex layouts, scanned PDFs, tables"},
            },
                "required": ["input_path"]}),
    ]


@server.call_tool()
async def handle_call_tool(name: str, arguments: dict) -> CallToolResult:
    if not isinstance(arguments, dict):
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": "arguments must be a dict"}))], isError=True)

    def safe_int(v, default=0):
        try:
            return int(v)
        except (TypeError, ValueError):
            return default

    def safe_float(v, default=0.0):
        try:
            return float(v)
        except (TypeError, ValueError):
            return default

    def _res(data) -> CallToolResult:
        ok = isinstance(data, dict) and data.get("success", False)
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(data, default=str))], isError=not ok)

    try:
        if name == "ping":
            c = get_http_client()
            results = {}
            for target, url in [("cloudflare", "https://1.1.1.1"), ("google", "https://www.google.com"), ("archive", "https://archive.org")]:
                try:
                    r = await c.get(url, timeout=5)
                    results[target] = {"reachable": True, "status": r.status_code, "ms": int(r.elapsed * 1000)}
                except Exception as e:
                    results[target] = {"reachable": False, "error": str(e)[:60]}
            return _res({"success": True, "connectivity": results})

        if name == "search":
            query = arguments["query"]
            count = min(safe_int(arguments.get("count",10)), MAX_RESULTS)
            depth = safe_int(arguments.get("depth",1))
            lang = str(arguments.get("language","en"))
            google_ai_only = bool(arguments.get("google_ai_only", False))
            r = await search_multi(query, count=max(count, depth * 3),
                google_ai_only=google_ai_only,
                search_type=str(arguments.get("search_type","auto")),
                timelimit=str(arguments.get("timelimit","")),
                safesearch=str(arguments.get("safesearch","moderate")),
                language=lang,
                upload_urls=arguments.get("upload_urls"),
                start_date=str(arguments.get("start_date","")),
                end_date=str(arguments.get("end_date","")),
                domain=str(arguments.get("domain","")),
                anysearch_tag=str(arguments.get("anysearch_tag","")),
                anysearch_zone=str(arguments.get("anysearch_zone","")),
                anysearch_language=str(arguments.get("anysearch_language","")),
                cdp_url=HELIUM_CDP)
            if r.get("success") and depth >= 2 and r.get("results"):
                try:
                    fetched = await enrich(r["results"], query, depth=depth,
                        cdp_url=HELIUM_CDP, count=count, language=lang)
                    if fetched.get("fetched_content"):
                        r["fetched_content"] = fetched["fetched_content"]
                except Exception as e:
                    logger.warning("enrich failed: %s", e)
            # Skip Groq synthesis when GAI already returned a full answer
            skip_synthesis = google_ai_only and r.get("ai_answer")
            if r.get("success") and bool(arguments.get("synthesize", True)) and r.get("results") and not skip_synthesis:
                try:
                    top = r["results"][:3]
                    ctx = "\n\n".join(f"[{i+1}] {x.get('title','')}: {x.get('content','')[:400]}"
                                     for i, x in enumerate(top))
                    _groq_keys = _KeyRotator("GROQ_API_KEYS")
                    if _groq_keys.has_keys:
                        c = get_http_client()
                        resp = await c.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {_groq_keys.next()}", "Content-Type": "application/json"},
                            json={"model": "openai/gpt-oss-120b",
                                  "messages": [{"role": "system", "content": "Answer concisely from sources. Use [N] citations like [1][2]."},
                                               {"role": "user", "content": f"Query: {query}\n\nResults:\n{ctx}"}],
                                  "temperature": 0.3, "max_tokens": 256}, timeout=15)
                        if resp.status_code == 200:
                            r["synthesis"] = {"answer": resp.json()["choices"][0]["message"]["content"].strip()}
                except Exception as e:
                    r["synthesis_error"] = str(e)
            return _res(r)
        elif name == "fetch":
            url = arguments["url"]
            offset = safe_int(arguments.get("offset", 0))
            max_chars = safe_int(arguments.get("max_chars", 5000))

            r = await fetch_url(url,
                    max_chars=max_chars,
                    offset=offset,
                    focus=str(arguments.get("focus", "")),
                    cache_ttl=safe_int(arguments.get("cache_ttl", 3600)),
                    target_language=str(arguments.get("target_language", "")),
                    fast=bool(arguments.get("fast", False)),
                    output_format=str(arguments.get("output_format", "markdown")),
                    include_images=bool(arguments.get("include_images", True)),
                    include_tables=bool(arguments.get("include_tables", True)),
                    include_formatting=bool(arguments.get("include_formatting", True)),
                    include_links=bool(arguments.get("include_links", True)),
                    raw=bool(arguments.get("raw", False)))

            # Auto-fallback: httpx failed/blocked → retry via CDP
            if not r.get("success"):
                content = (r.get("content", "") or "").strip()
                status = r.get("status", 0)
                r_error = (r.get("error", "") or "").lower()
                content_lower = content.lower()
                cf_hits = sum(1 for kw in ["just a moment", "checking your browser",
                               "cf-challenge", "__cf_chl_", "cf-turnstile",
                               "verify you are human", "attention required"]
                              if kw in content_lower[:800])
                should_retry = (
                    "cloudflare" in r_error
                ) or status in (403, 429, 503) or cf_hits >= 2 or (
                    len(content) < 300 and (
                        "blocked" in content_lower or "access denied" in content_lower
                        or "network security" in content_lower or "rate limit" in content_lower
                        or "too many requests" in content_lower
                    )
                ) or (len(content) < 100)

                if should_retry:
                    cdp_r = await scrapling_stealthy_fetch(url,
                        css_selector=arguments.get("css_selector"),
                        extraction_type=str(arguments.get("extraction_type", "markdown")),
                        cdp_url=HELIUM_CDP, network_idle=bool(arguments.get("network_idle", True)))
                    if cdp_r.get("success"):
                        r = cdp_r
                        # Re-apply focus on CDP result if we had one
                        focus_q = str(arguments.get("focus", ""))
                        if focus_q and r.get("content"):
                            from focus import filter_by_relevance as _focus_filter
                            r["content"] = _focus_filter(r["content"], focus_q)

            # Backward compat: start_line/end_line line-range slicing
            start_line = safe_int(arguments.get("start_line", 0))
            end_line = safe_int(arguments.get("end_line", 0))
            if start_line > 0 and r.get("content"):
                all_lines = r["content"].split("\n")
                r["total_lines"] = len(all_lines)
                if end_line > 0:
                    r["content"] = "\n".join(all_lines[start_line - 1:end_line])
                else:
                    r["content"] = "\n".join(all_lines[start_line - 1:])
                r["returned_lines"] = r["content"].count("\n") + 1
            return _res(r)
        elif name == "crawl":
            r = await crawl_url(arguments["url"],
                max_depth=safe_int(arguments.get("max_depth",1)),
                max_pages=safe_int(arguments.get("max_pages",10)),
                extract_links=bool(arguments.get("extract_links",True)),
                strategy=str(arguments.get("strategy","bfs")),
                css_selector=str(arguments.get("css_selector","")),
                exclude_domains=arguments.get("exclude_domains"),
                content_filter=str(arguments.get("content_filter","")),
                filter_query=str(arguments.get("filter_query","")),
                page_timeout=safe_int(arguments.get("page_timeout",60000)),
                check_robots_txt=bool(arguments.get("check_robots_txt",False)),
                capture_console_messages=bool(arguments.get("capture_console_messages",False)),
                capture_network_requests=bool(arguments.get("capture_network_requests",False)),
                bypass_cache=bool(arguments.get("bypass_cache",False)),
                exclude_all_images=bool(arguments.get("exclude_all_images",False)),
                exclude_external_images=bool(arguments.get("exclude_external_images",False)))
            return _res(r)
        elif name == "screenshot":
            url = arguments["url"]
            cap_type = str(arguments.get("type","screenshot"))
            full = bool(arguments.get("full_page",True))
            snap = None
            ss = None
            if cap_type in ("snapshot","both"):
                snap = await cdpa11y_snapshot(url, verbose=bool(arguments.get("verbose",False)),
                    max_chars=safe_int(arguments.get("max_chars",10000)),
                    depth=arguments.get("depth") or 5,
                    boxes=bool(arguments.get("boxes",False)))
            if cap_type in ("screenshot","both"):
                ss = await screenshot_cdp(url, full_page=full,
                    clip_x=safe_float(arguments.get("clip_x",0)),
                    clip_y=safe_float(arguments.get("clip_y",0)),
                    clip_width=safe_float(arguments.get("clip_width",0)),
                    clip_height=safe_float(arguments.get("clip_height",0)),
                    scale=str(arguments.get("scale","css")),
                    animations=str(arguments.get("animations","allow")),
                    quality=arguments.get("quality"),
                    image_type=str(arguments.get("image_type","png")),
                    omit_background=bool(arguments.get("omit_background",False)),
                    caret=str(arguments.get("caret","initial")))
            if cap_type == "snapshot":
                r = snap
            elif cap_type == "both":
                r = {"screenshot": ss, "snapshot": snap}
            else:
                r = ss
            # Line range slicing for snapshot content
            if isinstance(r, dict) and r.get("content"):
                all_lines = r["content"].split("\n")
                r["total_lines"] = len(all_lines)
                r["total_chars"] = len(r["content"])
                start_line = safe_int(arguments.get("start_line", 0))
                end_line = safe_int(arguments.get("end_line", 0))
                if start_line > 0:
                    if end_line > 0:
                        r["content"] = "\n".join(all_lines[start_line - 1:end_line])
                    else:
                        r["content"] = "\n".join(all_lines[start_line - 1:])
                    r["returned_lines"] = r["content"].count("\n") + 1
            return _res(r)
        elif name == "wikipedia":
            action = str(arguments.get("action", "search"))
            lang = str(arguments.get("language", "en"))
            q = str(arguments.get("query", ""))
            cnt = safe_int(arguments.get("count", 3))
            if action == "summary":
                r = await fetch_wikipedia_summary_rest(title=q, language=lang)
                return _res({"success": True, "result": r} if r else {"success": False, "error": "Not found"})
            if action == "summary_action":
                r = await fetch_wikipedia_summary(query=q, language=lang,
                    include_images=bool(arguments.get("include_images",False)))
                return _res({"success": True, "result": r} if r else {"success": False, "error": "Not found"})
            if action == "categories":
                r = await fetch_wikipedia_categories(title=q, language=lang)
                return _res({"success": True, "results": r})
            if action == "links":
                r = await fetch_wikipedia_links(title=q, language=lang,
                    namespace=safe_int(arguments.get("namespace", 0)), count=cnt)
                return _res({"success": True, "results": r})
            if action == "extlinks":
                r = await fetch_wikipedia_extlinks(title=q, language=lang, count=cnt)
                return _res({"success": True, "results": r})
            if action == "categorymembers":
                r = await search_wikipedia_category(category=str(arguments.get("category", q)),
                    language=lang, count=cnt)
                return _res({"success": True, "results": r})
            if action == "pageviews":
                r = await fetch_wikipedia_pageviews(title=q, language=lang,
                    days=safe_int(arguments.get("days", 30)))
                return _res({"success": True, "results": r})
            if action == "revisions":
                r = await fetch_wikipedia_revisions(title=q, language=lang, count=cnt)
                return _res({"success": True, "results": r})
            if action == "backlinks":
                r = await search_wikipedia_backlinks(title=q, language=lang, count=cnt)
                return _res({"success": True, "results": r})
            if action == "recentchanges":
                r = await search_wikipedia_recentchanges(language=lang, count=cnt,
                    type_filter=str(arguments.get("type_filter","")))
                return _res({"success": True, "results": r})
            if action == "geosearch":
                r = await search_wikipedia_geosearch(
                    lat=float(arguments.get("lat",0)), lon=float(arguments.get("lon",0)),
                    distance=safe_int(arguments.get("distance",1000)),
                    count=safe_int(arguments.get("count",10)), language=str(arguments.get("language","en")))
                return _res({"success": True, "results": r})
            if action == "random":
                r = await search_wikipedia_random(count=safe_int(arguments.get("count",5)),
                    language=str(arguments.get("language","en")))
                return _res({"success": True, "results": r})
            if action == "langlinks":
                r = await fetch_wikipedia_langlinks(title=q, language=lang, count=cnt)
                return _res({"success": True, "results": r})
            if action == "allpages":
                r = await search_wikipedia_allpages(
                    namespace=safe_int(arguments.get("namespace", 0)),
                    limit=safe_int(arguments.get("count", 50)),
                    language=lang)
                return _res({"success": True, "results": r})
            r = await search_wikipedia(query=q,
                count=cnt, language=lang,
                namespace=safe_int(arguments.get("namespace", 0)))
            return _res({"success": True, "results": r})
        elif name == "arxiv":
            r = await search_arxiv(query=str(arguments.get("query","")),
                count=safe_int(arguments.get("count",3)),
                search_field=str(arguments.get("search_field","all")),
                sort_by=str(arguments.get("sort_by","relevance")),
                sort_order=str(arguments.get("sort_order","descending")),
                start=safe_int(arguments.get("start",0)),
                id_list=str(arguments.get("id_list","")),
                category=str(arguments.get("category","")),
                raw_query=str(arguments.get("raw_query","")))
            return _res({"success": True, "results": r})

        elif name == "ddgs_extract":
            r = await ddgs_extract(url=str(arguments.get("url", "")),
                extract_type=str(arguments.get("extract_type", "markdown")))
            return _res({"success": True, "result": r})
        elif name == "extract":
            r = await extract_content(
                url=str(arguments["url"]),
                instruction=str(arguments.get("instruction") or "").strip() or None,
                strategy=str(arguments.get("strategy", "llm")),
                fields=arguments.get("fields"),
                chunk_threshold=safe_int(arguments.get("chunk_threshold", 2000)),
                css_selector=str(arguments.get("css_selector") or "").strip() or None,
                provider=str(arguments.get("provider", "")).strip() or None,
            )
            return _res(r)
        elif name == "pdf_extract":
            paths = arguments.get("input_path", [])
            if isinstance(paths, str):
                paths = [paths]
            hy = str(arguments.get("hybrid", ""))
            hm = str(arguments.get("hybrid_mode", ""))
            if bool(arguments.get("force_ocr", False)):
                hy = hy or "docling-fast"
                hm = hm or "full"
            if bool(arguments.get("enrich_formula", False)):
                hy = hy or "docling-fast"
                hm = "full"
            if bool(arguments.get("enrich_picture", False)):
                hy = hy or "docling-fast"
                hm = "full"
            r = await extract_pdf(paths,
                format=str(arguments.get("format", "markdown")),
                password=str(arguments.get("password", "")),
                pages=str(arguments.get("pages", "")),
                hybrid=hy, hybrid_mode=hm,
                hybrid_url=str(arguments.get("hybrid_url", "")),
                hybrid_timeout=str(arguments.get("hybrid_timeout", "")),
                table_method=str(arguments.get("table_method", "")),
                reading_order=str(arguments.get("reading_order", "")),
                image_output=str(arguments.get("image_output", "")),
                image_format=str(arguments.get("image_format", "")),
                sanitize=bool(arguments.get("sanitize", False)),
                keep_line_breaks=bool(arguments.get("keep_line_breaks", False)),
                markdown_with_html=bool(arguments.get("markdown_with_html", False)),
                include_header_footer=bool(arguments.get("include_header_footer", False)),
                detect_strikethrough=bool(arguments.get("detect_strikethrough", False)),
                use_struct_tree=bool(arguments.get("use_struct_tree", False)),
                content_safety_off=str(arguments.get("content_safety_off", "")),
                threads=str(arguments.get("threads", "")),
                replace_invalid_chars=str(arguments.get("replace_invalid_chars", "")))
            return _res(r)
        elif name == "map_site":
            r = await map_site(url=str(arguments.get("url","")),
                max_urls=safe_int(arguments.get("max_urls",1000)),
                max_depth=safe_int(arguments.get("max_depth",0)),
                include_sitemap=bool(arguments.get("include_sitemap",True)),
                include_links=bool(arguments.get("include_links",True)),
                same_domain=bool(arguments.get("same_domain",True)),
                exclude_patterns=arguments.get("exclude_patterns"))
            return _res(r)
        else:
            return CallToolResult(content=[TextContent(type="text", text=f"Unknown tool: {name}")], isError=True)
    except ValueError as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": str(e)}))], isError=True)
    except KeyError as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": f"Missing required argument: {e}"}))], isError=True)
    except TypeError as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": str(e)}))], isError=True)
    except RuntimeError as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": str(e)}))], isError=True)
    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps({"error": f"{type(e).__name__}: {e}"}))], isError=True)


async def _warmup_reranker():
    """Pre-load reranker model in background to avoid cold-start delay."""
    try:
        from reranker import warmup
        await warmup()
    except Exception:
        pass

async def _warmup_gai():
    """Pre-warm GAI CDP connection at server start."""
    try:
        from search_gai import get_gai_client
        await get_gai_client()
    except Exception:
        pass

async def main():
    asyncio.create_task(_warmup_reranker())
    asyncio.create_task(_warmup_gai())
    async with stdio_server() as (rs, ws):
        await server.run(rs, ws, server.create_initialization_options())

if __name__ == "__main__":
    asyncio.run(main())
