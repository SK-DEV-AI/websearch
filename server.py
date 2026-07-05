"""Unified web MCP server — multi-engine search + stealth scraping + content extraction."""

from __future__ import annotations

import asyncio
import json
import os
from typing import Any

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, TextContent, Tool

from config import MAX_RESULTS, HELIUM_CDP
from search_ddg import search_ddg, ddgs_extract
from search_gai import GoogleAIClient, get_gai_client
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

server = Server("websearch")

@server.list_tools()
async def handle_list_tools() -> list[Tool]:
    return [
        Tool(name="search",
            description="Multi-engine web search with dedup and reranking. depth=1 returns snippets; depth>=2 fetches full pages + re-ranks. synthesize=True (default) returns a Groq-synthesized answer with [N] citations.",
            inputSchema={"type": "object", "properties": {
                "query": {"type": "string"}, "count": {"type": "integer", "default": 10},
                "depth": {"type": "integer", "default": 1, "description": "1=snippets, 2+=fetch full pages + rerank"},
                "search_type": {"type": "string", "enum": ["auto","text","news","images","videos","books"]},
                "timelimit": {"type": "string", "description": "Time filter: d/w/m/y"},
                "safesearch": {"type": "string", "enum": ["off","moderate","strict"], "default": "moderate"},
                "language": {"type": "string", "default": "en", "description": "Content language for Wikipedia/summaries"},
                "upload_urls": {"type": "array", "items": {"type": "string"}, "description": "Image/PDF URLs or local file paths for GAI"},
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
            description="URL to markdown/text. Auto-fallback: fast httpx+trafilatura first, then CDP for Cloudflare/JS-heavy pages. Supports PDF, EPUB, DOCX (default path only). Default: fast (domcontentloaded only). Use network_idle=True for JS-heavy pages. Use start_line/end_line for range reads instead of guessing max_chars.",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string"}, "max_chars": {"type": "integer", "default": 5000},
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
            description="BFS/DFS deep crawl. Returns per-page markdown + combined text.",
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
            description="CDP screenshot or ARIA accessibility snapshot (AI-optimized for LLMs). Use start_line/end_line for snapshot text range reads.",
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
            description="Search Wikipedia: articles, summaries, geosearch, random. Actions: search, summary (REST API v1 fast), summary_action (Action API with images/sections), categories, links, extlinks, categorymembers, pageviews, revisions, backlinks, recentchanges.",
            inputSchema={"type": "object", "properties": {
                "action": {"type": "string", "enum": ["search","summary","summary_action","geosearch","random","categories","links","extlinks","categorymembers","pageviews","revisions","backlinks","recentchanges","langlinks","allpages"], "default": "search"},
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
            description="Search arXiv academic papers. Use raw_query for boolean operators (AND, OR, ANDNOT), phrase search (ti:\"exact phrase\"), wildcards (au:smith*).",
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
            description="Discover all pages on a website via sitemap XML (primary) and HTML link extraction (fallback). Returns the base domain, source type, total count, and an array of discovered URLs with metadata (last_modified, priority, changefreq when available).",
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
            description="Lightweight URL content extraction via DuckDuckGo's extract endpoint. Faster than fetch for simple pages — markdown or plain text. Best for search snippets and quick page reads where trafilatura is overkill.",
            inputSchema={"type": "object", "properties": {
                "url": {"type": "string"}, "extract_type": {"type": "string", "enum": ["markdown","text_plain","raw"], "default": "markdown"}},
                "required": ["url"]}),
         Tool(name="pdf_extract",
            description="PDF to structured data via opendataloader-pdf. Extracts text, tables, formulas, images with bounding boxes. Supports scanned PDFs (OCR), complex tables, and accessibility tagging.",
            inputSchema={"type": "object", "properties": {
                "input_path": {"type": "array", "items": {"type": "string"}, "description": "PDF file paths or URLs (local files, http/https, file://)"},
                "format": {"type": "string", "enum": ["markdown","json","html","tagged-pdf","markdown,json","markdown,json,html"], "default": "markdown"},
                "password": {"type": "string", "description": "PDF password for protected files"},
                "pages": {"type": "string", "description": "Page range e.g. 1-5,8,10-12"},
                "hybrid": {"type": "string", "enum": ["", "docling-fast", "docling-enterprise", "marker"], "description": "AI hybrid mode for complex layouts, scanned PDFs, tables"},
                "hybrid_mode": {"type": "string", "enum": ["", "full"], "description": "full enables formula/picture enrichment (requires hybrid)"},
                "force_ocr": {"type": "boolean", "description": "Enable OCR for scanned/image-based PDFs (requires --hybrid docling-fast)"},
                "enrich_formula": {"type": "boolean", "description": "Extract mathematical formulas as LaTeX (requires hybrid_mode=full)"},
                "enrich_picture": {"type": "boolean", "description": "Generate AI descriptions for charts/images (requires hybrid_mode=full)"},
                "sanitize": {"type": "boolean", "default": False, "description": "Sanitize sensitive data (emails, phones, IPs, credit cards, URLs)"},
                "keep_line_breaks": {"type": "boolean", "default": False},
                "include_header_footer": {"type": "boolean", "default": False},
                "detect_strikethrough": {"type": "boolean", "default": False, "description": "Wrap strikethrough text with ~~ in markdown (experimental)"},
                "markdown_with_html": {"type": "boolean", "default": False, "description": "Allow HTML tags inside markdown for complex tables"},
                "use_struct_tree": {"type": "boolean", "default": False, "description": "Use PDF structure tree (tagged PDF) for reading order"},
                "content_safety_off": {"type": "string", "enum": ["", "all", "hidden-text", "off-page", "tiny", "hidden-ocg"], "description": "Disable content safety filters"},
                "threads": {"type": "string", "description": "Worker threads for parallel per-page processing (experimental, e.g. '4')"},
                "replace_invalid_chars": {"type": "string", "description": "Replacement char for invalid/unrecognized characters (default: space)"},
                "table_method": {"type": "string", "enum": ["", "default", "cluster"], "description": "Table detection method"},
                "reading_order": {"type": "string", "enum": ["", "off", "xycut"], "description": "Reading order algorithm"},
                "image_output": {"type": "string", "enum": ["", "off", "embedded", "external"], "description": "Image output mode in extracted content"},
                "image_format": {"type": "string", "enum": ["", "png", "jpeg"], "description": "Output format for extracted images"},
                "hybrid_url": {"type": "string", "description": "Hybrid backend server URL (overrides default)"},
                "hybrid_timeout": {"type": "string", "description": "Hybrid backend request timeout in ms"}},
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
        if name == "search":
            query = arguments["query"]
            count = min(safe_int(arguments.get("count",10)), MAX_RESULTS)
            depth = safe_int(arguments.get("depth",1))
            lang = str(arguments.get("language","en"))
            google_ai_only = bool(arguments.get("google_ai_only", False))
            r = await search_multi(query, count=max(count, depth * 3),
                google_ai_only=google_ai_only,
                search_type=str(arguments.get("search_type","auto")),
                search_prompt=str(arguments.get("search_prompt","")),
                pro_mode=bool(arguments.get("pro_mode",False)),
                gl=str(arguments.get("gl","")), hl=str(arguments.get("hl","en")),
                tbs=str(arguments.get("tbs","")), pws=str(arguments.get("pws","")),
                backend=str(arguments.get("backend","auto")),
                timelimit=str(arguments.get("timelimit","")),
                page=safe_int(arguments.get("page",1)),
                region=str(arguments.get("region","wt-wt")),
                safesearch=str(arguments.get("safesearch","moderate")),
                language=lang,
                country=str(arguments.get("country","")),
                upload_urls=arguments.get("upload_urls"),
                query_expand=bool(arguments.get("query_expand",True)),
                tavily_topic=str(arguments.get("tavily_topic","general")),
                tavily_depth=str(arguments.get("tavily_depth","basic")),
                size=str(arguments.get("size","")),
                color=str(arguments.get("color","")),
                type_image=str(arguments.get("type_image","")),
                layout=str(arguments.get("layout","")),
                license_image=str(arguments.get("license_image","")),
                resolution=str(arguments.get("resolution","")),
                duration=str(arguments.get("duration","")),
                license_videos=str(arguments.get("license_videos","")),
                start_date=str(arguments.get("start_date","")),
                end_date=str(arguments.get("end_date","")),
                exact_phrase=bool(arguments.get("exact_phrase",False)),
                domain=str(arguments.get("domain","")),
                anysearch_tag=str(arguments.get("anysearch_tag","")),
                anysearch_zone=str(arguments.get("anysearch_zone","")),
                anysearch_language=str(arguments.get("anysearch_language","")),
                cdp_url=HELIUM_CDP)
            if r.get("success") and depth >= 2 and r.get("results"):
                fetched = await enrich(r["results"], query, depth=depth,
                    cdp_url=HELIUM_CDP, count=count, language=lang)
                if fetched.get("fetched_content"):
                    r["fetched_content"] = fetched["fetched_content"]
            # Skip Groq synthesis when GAI already returned a full answer
            skip_synthesis = google_ai_only and r.get("ai_answer")
            if r.get("success") and bool(arguments.get("synthesize", True)) and r.get("results") and not skip_synthesis:
                try:
                    top = r["results"][:3]
                    ctx = "\n\n".join(f"[{i+1}] {x.get('title','')}: {x.get('content','')[:400]}"
                                     for i, x in enumerate(top))
                    key = None
                    for k in os.environ.get("GROQ_API_KEYS", "").split(","):
                        k = k.strip()
                        if k: key = k; break
                    if key:
                        from config import get_http_client
                        c = get_http_client()
                        resp = await c.post(
                            "https://api.groq.com/openai/v1/chat/completions",
                            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                            json={"model": "llama-3.3-70b-versatile",
                                  "messages": [{"role": "system", "content": "Answer concisely from sources. Use [N] citations like [1][2]."},
                                               {"role": "user", "content": f"Query: {query}\n\nResults:\n{ctx}"}],
                                  "temperature": 0.3, "max_tokens": 256}, timeout=15)
                        if resp.status_code == 200:
                            r["synthesis"] = {"answer": resp.json()["choices"][0]["message"]["content"].strip()}
                except Exception:
                    pass
            return _res(r)
        elif name == "fetch":
            r = await fetch_url(arguments["url"],
                    max_chars=safe_int(arguments.get("max_chars",5000)),
                    main_content_only=bool(arguments.get("main_content_only",True)),
                    target_language=str(arguments.get("target_language","")),
                    favor_precision=bool(arguments.get("favor_precision",False)),
                    favor_recall=bool(arguments.get("favor_recall",False)),
                    fast=bool(arguments.get("fast",False)),
                    deduplicate=bool(arguments.get("deduplicate",True)),
                    output_format=str(arguments.get("output_format","markdown")),
                    include_images=bool(arguments.get("include_images",True)),
                    include_tables=bool(arguments.get("include_tables",True)),
                    include_comments=bool(arguments.get("include_comments",True)),
                    include_formatting=bool(arguments.get("include_formatting",True)),
                    include_links=bool(arguments.get("include_links",True)),
                    prune_xpath=str(arguments.get("prune_xpath","")),
                    url_blacklist=str(arguments.get("url_blacklist","")),
                    author_blacklist=str(arguments.get("author_blacklist","")),
                    min_output_size=safe_int(arguments.get("min_output_size", 0)),
                    raw=bool(arguments.get("raw", False)))
            # Auto-fallback: if Cloudflare/403/blocked/empty content, retry via CDP
            content = (r.get("content", "") or "").strip()
            status = r.get("status", 0)
            r_error = (r.get("error", "") or "").lower()
            content_lower = content.lower()
            should_retry = (
                not r.get("success") and "cloudflare" in r_error
            ) or status in (403, 429, 503) or (
                len(content) < 300 and (
                    "blocked" in content_lower or "access denied" in content_lower
                    or "network security" in content_lower or "rate limit" in content_lower
                    or "too many requests" in content_lower
                )
            ) or (
                len(content) < 100
            )
            if should_retry:
                r = await scrapling_stealthy_fetch(arguments["url"],
                    css_selector=arguments.get("css_selector"),
                    extraction_type=str(arguments.get("extraction_type","markdown")),
                    cdp_url=HELIUM_CDP, network_idle=bool(arguments.get("network_idle",True)))
            # Line range slicing after fetch
            start_line = safe_int(arguments.get("start_line", 0))
            end_line = safe_int(arguments.get("end_line", 0))
            if r.get("content"):
                all_lines = r["content"].split("\n")
                r["total_lines"] = len(all_lines)
                r["total_chars"] = len(r["content"])
                if start_line > 0:
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
                css_extract=arguments.get("css_extract"),
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
                    depth=arguments.get("depth"),
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
