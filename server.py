"""Unified web MCP server — multi-engine search + stealth scraping + content extraction."""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
from typing import Any

logger = logging.getLogger("websearch")

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import CallToolResult, ListToolsResult, TextContent, Tool

from config import MAX_RESULTS, HELIUM_CDP, get_http_client, close_http_client, _KeyRotator
from cdp_actions import browser_action as _browser_action
from errors import annotate as _err_annotate, classify_exception as _err_exc
from ghost_state import CHALLENGE, CONTENT_OK, classify, ghost
from fetch import fetch_url, scrapling_stealthy_fetch
from pdf_extract import extract_pdf
from screenshot import cdpa11y_snapshot
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
import webtools as _webtools
import session as _session

_groq_keys = _KeyRotator("GROQ_API_KEYS")

INSTRUCTIONS = """# WebSearch MCP

Multi-engine search + content extraction.

## Start here

Run **doctor()** first. It reports which lanes are usable right now and the exact
command to enable the rest. Features that need optional setup: Google AI Mode
(needs a signed-in Chromium browser over CDP), browser_action (same browser),
the bot-wall CDP fallback, in-tool distillation (needs a model endpoint), and
cross-encoder reranking. Without those, the keyless
HTTP lanes still work fully. Never assume a lane is broken before doctor() says so.

## Sessions: when a task spans more than one call

`browser_action` opens a fresh tab per call and closes it, so logins, cookies
and scroll state do not survive. When a task needs state across calls (log in
then act, fill a form over several steps, paginate), do this instead:
1. `browser_session(action='create', label='login', url='https://site/login')`
2. pass the returned `session_id` to every `browser_action` call
3. `browser_session(action='close', session_id='...')` when done, so no tab leaks
Sessions are local-only (they need CDP) and capped at 8.

## When to use what

- **search** for finding information (9 engines, dedup+rerank+synthesis). **synthesize=false** when you only need raw results. Set `domain` for vertical search (code, academic, finance, health, travel, legal).
- **fetch** to read a specific page (auto-CDP fallback on blocked pages. PDF/EPUB/DOCX support). Pass `distill=true` plus `distill_question` when the page is large and you only need facts from it: a local model compresses it first and the result says whether it was distilled, how big the raw text was, and how to get it back with `distill=false`.
- **fetch_many** for 2+ known URLs in one call. **crawl** to walk a site with caps. **extract_links** for every link on a page. **extract_table** for tables as json/csv. **download** to save a file and get a path. **diff_page** to see whether a page changed since it was cached.
- **screenshot** for LLM-readable page text (ARIA snapshot).
- **browser_action** to DRIVE a real browser page (click, type, key press, JS eval, HTML, text, AX tree, PNG screenshot) — for pages that need interaction, not just reading.
- **extract** for structured JSON: **strategy=css** fastest+free (simple fields), **strategy=llm** works on any content (costs quota).
- **map_site** for sitemap discovery.
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

async def handle_list_tools(ctx, params) -> ListToolsResult:
    return ListToolsResult(tools=[
        Tool(name="doctor",
            description="Preflight: report which lanes are usable RIGHT NOW and the exact command to enable the rest. Run this first, and whenever a feature returns empty results. Distinguishes 'broken' from 'needs optional setup' — the browser lanes (Google AI Mode, browser_action, bot-wall bypass) require a Chromium-based browser with --remote-debugging-port on the same host, so they are unavailable on headless servers; the keyless HTTP lanes work anywhere. Never prints key values, only which env var names are set. e.g. doctor()",
            input_schema={"type": "object", "properties": {
                "probe_network": {"type": "boolean", "default": True,
                                  "description": "Run an outbound HTTPS check (adds ~1s)"},
            }},
        ),
        Tool(name="ping",
            description="Lightweight connectivity check — verifies internet and key search endpoints are reachable. Use before expensive calls when connectivity is uncertain. No params needed. e.g. ping()",
            input_schema={"type": "object", "properties": {}},
        ),
        Tool(name="search",
            description="Multi-engine web search with dedup, reranking, and synthesis. Use for open research questions; for a known URL use fetch, for papers use arxiv, for encyclopedic facts use wikipedia. depth=1 returns snippets with relevance_score+fetch_relevance per result. depth>=2 fetches full pages + re-ranks. synthesize=True (default) returns Groq answer with [N] citations. google_ai_only skips all engines for Google AI Mode. e.g. search(query='latest AI models', depth=1)",
            input_schema={"type": "object", "properties": {
                "query": {"type": "string"}, "count": {"type": "integer", "default": 10},
                "depth": {"type": "integer", "default": 1, "description": "1=snippets, 2+=fetch full pages + rerank"},
                "search_type": {"type": "string", "enum": ["auto","text","news","images","videos","books"]},
                "timelimit": {"type": "string", "description": "Time filter: d/w/m/y"},
                "safesearch": {"type": "string", "enum": ["off","moderate","strict"], "default": "moderate"},
                "language": {"type": "string", "default": "en", "description": "Content language for Wikipedia/summaries"},
                "upload_urls": {"type": "array", "items": {"type": "string"}, "description": "GAI file upload: supported formats .avif .bmp .heic .heif .jpeg .pdf .png .webp. 10MB max. Only one file per call (GAI drops all but the last). Local: file:///path or remote URL."},
                "start_date": {"type": "string", "description": "Tavily date filter start (YYYY-MM-DD)"},
                "end_date": {"type": "string", "description": "Tavily date filter end (YYYY-MM-DD)"},
                "include_domains": {"type": "array", "items": {"type": "string"}, "description": "Tavily domain include filter"},
                "exclude_domains": {"type": "array", "items": {"type": "string"}, "description": "Tavily domain exclude filter"},
                "synthesize": {"type": "boolean", "default": True, "description": "Groq-synthesize top results into a concise answer with citations"},
                "domain": {"type": "string", "description": "AnySearch vertical: finance, code, academic, health, travel, legal, security. Guessed from query via simple heuristic — may be wrong, omit for general search."},
                "anysearch_tag": {"type": "string", "description": "AnySearch precise sub-domain tag in {domain}.{sub_domain} format (e.g. code.doc, finance.us_stock). Overrides domain."},
                "anysearch_zone": {"type": "string", "enum": ["", "cn", "intl"], "description": "AnySearch geo zone (cn or intl)"},
                "anysearch_language": {"type": "string", "description": "AnySearch content language (e.g. en, zh-CN)"},
                "google_ai_only": {"type": "boolean", "description": "Skip all other search engines and answer only via Google AI Mode. Requires a signed-in Chromium session reachable over CDP (WEBSEARCH_CDP_URL); on a headless server or without a browser this returns an error, not results — call doctor() to check first"},
                "history": {"type": "string", "description": "Conversation context for query rewriting (resolves pronouns like 'the second one')"}},
                "required": ["query"]}),
         Tool(name="fetch",
            description="URL to markdown/text. Auto-fallback: direct fetch → CDP for blocked/JS pages. Supports PDF, EPUB, DOCX. SSRF-protected. Use focus=\"query\" to filter content by relevance. Paginated via offset (response: next_offset). Results cached 1h; cache_ttl=0 fresh. For structured JSON extraction use `extract` instead. e.g. fetch(url='https://example.com')",
            input_schema={"type": "object", "properties": {
                "url": {"type": "string"}, "max_chars": {"type": "integer", "default": 5000, "description": "Chars to return per call (for pagination)"},
                "offset": {"type": "integer", "default": 0, "description": "Char offset for paginated reads (0 = start). Response includes is_truncated, next_offset, total_extracted_chars (full page size)."},
                "focus": {"type": "string", "description": "BM25 relevance filter — extract only content blocks relevant to this query. Use ONLY to pull specific sections from a large page (e.g. focus='pricing'). On crucial pages INSTEAD omit focus and fetch the whole content. Misses drop content the page's own Ctrl-F would find (focus is semantic, not literal) — never use focus when the full content is the deliverable."},
                "cache_ttl": {"type": "integer", "default": 3600, "description": "Cache TTL in seconds (0 = force fresh fetch). Cache keyed by URL+extraction_type+css_selector, not focus/offset."},
                "css_selector": {"type": "string", "description": "CSS selector — extract only the matching element's text (CDP tier only; ignored on the direct-fetch tier)"},
                "extraction_type": {"type": "string", "enum": ["markdown","text","html"]},
                "target_language": {"type": "string"},
                "output_format": {"type": "string", "enum": ["markdown","txt","json","xml","csv"], "default": "markdown"},
                "fast": {"type": "boolean"},
                "raw": {"type": "boolean", "description": "Skip CDP/trafilatura, return raw text directly (use for GitHub raw files, pastebin, etc.)"},
                "network_idle": {"type": "boolean", "default": True, "description": "Wait for network idle before extracting (slower but captures JS-rendered content; CDP tier only)"},
                   "include_images": {"type": "boolean", "default": True, "description": "Include image captions/alt text"},
                   "include_links": {"type": "boolean", "default": True, "description": "Include hyperlinks in output"},
                   "include_formatting": {"type": "boolean", "default": True, "description": "Preserve text formatting (bold, italic, etc)"},
                   "include_tables": {"type": "boolean", "default": True, "description": "Extract tables from HTML"},
                    "start_line": {"type": "integer", "description": "1-based start line for reading a range (slices content by newline)"},
                    "end_line": {"type": "integer", "description": "1-based end line (inclusive). Use with start_line for targeted reading."},
                    "quality_floor": {"type": "number", "default": 0, "description": "Min acceptable extraction quality 0-1 (length+density+structure score); below it the result is flagged low_quality"},
                    "mineru": {"type": "boolean", "default": False, "description": "Opt-in MinerU-HTML SLM re-extraction for low-quality pages (heavy, CPU, seconds per page)"},
                    "cookies": {"type": "array", "items": {"type": "object"}, "description": "Optional cookies injected into the request: [{name, value, domain?, path?}] (e.g. session cookie for a page behind login)"},
                    "distill": {"type": "boolean", "default": False, "description": "Opt-in: compress the page with a cheap local model before returning it, to save your context. Never on by default and never silent - the result carries distill.distilled, raw_chars and how_to_get_raw, and the untouched text is one call away with distill=false. Skipped automatically below the size threshold, where it would cost more than it saves."},
                    "distill_question": {"type": "string", "description": "What you are actually looking for. Gives the distiller something to focus on, which reduces the chance it drops the detail you needed."},
                    },
                "required": ["url"]}),
          Tool(name="screenshot",
            description="ARIA accessibility snapshot (LLM-readable page text). Use to see a page's interactive structure and what is actually visible; for readable article content use fetch. e.g. screenshot(url='https://example.com')",
            input_schema={"type": "object", "properties": {
                "url": {"type": "string"},
                "max_chars": {"type": "integer", "default": 10000},
                "depth": {"type": "integer", "description": "ARIA snapshot tree depth limit"},
                "verbose": {"type": "boolean", "default": False, "description": "Show all ARIA roles (not just interactive)"},
                "start_line": {"type": "integer", "description": "1-based start line for snapshot text range"},
                "end_line": {"type": "integer", "description": "1-based end line (inclusive) for snapshot text range"}},
                "required": ["url"]}),
         Tool(name="browser_action",
            description="Drive a real browser page (needs a Chromium-based browser with --remote-debugging-port on this host, e.g. chromium/google-chrome/brave/Helium, set via WEBSEARCH_CDP_URL; run doctor() to check): navigate, click, type, press a key, evaluate JS, read page HTML/text, get the accessibility tree, or capture a PNG. Use when a page needs interaction (login form, search box, 'load more' button) — for plain reading use fetch, for structure use screenshot. One call = one fresh tab, closed afterwards; pass steps[] to run a multi-step flow (focus -> type -> Enter -> read) in a single tab. URLs are SSRF-validated. e.g. browser_action(action='click', url='https://example.com', selector='#more')",
            input_schema={"type": "object", "properties": {
                "action": {"type": "string", "enum": ["navigate","click","type","key","evaluate","html","text","aria","screenshot","wait"], "description": "navigate=go to url, click=mouse click on selector, type=insert text (focuses selector first if given), key=press a key (Enter/Escape/Tab), evaluate=run JS and return its value, html=page HTML, text=inner text of the page or of one selector, aria=accessibility tree, screenshot=PNG/JPEG written to a temp file, wait=wait for a load milestone"},
                "url": {"type": "string", "description": "URL to act on (required for action=navigate; optional for other actions — the tab navigates there first). SSRF-validated."},
                "selector": {"type": "string", "description": "CSS selector — required for action=click, optional focus target for action=type, element for action=text"},
                "text": {"type": "string", "description": "Text to type (action=type)"},
                "key": {"type": "string", "description": "DOM key value to press (action=key), e.g. Enter, Escape, Tab, ArrowDown"},
                "expression": {"type": "string", "description": "JavaScript expression to evaluate in the page (action=evaluate). Arrow functions are auto-invoked."},
                "wait_state": {"type": "string", "enum": ["domcontentloaded","load","networkidle"], "default": "domcontentloaded", "description": "Load milestone to wait for after navigating (and for action=wait)"},
                "steps": {"type": "array", "items": {"type": "object"}, "description": "Ordered multi-step flow executed in ONE tab (max 25). Each step is an object with the same fields as the top-level params (action, url, selector, text, key, expression, wait_state). Use for flows like: type into a box, press Enter, then read the results."},
                "session_id": {"type": "string", "description": "Reuse a persistent tab from browser_session(action='create') instead of opening a throwaway one. Required for anything stateful across calls: logins, cookies, multi-page flows, forms. Without it, each call gets a fresh tab that is closed afterwards."},
                "timeout": {"type": "number", "default": 30, "description": "Per-step timeout in seconds (1-120)"},
                "max_chars": {"type": "integer", "default": 20000, "description": "Max chars returned for html/text/aria (200-200000)"},
                "full_page": {"type": "boolean", "default": True, "description": "action=screenshot: capture beyond the viewport"},
                "image_type": {"type": "string", "enum": ["png","jpeg"], "default": "png", "description": "action=screenshot: image format"},
                "quality": {"type": "integer", "description": "action=screenshot: JPEG quality 0-100"},
                "depth": {"type": "integer", "default": 5, "description": "action=aria: accessibility tree depth (1-100)"}},
                "required": []}),
         Tool(name="browser_session",
            description="Persistent browser tab you can reuse across calls (needs a Chromium with --remote-debugging-port; run doctor()). Without a session every browser_action call gets a fresh tab that is closed immediately, so logins, cookies and multi-step flows cannot survive. Create a session, then pass session_id to browser_action. action=create|list|info|close. Sessions are capped (8) and the least recently used is evicted; close when done so no tab leaks. e.g. browser_session(action='create', label='login', url='https://example.com') -> then browser_action(action='click', selector='#x', session_id='...')",
            input_schema={"type": "object", "properties": {
                "action": {"type": "string", "enum": ["create", "list", "info", "close"], "default": "list"},
                "session_id": {"type": "string", "description": "Required for action=info/close; omitted from action=close closes every session"},
                "label": {"type": "string", "description": "Human label folded into the generated id, e.g. 'login'"},
                "url": {"type": "string", "description": "Optional page to open in the new session"}},
                "required": []}),
         Tool(name="fetch_many",
            description="Fetch several URLs in one call. Replaces a loop of fetch calls or a throwaway requests script: dedups URLs, caps concurrency, honours robots.txt, shares the cache, and returns per-URL success/status/content plus counts. Reports `truncated` when the batch cap (50) is hit so you know to call again for the rest. Use when you have 2+ URLs; use fetch for a single known page.",
            input_schema={"type": "object", "properties": {
                "urls": {"type": "array", "items": {"type": "string"}},
                "max_concurrent": {"type": "integer", "default": 3, "description": "Parallel fetches (1-8)"},
                "max_chars": {"type": "integer", "default": 5000},
                "cache_ttl": {"type": "integer", "default": 3600, "description": "Shared cache TTL in seconds; 0 forces a refetch of every URL."},
                "focus": {"type": "string", "description": "Keep only content relevant to this query"},
                "respect_robots": {"type": "boolean", "default": True}},
                "required": ["urls"]}),
         Tool(name="crawl",
            description="Breadth-first crawl of a site with hard caps. Replaces a bespoke scraper and makes politeness non-optional: per-page cap, depth cap, robots.txt, dedup, concurrency limit. Returns each page's title and trimmed content plus a stop_reason so you know whether the site was exhausted or a cap stopped it. Use for 'read this whole docs site'; for a known list of URLs use fetch_many.",
            input_schema={"type": "object", "properties": {
                "start_url": {"type": "string"},
                "max_depth": {"type": "integer", "default": 1, "description": "Link hops from start (0-5)"},
                "max_urls": {"type": "integer", "default": 20, "description": "Hard page cap (1-100)"},
                "same_domain": {"type": "boolean", "default": True},
                "include_patterns": {"type": "array", "items": {"type": "string"}, "description": "Only URLs containing one of these substrings"},
                "exclude_patterns": {"type": "array", "items": {"type": "string"}},
                "concurrency": {"type": "integer", "default": 2},
                "respect_robots": {"type": "boolean", "default": True},
                "max_chars": {"type": "integer", "default": 4000, "description": "Content kept per page"}},
                "required": ["start_url"]}),
         Tool(name="extract_links",
            description="Every link on a page with its anchor text, title and rel. Replaces hand-parsing HTML before crawling or finding a 'next' button. Resolves relative hrefs, drops javascript:/mailto:, dedups, and can stay on one domain. Use before crawl or to locate a specific navigation target.",
            input_schema={"type": "object", "properties": {
                "url": {"type": "string"},
                "same_domain": {"type": "boolean", "default": True},
                "max_links": {"type": "integer", "default": 500},
                "include_text": {"type": "boolean", "default": True}},
                "required": ["url"]}),
         Tool(name="extract_table",
            description="Pull HTML tables into json, csv or markdown. Replaces a pandas/lxml script every time a page has a pricing or stats table. Reports tables_found so you can pick one by index, and marks truncated when max_rows is hit. Use extract(css) for non-tabular structure instead.",
            input_schema={"type": "object", "properties": {
                "url": {"type": "string"},
                "selector": {"type": "string", "description": "CSS selector to narrow which tables are considered"},
                "fmt": {"type": "string", "enum": ["json", "csv", "markdown"], "default": "json"},
                "index": {"type": "integer", "default": 0, "description": "0-based index among matched tables"},
                "max_rows": {"type": "integer", "default": 5000}},
                "required": ["url"]}),
         Tool(name="download",
            description="Save a URL to disk and return the local path, byte count, sha256 and content type. Replaces curl/requests plus a filename guess. Writes only inside dest_dir (default ~/.cache/websearch/downloads) and refuses path traversal. Marks truncated if the byte cap stopped the transfer, so you know the file is incomplete.",
            input_schema={"type": "object", "properties": {
                "url": {"type": "string"},
                "dest_dir": {"type": "string"},
                "filename": {"type": "string"},
                "max_bytes": {"type": "integer", "default": 104857600}},
                "required": ["url"]}),
         Tool(name="diff_page",
            description="Has this page changed since the last time it was fetched? Compares the live page against the cached snapshot and returns changed/old_hash/new_hash plus the added and removed lines. Replaces a hand-rolled polling and hashing script. Returns had_previous=false on the first call for a URL: fetch it, then diff later. Useful for tracking docs, pricing and changelogs.",
            input_schema={"type": "object", "properties": {
                "url": {"type": "string"},
                "extraction_type": {"type": "string", "default": "markdown"}},
                "required": ["url"]}),
         Tool(name="wikipedia",
            description="Full Wikipedia API client: articles, summaries, categories, links, pageviews, revisions, backlinks, geosearch, recent changes, random pages. Use for encyclopedic facts; for open research use search. e.g. wikipedia(query='Python', action='summary')",
            input_schema={"type": "object", "properties": {
                "action": {"type": "string", "enum": ["search","summary","summary_action","categories","links","pageviews","extlinks","categorymembers","revisions","backlinks","recentchanges","geosearch","random","langlinks","allpages"], "default": "search", "description": "search=find articles, summary=REST fast extract, summary_action=Action API+images, categories=list page cats, links=page links, pageviews=traffic stats"},
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
            description="Search arXiv academic papers (single-source; for multi-source paper search use codesearch papers). Use raw_query for boolean operators (AND, OR, ANDNOT), phrase search (ti:\"exact phrase\"), wildcards (au:smith*). e.g. arxiv(query='cs.AI transformer', count=5)",
            input_schema={"type": "object", "properties": {
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
            description="Discover all pages on a website via sitemap XML (primary) and HTML link extraction (fallback). Use to survey a site's pages before fetching individually. Returns the base domain, source type, total count, and an array of discovered URLs with metadata (last_modified, priority, changefreq when available). e.g. map_site(url='https://example.com')",
            input_schema={"type": "object", "properties": {
                "url": {"type": "string", "description": "Full URL of the site to map (e.g. https://example.com)"},
                "max_urls": {"type": "integer", "default": 1000, "description": "Cap on returned URLs"},
                "max_depth": {"type": "integer", "default": 0, "description": "Recursive link extraction depth (0=homepage only). Only used when no sitemap exists."},
                "include_sitemap": {"type": "boolean", "default": True, "description": "Try sitemap discovery first"},
                "include_links": {"type": "boolean", "default": True, "description": "Fall back to HTML link extraction when no sitemap"},
                "same_domain": {"type": "boolean", "default": True, "description": "Only include URLs from the same domain"},
                 "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "Regex patterns to exclude matching URLs"},
                 "rank": {"type": "boolean", "default": False, "description": "Score each URL by fetch quality + sitemap priority and return important-first with rank_score (costs one scored pass; use before deep-fetching a site)"}},
                  "required": ["url"]}),
         Tool(name="extract",
            description="Extract structured JSON from a webpage using LLM, CSS, or regex strategies. Persistent cache — repeat calls free. For LLM strategy, describe what you want and get clean JSON back. For CSS strategy, provide field names (fastest, free). For raw page content, use `fetch` instead. Examples: extract(url='...', instruction='extract product name and price') or extract(url='...', fields=['name','price'], strategy='css')",
            input_schema={"type": "object", "properties": {
                "url": {"type": "string", "description": "Target URL to extract data from"},
                "instruction": {"type": "string", "description": "Natural language extraction instruction (used with strategy=llm). Example: 'extract all product names, prices, and ratings from this page'"},
                "strategy": {"type": "string", "enum": ["llm", "css", "regex"], "default": "css", "description": "css=CSS-selector-based extraction (fast, free), llm=AI-powered (costs quota), regex=pattern-based (emails, phones, URLs)"},
                "fields": {"type": "array", "items": {"type": "string"}, "description": "List of field names for extraction (e.g. ['name', 'price', 'rating']). Used with strategy=css or strategy=llm"},
                "chunk_threshold": {"type": "integer", "default": 2000, "description": "Max tokens per chunk for LLM extraction (lower = cheaper, higher = more context)"},
                "css_selector": {"type": "string", "description": "CSS selector for the container element (used with strategy=css). Defaults to 'body'"},
                "provider": {"type": "string", "default": "groq/openai/gpt-oss-120b", "description": "LLM provider string in LiteLLM format (e.g. groq/openai/gpt-oss-120b, openai/gpt-4o, ollama/llama2)"}},
                "required": ["url"]}),
         Tool(name="pdf_extract",
            description="PDF extraction: PyMuPDF fast text-layer path, Docling+OCR fallback for scanned/image-only PDFs (CPU). Formats: markdown (default), json, html, or comma combos like 'markdown,json'. e.g. pdf_extract(input_path='/path/to/doc.pdf', format='markdown')",
            input_schema={"type": "object", "properties": {
                "input_path": {"type": "array", "items": {"type": "string"}, "description": "PDF file paths or URLs (local files, http/https, file://)"},
                "format": {"type": "string", "enum": ["markdown","json","html","markdown,json","markdown,json,html"], "default": "markdown"},
                "password": {"type": "string", "description": "PDF password for protected files"},
                "pages": {"type": "string", "description": "Page range e.g. 1-5,8,10-12"},
                "hybrid": {"type": "string", "enum": ["", "docling-fast"], "description": "Force the Docling OCR fallback regardless of text layer (upper bound: CPU cost only)"},
                "hybrid_mode": {"type": "string", "description": "Alias for hybrid (same values)"},
                "force_ocr": {"type": "boolean", "default": False, "description": "Shortcut: force the Docling OCR fallback even when a text layer exists"},
            },
                "required": ["input_path"]}),
    ])


async def verify_grounding(answer: str, sources: list[dict], groq_key: str) -> dict:
    """Re-prompt Groq: for each sentence, is the claim backed by a cited source?

    Returns {"flagged": [{"sentence": str, "issue": str}], "ok": bool}.
    """
    sources_block = "\n".join(f"[{s['n']}] {s['title']} — {s['url']}" for s in sources)
    sys_p = (
        "You are a grounding verifier. Given an answer and its source list, "
        "identify every sentence whose central claim is NOT supported by any "
        "cited source [N]. Output ONLY JSON: "
        '{"flagged":[{"sentence":str,"issue":str}],"ok":bool}.'
    )
    c = get_http_client()
    resp = await c.post(
        "https://api.groq.com/openai/v1/chat/completions",
        headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
        json={"model": "openai/gpt-oss-120b",
              "messages": [{"role": "system", "content": sys_p},
                           {"role": "user", "content": f"Sources:\n{sources_block}\n\nAnswer:\n{answer}"}],
              "temperature": 0.0, "max_tokens": 1024}, timeout=15)
    if resp.status_code != 200:
        return {"flagged": [], "ok": None, "error": f"http {resp.status_code}"}
    text = resp.json()["choices"][0]["message"]["content"].strip()
    if not text:
        return {"flagged": [], "ok": None, "error": "empty completion"}
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        parsed = json.loads(text)
    except Exception:
        a, b = text.find("{"), text.rfind("}")
        parsed = json.loads(text[a:b+1]) if a >= 0 and b > a else None
    if isinstance(parsed, dict):
        flagged = parsed.get("flagged", []) if isinstance(parsed.get("flagged"), list) else []
        return {"flagged": flagged, "ok": bool(parsed.get("ok", not flagged))}
    return {"flagged": [], "ok": None, "error": "unparseable"}


async def handle_call_tool(ctx, params) -> CallToolResult:
    name = params.name
    arguments = params.arguments or {}
    if not isinstance(arguments, dict):
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(_err_annotate({"error": "arguments must be a dict"})))], is_error=True)

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
        if not ok and isinstance(data, dict):
            _err_annotate(data)
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(data, default=str))], is_error=not ok)

    try:
        if name == "doctor":
            from doctor import preflight
            report = await preflight(probe_network=bool(arguments.get("probe_network", True)))
            return _res({"success": True, **report})

        if name == "browser_session":
            act = str(arguments.get("action") or "list").strip().lower()
            sid = str(arguments.get("session_id") or "").strip()
            if act == "create":
                return _res(await _session.create(
                    label=str(arguments.get("label") or ""),
                    url=str(arguments.get("url") or "")))
            if act == "info":
                if not sid:
                    return _res({"success": False, "error": "session_id is required for action=info"})
                return _res(await _session.info(sid))
            if act == "close":
                return _res(await _session.close(sid))
            return _res(await _session.list_all())

        if name == "fetch_many":
            urls = arguments.get("urls") or []
            if isinstance(urls, str):
                urls = [u.strip() for u in urls.replace("\n", ",").split(",") if u.strip()]
            return _res(await _webtools.fetch_many(
                list(urls),
                max_concurrent=safe_int(arguments.get("max_concurrent", 3)),
                max_chars=safe_int(arguments.get("max_chars", 5000)),
                cache_ttl=safe_int(arguments.get("cache_ttl", 3600)),
                focus=str(arguments.get("focus") or ""),
                respect_robots=bool(arguments.get("respect_robots", True))))

        if name == "crawl":
            start = str(arguments.get("start_url") or "").strip()
            if not start:
                return _res({"success": False, "error": "start_url is required"})
            return _res(await _webtools.crawl(
                start,
                max_depth=safe_int(arguments.get("max_depth", 1)),
                max_urls=safe_int(arguments.get("max_urls", 20)),
                same_domain=bool(arguments.get("same_domain", True)),
                include_patterns=list(arguments.get("include_patterns") or []),
                exclude_patterns=list(arguments.get("exclude_patterns") or []),
                concurrency=safe_int(arguments.get("concurrency", 2)),
                respect_robots=bool(arguments.get("respect_robots", True)),
                max_chars=safe_int(arguments.get("max_chars", 4000))))

        if name == "extract_links":
            u = str(arguments.get("url") or "").strip()
            if not u:
                return _res({"success": False, "error": "url is required"})
            return _res(await _webtools.extract_links(
                u,
                same_domain=bool(arguments.get("same_domain", True)),
                max_links=safe_int(arguments.get("max_links", 500)),
                include_text=bool(arguments.get("include_text", True))))

        if name == "extract_table":
            u = str(arguments.get("url") or "").strip()
            if not u:
                return _res({"success": False, "error": "url is required"})
            return _res(await _webtools.extract_table(
                u,
                selector=str(arguments.get("selector") or ""),
                fmt=str(arguments.get("fmt") or "json"),
                index=safe_int(arguments.get("index", 0)),
                max_rows=safe_int(arguments.get("max_rows", 5000))))

        if name == "download":
            u = str(arguments.get("url") or "").strip()
            if not u:
                return _res({"success": False, "error": "url is required"})
            return _res(await _webtools.download(
                u,
                dest_dir=str(arguments.get("dest_dir") or ""),
                filename=str(arguments.get("filename") or ""),
                max_bytes=safe_int(arguments.get("max_bytes", 104857600))))

        if name == "diff_page":
            u = str(arguments.get("url") or "").strip()
            if not u:
                return _res({"success": False, "error": "url is required"})
            return _res(await _webtools.diff_page(
                u, extraction_type=str(arguments.get("extraction_type") or "markdown")))

        if name == "ping":
            c = get_http_client()
            results = {}
            for target, url in [("cloudflare", "https://1.1.1.1"), ("google", "https://www.google.com"), ("archive", "https://archive.org")]:
                try:
                    r = await c.get(url, timeout=5)
                    results[target] = {"reachable": True, "status": r.status_code, "ms": int(r.elapsed.total_seconds() * 1000)}
                except Exception as e:
                    results[target] = {"reachable": False, "error": str(e)[:60]}
            return _res({"success": True, "connectivity": results})

        if name == "search":
            query = str(arguments.get("query", "")).strip()
            if not query:
                return _res({"success": False, "error": "query must not be empty"})
            depth = safe_int(arguments.get("depth",1))
            count = min(max(safe_int(arguments.get("count",10)), depth * 3), MAX_RESULTS)
            lang = str(arguments.get("language","en"))
            google_ai_only = bool(arguments.get("google_ai_only", False))
            r = await search_multi(query, count=count,
                google_ai_only=google_ai_only,
                search_type=str(arguments.get("search_type","auto")),
                timelimit=str(arguments.get("timelimit","")),
                safesearch=str(arguments.get("safesearch","moderate")),
                language=lang,
                upload_urls=arguments.get("upload_urls"),
                start_date=str(arguments.get("start_date","")),
                end_date=str(arguments.get("end_date","")),
                include_domains=arguments.get("include_domains"),
                exclude_domains=arguments.get("exclude_domains"),
                domain=str(arguments.get("domain","")),
                anysearch_tag=str(arguments.get("anysearch_tag","")),
                anysearch_zone=str(arguments.get("anysearch_zone","")),
                anysearch_language=str(arguments.get("anysearch_language","")),
                cdp_url=HELIUM_CDP,
                depth=depth,
                history=str(arguments.get("history", "")))
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
                    top = (r.get("fetched_content") or r["results"])[:max(3, min(depth * 2, 8))]
                    ctx = "\n\n".join(
                        f"[{i+1}]\nSource: {x.get('url', '')}\nTitle: {x.get('title','')}\n"
                        f"Content: {(x.get('content') or x.get('snippet') or '')[:1500]}"
                        for i, x in enumerate(top))
                    if _groq_keys.has_keys:
                        groq_key = await _groq_keys.next()
                        c = get_http_client()
                        answer = ""
                        for _ in range(2):
                            resp = await c.post(
                                "https://api.groq.com/openai/v1/chat/completions",
                                headers={"Authorization": f"Bearer {groq_key}", "Content-Type": "application/json"},
                                json={"model": "openai/gpt-oss-120b",
                                      "messages": [{"role": "system", "content": "Synthesize a comprehensive answer based ONLY on the provided search results. Cite your sources using numbers [1], [2], ... corresponding to the search results. If the results are insufficient, state that clearly."},
                                                   {"role": "user", "content": f"Query: {query}\n\nResults:\n{ctx}"}],
                                      "temperature": 0.3, "max_tokens": 512}, timeout=15)
                            if resp.status_code == 200:
                                answer = (resp.json()["choices"][0]["message"]["content"] or "").strip()
                                if answer:
                                    break
                        if answer:
                            sources = [{"n": i + 1, "title": x.get("title", ""), "url": x.get("url", "")}
                                       for i, x in enumerate(top)]
                            answer += "\n\n---\n\n## Sources\n\n" + "\n".join(
                                f"[{s['n']}] {s['title']} — {s['url']}" for s in sources)
                            r["synthesis"] = {"answer": answer, "sources": sources}
                            if len(answer) > 120:
                                try:
                                    r["synthesis"]["grounded"] = await verify_grounding(
                                        answer, sources, groq_key)
                                except Exception as e:
                                    logger.warning("grounding check failed: %s", e)
                except Exception as e:
                    r["synthesis_error"] = str(e)
            return _res(r)
        elif name == "fetch":
            url = arguments["url"]
            offset = safe_int(arguments.get("offset", 0))
            max_chars = safe_int(arguments.get("max_chars", 5000))

            host = ghost.host_of(url)
            route = ghost.route_for(host)

            # ── Tier-1 fetch, routed by domain profile ──────────────
            if route == "skip_to_solve":
                # recent failed cold check + stale cookies: skip the
                # doomed tier-1 round-trip, solve straight in browser
                r = None
            else:
                # explicit model-supplied cookies win over the ghost vault
                cookies = arguments.get("cookies") or (ghost.vault(host) if route == "warm" else None)
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
                        raw=bool(arguments.get("raw", False)),
                        cookies=cookies,
                        quality_floor=float(arguments.get("quality_floor", 0.0)),
                        mineru=bool(arguments.get("mineru", False)))
                if r.get("success"):
                    if route == "warm":
                        ghost.warm_ok(host)
                    else:
                        ghost.record_fetch(host, CONTENT_OK)
                else:
                    verdict = classify(r.get("status", 0), r.get("content", ""),
                                       r.get("title", ""))
                    if verdict != CONTENT_OK:
                        ghost.record_fetch(host, verdict)
                    if route == "warm":
                        ghost.record_warm_stale(host)

            # ── Escalation decision ─────────────────────────────────
            should_retry = False
            cf_hits = 0
            _is_js_shell = False
            if r is None:
                should_retry = True
            elif not r.get("success"):
                content = (r.get("content", "") or "").strip()
                status = r.get("status", 0)
                r_error = (r.get("error", "") or "").lower()
                content_lower = content.lower()
                # ponytail: twelvedata/docs is a JS shell (GTM) not a bot wall — don't burn CDP on it
                _is_js_shell = "gtm.start" in content_lower[:2000] or "__NEXT_DATA__" in content_lower[:2000]
                cf_hits = sum(1 for kw in ["just a moment", "checking your browser",
                               "cf-challenge", "__cf_chl_", "cf-turnstile",
                               "verify you are human", "attention required"]
                              if kw in content_lower[:800])
                should_retry = (not _is_js_shell) and (
                    "cloudflare" in r_error
                    or status in (403, 429, 503) or cf_hits >= 2 or (
                    len(content) < 300 and (
                        "blocked" in content_lower or "access denied" in content_lower
                        or "network security" in content_lower or "rate limit" in content_lower
                        or "too many requests" in content_lower
                    )
                ) or (len(content) < 100 and (
                    # tiny-but-real pages (small HTML, text extracted) don't
                    # benefit from a browser re-render — skip the wasted CDP
                    # round-trip; big HTML with no text is a JS shell → retry
                    not r.get("raw_html_len") or r.get("raw_html_len", 0) > 5000)))

            # ── Tier-1.5 Jina Reader (proposal 6): keyless r.jina.ai fallback
            # after direct fetch fails, BEFORE the 12s CDP escalation. Cuts
            # browser load on soft walls. Skipped for tiny-but-real pages
            # (nothing to gain) and JS shells (Jina renders no JS either).
            if (should_retry and r is not None and not r.get("success")
                    and ghost.tier_allowed(host, "t2")):
                _content_len = len((r.get("content", "") or "").strip())
                _raw_len = r.get("raw_html_len") or 0
                _tiny_real = 0 < _content_len < 100 and _raw_len <= 5000
                if not _is_js_shell and not _tiny_real:
                    try:
                        from recovery_tiers import jina_fetch as _jina_fetch
                        jina_r = await _jina_fetch(url)
                    except Exception:
                        jina_r = None
                    if jina_r and jina_r.get("success"):
                        ghost.record_fetch(host, CONTENT_OK, tier="t1.5")
                        from fetch import _build_paginated_response as _paginate
                        r = _paginate(
                            url, jina_r.get("content", ""),
                            jina_r.get("status", 200),
                            jina_r.get("title", ""),
                            jina_r.get("metadata") or {},
                            jina_r.get("content_type", "text/plain"),
                            offset, max_chars, method="jina-reader")
                        should_retry = False

            # ── Tier-2 browser solve + solve-and-bounce handoff ─────
            if should_retry and ghost.tier_allowed(host, "t2"):
                async def _replay(cks):
                    return await fetch_url(url,
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
                            raw=bool(arguments.get("raw", False)),
                            cookies=cks)

                # Turnstile widget solve first (cheap, no full render)
                cf_cookies = None
                if cf_hits >= 2:
                    from turnstile_solve import turnstile_solve as _cf_solve
                    cf_cookies = await _cf_solve(url)
                if cf_cookies:
                    ghost.record_solved(host, cf_cookies, replay_ok=False, tier="t2")
                    vault = ghost.vault(host)
                    if vault:
                        replay = await _replay(vault)
                        if replay.get("success"):
                            ghost.set_replay_ok(host, True)
                            r = replay
                if not cf_cookies or not r.get("success"):
                    try:
                        cdp_r = await asyncio.wait_for(scrapling_stealthy_fetch(url,
                            css_selector=arguments.get("css_selector"),
                            extraction_type=str(arguments.get("extraction_type", "markdown")),
                            cdp_url=HELIUM_CDP, network_idle=bool(arguments.get("network_idle", True))), timeout=12)
                    except (asyncio.TimeoutError, Exception):
                        cdp_r = {"success": False}
                    if cdp_r.get("success"):
                        # browser solved the wall → store clearance cookies
                        ghost.record_solved(host, cdp_r.get("cookies", []),
                                            replay_ok=False, tier="t2")
                        vault = ghost.vault(host)
                        if vault:
                            # replay the cheap tier-1 fetch with clearance
                            # cookies; real content means future warm
                            # fetches skip the browser entirely
                            replay = await _replay(vault)
                            if replay.get("success"):
                                ghost.set_replay_ok(host, True)
                                r = replay
                            else:
                                ghost.set_replay_ok(host, False)
                                r = cdp_r
                        else:
                            r = cdp_r
                    else:
                        ghost.record_fetch(host, cdp_r.get("verdict", CHALLENGE), tier="t2")
                        r = cdp_r

            # Re-apply focus on browser-served results (httpx/cache
            # paths already applied it inside fetch_url)
            if r.get("success") and r.get("method") == "cdp":
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

            # ── Optional in-tool distillation ────────────────────────
            # Opt-in only. Never silently replaces content: the result
            # advertises distilled=true, raw_chars, and how to get the raw
            # text back, so the caller cannot mistake a summary for the page.
            if bool(arguments.get("distill", False)) and r.get("success") and r.get("content"):
                import distill as _distill_mod
                dd = await _distill_mod.distill(
                    r["content"],
                    question=str(arguments.get("distill_question", "") or ""),
                    url=url)
                r["distill"] = {k: v for k, v in dd.items() if k != "content"}
                if dd.get("distilled"):
                    r["content"] = dd["content"]
                else:
                    r.setdefault("distill_skipped", dd.get("distill_skipped", ""))
            return _res(r)
        elif name == "screenshot":
            snap = await cdpa11y_snapshot(url=str(arguments["url"]),
                verbose=bool(arguments.get("verbose",False)),
                max_chars=safe_int(arguments.get("max_chars",10000)),
                depth=arguments.get("depth") or 5)
            r = snap
            # Line range slicing for snapshot content (cdpa11y_snapshot returns "snapshot", pdf extract "content")
            text_key = None
            if isinstance(r, dict):
                if r.get("snapshot") is not None:
                    text_key = "snapshot"
                elif r.get("content") is not None:
                    text_key = "content"
            if text_key:
                text = r[text_key]
                all_lines = text.split("\n")
                r["total_lines"] = len(all_lines)
                r["total_chars"] = len(text)
                start_line = safe_int(arguments.get("start_line", 0))
                end_line = safe_int(arguments.get("end_line", 0))
                if start_line > 0:
                    if end_line > 0:
                        r[text_key] = "\n".join(all_lines[start_line - 1:end_line])
                    else:
                        r[text_key] = "\n".join(all_lines[start_line - 1:])
                    r["returned_lines"] = r[text_key].count("\n") + 1
            return _res(r)
        elif name == "browser_action":
            steps = arguments.get("steps")
            if steps is not None and not isinstance(steps, list):
                return _res({"success": False, "error": "steps must be a list of action objects"})
            r = await _browser_action(
                action=str(arguments.get("action", "") or ""),
                url=str(arguments.get("url", "") or ""),
                selector=str(arguments.get("selector", "") or ""),
                text=str(arguments.get("text", "") or ""),
                key=str(arguments.get("key", "") or ""),
                expression=str(arguments.get("expression", "") or ""),
                wait_state=str(arguments.get("wait_state", "domcontentloaded") or "domcontentloaded"),
                steps=steps,
                session_id=str(arguments.get("session_id", "") or ""),
                timeout=safe_float(arguments.get("timeout", 30.0), 30.0),
                max_chars=safe_int(arguments.get("max_chars", 20000), 20000),
                full_page=bool(arguments.get("full_page", True)),
                image_type=str(arguments.get("image_type", "png") or "png"),
                quality=arguments.get("quality"),
                depth=safe_int(arguments.get("depth", 5), 5),
            )
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
            return _res({"success": True, **r})

        elif name == "extract":
            r = await extract_content(
                url=str(arguments["url"]),
                instruction=str(arguments.get("instruction") or "").strip() or None,
                strategy=str(arguments.get("strategy", "css")),
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
            r = await extract_pdf(paths,
                format=str(arguments.get("format", "markdown")),
                password=str(arguments.get("password", "")),
                pages=str(arguments.get("pages", "")),
                hybrid=hy, hybrid_mode=hm)
            return _res(r)
        elif name == "map_site":
            r = await map_site(url=str(arguments.get("url","")),
                max_urls=safe_int(arguments.get("max_urls",1000)),
                max_depth=safe_int(arguments.get("max_depth",0)),
                include_sitemap=bool(arguments.get("include_sitemap",True)),
                include_links=bool(arguments.get("include_links",True)),
                same_domain=bool(arguments.get("same_domain",True)),
                exclude_patterns=arguments.get("exclude_patterns"),
                rank=bool(arguments.get("rank", False)))
            return _res(r)
        else:
            return CallToolResult(content=[TextContent(type="text", text=json.dumps(_err_annotate({"error": f"Unknown tool: {name}"})))], is_error=True)
    except ValueError as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(_err_annotate({"error": str(e)})))], is_error=True)
    except KeyError as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(_err_annotate({"error": f"Missing required argument: {e}"})))], is_error=True)
    except TypeError as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(_err_annotate({"error": str(e)})))], is_error=True)
    except RuntimeError as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(_err_annotate({"error": str(e)})))], is_error=True)
    except Exception as e:
        return CallToolResult(content=[TextContent(type="text", text=json.dumps(_err_annotate({"error": f"{type(e).__name__}: {e}", "errorKind": _err_exc(e)})))], is_error=True)


server = Server("websearch", instructions=INSTRUCTIONS,
    on_list_tools=handle_list_tools,
    on_call_tool=handle_call_tool,
)


async def _warmup_gai():
    """Pre-warm GAI CDP connection at server start."""
    try:
        from search_gai import get_gai_client
        await get_gai_client()
    except Exception:
        pass

async def _parent_watchdog():
    import os
    while True:
        await asyncio.sleep(2)
        if os.getppid() == 1:
            # parent died / reparented to init — opencode closed, exit with it
            import sys
            sys.exit(0)

async def main():
    asyncio.create_task(_parent_watchdog())
    asyncio.create_task(_warmup_gai())
    try:
        async with stdio_server() as (rs, ws):
            await server.run(rs, ws, server.create_initialization_options())
    finally:
        await close_http_client()

if __name__ == "__main__":
    asyncio.run(main())


def cli_main():
    """Console-script entry point (sync wrapper around the async main).

    Used by the packaged distribution: `websearch-mcp` on PATH, and by MCP
    clients that launch the module as a stdio server.
    """
    import asyncio
    asyncio.run(main())
