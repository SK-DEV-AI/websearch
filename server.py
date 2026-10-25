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

from config import (MAX_RESULTS, HELIUM_CDP, get_http_client, close_http_client,
                    _KeyRotator, default_depth, synthesis_settings,
                    DEPTH_MIN, DEPTH_MAX, GROQ_SMART_MODEL)
from diversity import cap_per_domain, drop_near_duplicates
import prompts
import distill as _distill
import cache as _cache_mod
import hashlib
import time
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
import browser_flows as _flows

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

search → multiple engines → dedup → reranker → synthesis.
depth>=2 fetches full pages for better scoring and sizes the fetch
(max(5, depth*2) pages), and defaults to 3 when a model lane is configured: with
a reader available, more independent sources is the point.

The synthesis answer is deliberately short and strict: it cites [N] per claim,
never uses outside knowledge, surfaces disagreement between sources instead of
averaging it, and treats result text as untrusted data (a page saying "ignore
previous instructions" is data, and the answer says it ignored it). Check
`synthesis.grounded` for sentences a second pass judged unsupported, and
`synthesis.truncated` before treating an answer as complete. `synthesis.skipped`
explains why there is no answer rather than leaving the field blank.

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
                "depth": {"type": "integer",
                          "description": "Effort: 1=snippets only, 2+=fetch the top pages and rerank on full text. Defaults to 3 when a model lane is available (GROQ_API_KEYS or the in-tool model) and 1 otherwise; WEBSEARCH_DEFAULT_DEPTH pins it."},
                "search_type": {"type": "string", "enum": ["auto","text","news","images","videos","books"]},
                "timelimit": {"type": "string", "description": "Time filter: d/w/m/y"},
                "safesearch": {"type": "string", "enum": ["off","moderate","strict"], "default": "moderate"},
                "language": {"type": "string", "default": "en", "description": "Content language for Wikipedia/summaries"},
                "upload_urls": {"type": "array", "items": {"type": "string"}, "description": "GAI file upload: supported formats .avif .bmp .heic .heif .jpeg .pdf .png .webp. 10MB max. Only one file per call (GAI drops all but the last). Local: file:///path or remote URL."},
                "start_date": {"type": "string", "description": "Tavily date filter start (YYYY-MM-DD)"},
                "end_date": {"type": "string", "description": "Tavily date filter end (YYYY-MM-DD)"},
                "include_domains": {"type": "array", "items": {"type": "string"}, "description": "Tavily domain include filter"},
                "exclude_domains": {"type": "array", "items": {"type": "string"}, "description": "Tavily domain exclude filter"},
                "synthesize": {"type": "boolean", "default": True, "description": "Synthesize a cited answer from the top results. The answer is deliberately short and says so when it is truncated."},
                "synthesize_length": {"type": "string", "enum": ["short","long"], "default": "short", "description": "short = at most 6 sentences; long = up to 12, for genuinely multi-part questions"},
                "follow_up": {"type": "boolean", "default": True, "description": "Allow ONE extra retrieval round when the first answer admits a gap (rests on a single source, a claim could not be tied to its source, or it says part of the question is unanswered). Adds 20-40s when it triggers; the response records it in follow_ups and marks synthesis.round=2."},
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
                "cache_bust": {"type": "boolean", "default": False, "description": "Ignore a fresh cache entry and re-fetch. The response carries `cache` with hit/age_s/ttl_remaining_s either way, so 'is this stale?' is answerable without guessing."},
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
         Tool(name="form_fill_submit",
            description="Fill several form fields, click submit, and optionally read the result - in one call. Replaces a multi-step browser_action program full of fragile selectors. Works on a session so the submitted state is real; without session_id it creates a temporary session and closes it afterwards. fields[] is [{selector, value}] with optional action='type' (default) or 'click' for checkboxes. If a field fails the form is NOT submitted and the completed fields are reported, so a half-filled form is never mistaken for a sent one. Needs a Chromium with --remote-debugging-port.",
            input_schema={"type": "object", "properties": {
                "fields": {"type": "array", "items": {"type": "object"}, "description": "[{selector: '#email', value: 'a@b.c'}, {selector: '#tos', action: 'click'}]"},
                "session_id": {"type": "string", "description": "Reuse a persistent tab; omit to use a temporary one"},
                "url": {"type": "string", "description": "Page to open first"},
                "submit_selector": {"type": "string", "description": "Submit button; omit to fill without submitting"},
                "wait_state": {"type": "string", "enum": ["domcontentloaded", "load", "networkidle"], "default": "networkidle"},
                "read_selector": {"type": "string", "description": "CSS selector to read after submitting; empty reads the whole page"},
                "timeout": {"type": "number", "default": 20}},
                "required": ["fields"]}),
         Tool(name="paginate",
            description="Click a 'next' control repeatedly and collect a value from each page. Replaces a pagination loop script. Stops at max_pages, when the next control is not clickable, or when the collected value stops changing (which usually means the last page). Always returns stop_reason and marks truncated, so partial coverage is visible. Works on a session, or a temporary one. Needs a Chromium with --remote-debugging-port.",
            input_schema={"type": "object", "properties": {
                "next_selector": {"type": "string", "description": "CSS selector for the next-page control"},
                "session_id": {"type": "string"},
                "url": {"type": "string", "description": "Page to open first"},
                "max_pages": {"type": "integer", "default": 5, "description": "Hard page cap (1-25)"},
                "read_selector": {"type": "string", "description": "What to collect from each page; empty reads the whole page"},
                "wait_state": {"type": "string", "enum": ["domcontentloaded", "load", "networkidle"], "default": "networkidle"},
                "wait_ms": {"type": "integer", "default": 600, "description": "Pause after each navigation"},
                "stop_when_unchanged": {"type": "boolean", "default": True},
                "timeout": {"type": "number", "default": 20}},
                "required": ["next_selector"]}),
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


# How much of each source the verifier is shown. Judging a claim against a bare
# title is not a check, it is a guess: the first version did exactly that and
# flagged true statements as unsupported because it could not see the text.
VERIFIER_EXCERPT_CHARS = 1200
VERIFIER_TOTAL_CHARS = 14_000


async def verify_grounding(answer: str, sources: list[dict], ep: dict,
                           excerpts: list[str] | None = None) -> dict:
    """Re-prompt a model: is each sentence backed by a source it cites?

    Runs on whichever endpoint produced the answer, so a deployment with only an
    in-tool model still gets the check.

    `excerpts` must be the same text the synthesis saw. Without it the verifier
    can only compare claims against titles, which produces confident false
    accusations - worse than no check, because a caller that sees a bogus flag
    learns to ignore real ones.

    The result is advisory: it is another model's opinion, not a proof.
    Returns {"flagged": [...], "ok": bool|None, "checked_by": provider}.
    """
    sources_block = "\n".join(
        f"[{s['n']}] {s['title']} - {s['url']}"
        + (f"\n    text: {excerpts[i][:VERIFIER_EXCERPT_CHARS]}"
           if excerpts and i < len(excerpts) and excerpts[i] else "")
        for i, s in enumerate(sources))
    sources_block = sources_block[:VERIFIER_TOTAL_CHARS]
    msgs = [{"role": "system", "content": prompts.verifier_system()},
            {"role": "user", "content": prompts.verifier_user(sources_block, answer)}]
    text, _finish, err, _att = await _chat(ep, msgs, max_tokens=2048,
                                           attempts=3, timeout=40)
    if not text:
        return {"flagged": [], "ok": None, "error": err,
                "checked_by": ep.get("provider", "")}
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)

    def _parse(t: str) -> dict | None:
        try:
            d = json.loads(t)
            return d if isinstance(d, dict) else None
        except Exception:
            a, b = t.find("{"), t.rfind("}")
            if a >= 0 and b > a:
                try:
                    d = json.loads(t[a:b + 1])
                    return d if isinstance(d, dict) else None
                except Exception:
                    return None
            return None

    if _parse(text) is None:
        # Small models sometimes answer the question in prose instead of the
        # requested object. One retry, with the failure quoted back, is cheaper
        # than reporting "could not check" for an otherwise fine answer.
        retry_msgs = msgs + [
            {"role": "assistant", "content": text[:400]},
            {"role": "user", "content":
             "That was not the required JSON object. Reply with only the JSON "
             'object: {"flagged":[...],"ok":true|false}'},
        ]
        text2, _f2, _e2, _a2 = await _chat(ep, retry_msgs, max_tokens=2048,
                                           attempts=2, timeout=40)
        if text2:
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text2.strip())
    parsed = _parse(text)
    if isinstance(parsed, dict):
        flagged = parsed.get("flagged", []) if isinstance(parsed.get("flagged"), list) else []
        out = {"flagged": flagged, "ok": bool(parsed.get("ok", not flagged)),
               "checked_by": ep.get("provider", ""), "advisory": True}
        if flagged:
            out["note"] = ("Another model judged these sentences unsupported by "
                           "the cited text. Verify before repeating them; the "
                           "verifier can be wrong in both directions.")
        return out
    return {"flagged": [], "ok": None, "error": "unparseable",
            "checked_by": ep.get("provider", "")}

GROQ_URL = "https://api.groq.com/openai/v1"


def _local_endpoint() -> dict:
    """The in-tool model as an endpoint, or {} when it is not configured."""
    try:
        d = _distill.available()
    except Exception:  # noqa: BLE001
        d = {"available": False}
    if not d.get("available"):
        return {}
    r = _distill.resolve()
    return {"provider": "local", "base_url": r["base_url"],
            "model": r["model"], "api_key": r["api_key"], "rotate": False}


def _endpoints() -> list[dict]:
    """Candidate endpoints, best first.

    The configured in-tool model comes first by default: it is free at the point
    of use, so a keyed cloud lane should not be spent on a task the local model
    can do. Groq stays as the fallback for when the local endpoint is down, and
    can be promoted with WEBSEARCH_SYNTH_PROVIDER.

    WEBSEARCH_SYNTH_PROVIDER: local (default) | groq | auto
      local - the in-tool model first, Groq only if it fails
      groq  - Groq first, in-tool model only if it fails
      auto  - whichever is actually configured, in-tool model preferred
    """
    prefer = (os.environ.get("WEBSEARCH_SYNTH_PROVIDER") or "local").strip().lower()
    groq = ({"provider": "groq", "base_url": GROQ_URL,
             "model": GROQ_SMART_MODEL, "api_key": "", "rotate": True}
            if _groq_keys.has_keys else None)
    local = _local_endpoint() or None

    if prefer == "groq":
        order = [groq, local]
    else:
        order = [local, groq]
    return [e for e in order if e]


def _endpoint() -> dict:
    eps = _endpoints()
    return eps[0] if eps else {}


async def _chat(ep: dict, messages: list[dict], *, max_tokens: int,
                attempts: int = 2, timeout: float = 25) -> tuple[str, str, str, int]:
    """One OpenAI-compatible completion, with retries. Returns (text, finish, err, attempts).

    Same failure taxonomy as the distiller: transport errors and 5xx/429 are
    retried, a non-200 that is not transient is permanent, and a 200 with empty
    content is reported rather than passed on as an empty answer.
    """
    if not ep:
        return "", "", "no model endpoint configured", 0
    url = ep["base_url"].rstrip("/") + "/chat/completions"
    c = get_http_client()
    last = "no attempt made"
    # A reasoning model spends tokens on reasoning_content BEFORE emitting
    # content, so a budget that is right for the answer alone comes back as a
    # 200 with empty content and finish_reason=length. Escalate the budget when
    # that happens instead of reporting a mystery failure.
    budget = max_tokens
    ceiling = min(max(max_tokens * 4, 8192), 32_000)
    for attempt in range(1, max(1, attempts) + 1):
        key = await _groq_keys.next() if ep.get("rotate") else ep.get("api_key", "")
        try:
            resp = await c.post(
                url,
                headers={"Authorization": f"Bearer {key}",
                         "Content-Type": "application/json"},
                json={"model": ep["model"], "messages": messages,
                      # 0.0: same sources must yield the same text. Sampling
                      # variance in a factual answer is invisible noise.
                      "temperature": 0.0, "max_tokens": budget},
                timeout=timeout)
        except Exception as e:  # noqa: BLE001
            last = f"{type(e).__name__}: {e}"
            continue
        if resp.status_code != 200:
            last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code in (429, 500, 502, 503, 504):
                continue
            return "", "", last, attempt
        try:
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            text = ((choice.get("message") or {}).get("content") or "").strip()
            finish = choice.get("finish_reason") or ""
        except Exception as e:  # noqa: BLE001
            last = f"unparseable response: {type(e).__name__}: {e}"
            continue
        if text:
            return text, finish, "", attempt
        last = (f"empty content (finish_reason={finish}, max_tokens={budget})"
                if finish else f"empty content (max_tokens={budget})")
        if budget < ceiling:
            budget = min(budget * 2, ceiling)
            logger.info("synthesis: empty content, raising max_tokens to %d", budget)
    return "", "", last, max(1, attempts)


async def _synthesize(query: str, r: dict, depth: int, length: str = "short",
                      use_cache: bool = True) -> dict:
    """Produce the answer the caller actually reads.

    The returned dict always says what happened. That is the design goal: a
    synthesis that quietly did not run is indistinguishable from a bad answer,
    so `skipped` and `error` are first-class outcomes, `truncated` marks an
    answer the model ran out of budget for, and `diversity` reports the source
    mix the answer was built from.
    """
    s = synthesis_settings()
    pool = list(r.get("fetched_content") or r.get("results") or [])
    # A source the model cannot read is not a source. enrich() deliberately keeps
    # dead links in the pool at a demoted score so the caller still sees them, but
    # feeding one to the synthesis produced a citation [6] pointing at a page with
    # no text - an unciteable claim dressed as a citation.
    unreadable = [x for x in pool
                  if (x.get("dead") or x.get("success") is False)
                  and not (x.get("content") or x.get("snippet"))]
    pool = [x for x in pool if not ((x.get("dead") or x.get("success") is False)
                                    and not (x.get("content") or x.get("snippet")))]
    # Syndicated copies next: five sites running the same wire story is one
    # source, and citing all five makes a single origin look corroborated.
    pool, dedup = drop_near_duplicates(pool)
    limit = max(3, min(depth * 2, s["max_sources"]))
    top, div = cap_per_domain(pool, limit=limit, per_domain=s["domain_cap"],
                              min_sources=s["min_sources"])
    if not top:
        return {"available": True, "skipped": "no results to synthesise on",
                "diversity": div, "duplicates_removed": dedup["removed"],
                "unreadable_sources": len(unreadable)}

    def _freshness(x: dict) -> str:
        """How old the source is, stated precisely.

        Publication and last-modified are both shown when both exist. Either one
        alone is misleading: a wiki's "published" is its article creation date,
        so a page edited today can read as twenty years old, while "modified"
        alone makes a typo fix look like fresh reporting.
        """
        pub = str(x.get("published") or "").strip()
        mod = str(x.get("modified") or "").strip()
        if pub and mod and mod != pub:
            return f"{pub} (last modified {mod})"
        if pub:
            return pub
        if mod:
            return f"not stated (last modified {mod})"
        return "not stated"

    def _line(i: int, x: dict) -> str:
        head = (f"[{i + 1}]\nSource: {x.get('url', '')}\nTitle: {x.get('title', '')}"
                f"\nPublished: {_freshness(x)}")
        body = (x.get("content") or x.get("snippet") or "")[:s["snippet_chars"]]
        return f"{head}\nContent: {body}"

    ctx = "\n\n".join(_line(i, x) for i, x in enumerate(top))

    # The newest date among the sources. A caller asking "latest" needs to know
    # whether the freshest evidence is from last week or from 2023, and that is
    # not something the prose reliably states.
    _pubs = sorted({str(x.get("published")).strip() for x in top
                    if str(x.get("published") or "").strip()})
    _mods = sorted({str(x.get("modified")).strip() for x in top
                    if str(x.get("modified") or "").strip()})
    as_of = (_pubs or _mods or [""])[-1]
    as_of_kind = "published" if _pubs else ("modified" if _mods else "")

    excerpts = [(x.get("content") or x.get("snippet") or "") for x in top]
    # Key includes the source text itself, not just the URLs: an answer cached
    # against an old revision of a page would otherwise be served as current.
    _h = hashlib.sha256(
        (prompts.SYNTHESIS_VERSION + "|" + query + "|" + length + "|"
         + "|".join(excerpts)).encode()).hexdigest()[:20]
    cache_key = "synthesis://" + _h
    try:
        hit = (await _cache_mod.get_cached(cache_key, extraction_type="synthesis",
                                           ttl=3600)) if use_cache else None
        if hit and hit.get("content"):
            cached = json.loads(hit["content"])
            cached["cached"] = True
            return cached
    except Exception as e:  # noqa: BLE001
        logger.debug("synthesis cache read failed: %s", e)

    eps = _endpoints()
    if not eps:
        return {"available": False,
                "skipped": "no model endpoint: set GROQ_API_KEYS for Groq, or "
                           "DISTILL_API_KEY (with DISTILL_BASE_URL/DISTILL_MODEL) "
                           "for any OpenAI-compatible endpoint",
                "diversity": div}

    _t0 = time.perf_counter()
    system = prompts.synthesis_system(query, len(top), long=(length == "long"))
    answer = finish = ""
    err = "no attempt made"
    attempts = 0
    ep: dict = eps[0]
    tried: list[dict] = []
    for candidate in eps:
        ep = candidate
        answer, finish, perr, att = await _chat(
            ep,
            [{"role": "system", "content": system},
             {"role": "user", "content": prompts.synthesis_user(query, ctx)}],
            max_tokens=s["max_tokens"], attempts=2, timeout=30)
        attempts += att
        tried.append({"provider": ep["provider"], "model": ep["model"],
                      "ok": bool(answer), "error": perr if not answer else ""})
        if answer:
            err = ""
            break
        err = perr

    if not answer:
        return {"available": True, "provider": ep["provider"],
                "model": ep["model"], "error": err, "attempts": attempts,
                "endpoints_tried": tried, "diversity": div,
                "note": "Results are still returned below; only the written "
                        "answer failed."}

    sources = [{"n": i + 1, "title": x.get("title", ""), "url": x.get("url", "")}
               for i, x in enumerate(top)]
    truncated = finish == "length"
    # Which sources the answer actually leans on. Available sources are not
    # cited sources, and a single cited source is a single point of view even
    # when eight were offered.
    cited = sorted({int(m) for m in re.findall(r"\[(\d+)\]", answer)})
    body = answer
    # Eight sources offered, one cited: the answer is a single point of view and
    # the reader cannot tell that from the text alone.
    if len(cited) == 1 and len(sources) > 1:
        body += ("\n\n[This answer cites one source of the "
                 f"{len(sources)} retrieved; treat it as a single point of view.]")

    # Deterministic figure check: no model call, no opinion. A number in the
    # answer that appears nowhere in the cited text was produced by the model,
    # so it is either a recomputation or an invention - both worth surfacing.
    unverified: list[str] = []
    try:
        # Strip citation markers first: "[3]" is a number that by definition
        # does not appear in the source, and counting it would flag every
        # correctly cited answer.
        _body_only = re.sub(r"\[\d+\]", " ", answer)
        _g = _distill.grounding_check(_body_only, "\n\n".join(excerpts))
        unverified = _g.get("numbers_unverified", [])
    except Exception as e:  # noqa: BLE001
        logger.debug("number grounding failed: %s", e)
    if truncated:
        # Say it in the text as well as the field: half a sentence otherwise
        # reads like a complete thought.
        body += (f"\n\n[truncated at {s['max_tokens']} tokens - raise "
                 f"WEBSEARCH_SYNTH_TOKENS for a longer answer]")
    if unverified:
        body += (f"\n\n[{len(unverified)} figure(s) in this answer appear in no "
                 f"cited source: {', '.join(unverified[:5])}]")
    body += "\n\n---\n\n## Sources\n\n" + "\n".join(
        f"[{x['n']}] {x['title']} - {x['url']}" for x in sources)

    try:
        grounded = await verify_grounding(answer, sources, ep, excerpts)
    except Exception as e:  # noqa: BLE001
        grounded = {"flagged": [], "ok": None, "error": str(e)}

    result = {
        "available": True,
        "answer": body,
        "sources": sources,
        "provider": ep["provider"],
        "model": ep["model"],
        "prompt_version": prompts.SYNTHESIS_VERSION,
        "temperature": 0.0,
        "max_tokens": s["max_tokens"],
        "finish_reason": finish,
        "truncated": truncated,
        "sources_used": len(top),
        "duplicates_removed": dedup["removed"],
        "unreadable_sources": len(unreadable),
        "citations_used": len(cited),
        "as_of": as_of,
        "as_of_kind": as_of_kind,
        "figures_unverified": unverified,
        "duration_ms": int((time.perf_counter() - _t0) * 1000),
        "diversity": div,
        "grounded": grounded,
        "attempts": attempts,
        "endpoints_tried": tried,
    }
    try:
        if use_cache:
            await _cache_mod.set_cached(cache_key, json.dumps(result),
                                        extraction_type="synthesis",
                                        title=query[:100], ttl=3600)
    except Exception as e:  # noqa: BLE001
        logger.debug("synthesis cache write failed: %s", e)
    return result


# ───────────────────── gap-driven second round ─────────────────────
# The first pass retrieves one pool and answers from it. When the answer itself
# says a part of the question is unanswered, the system was previously done -
# even though the subqueries it generated are sitting right there in the
# response, unused. One extra round on the subquery that covers the gap is the
# cheapest real capability gain available: no new engine, no new key.

FOLLOW_UP_TIME_BUDGET_MS = 75_000   # don't start a round a client will time out waiting for
FOLLOW_UP_MAX_NEW_SOURCES = 6

_GAP_PHRASES = ("unsupported", "not stated", "does not state", "do not state",
                "cannot be determined", "no source", "without a source",
                "insufficient evidence", "not specified in")


def _gap_reason(synth: dict) -> str:
    """Why a second round might help, or '' when the answer is self-sufficient.

    Deliberately conservative: a follow-up costs 20-40s, so it only runs when
    the first answer *itself* advertises a weakness, or when it rests on a single
    citation while several were available.
    """
    if not synth or not synth.get("available"):
        return ""
    if synth.get("error"):
        return "the first pass produced no answer"
    cited = synth.get("citations_used") or 0
    if cited <= 1:
        return f"the answer rests on {cited} source(s)"
    grounded = synth.get("grounded") or {}
    if grounded.get("ok") is False and grounded.get("flagged"):
        return "a claim could not be tied to the source that supposedly supports it"
    text = (synth.get("answer") or "").lower()
    for phrase in _GAP_PHRASES:
        if phrase in text:
            return f"the answer states part of the question is unanswered ({phrase!r})"
    return ""


def _follow_up_query(r: dict) -> tuple[str, str]:
    """Pick the subquery most likely to fill the gap, and say why it was picked."""
    subs = r.get("subqueries") or []
    gap_words = ("limitation", "critic", "issue", "counter", "gap", "method",
                 "evidence", "risk", "drawback", "alternativ")
    for sub in subs:
        goal = str(sub.get("researchGoal") or "").lower()
        q = str(sub.get("query") or "").strip()
        if q and any(w in goal for w in gap_words):
            return q, f"subquery targeting {sub.get('researchGoal', '')[:80]}"
    for sub in subs:
        q = str(sub.get("query") or "").strip()
        if q:
            return q, "first unused subquery"
    fu = str(r.get("follow_up") or "").strip()
    if fu:
        return fu, "engine follow_up suggestion"
    rel = r.get("related_queries") or []
    if rel:
        return str(rel[0]).strip(), "engine related_queries"
    return "", ""


def _merge_round(base: dict, extra: dict) -> int:
    """Merge a second round into the first, by URL. Returns how many were added."""
    seen = {x.get("url") for x in (base.get("results") or []) if x.get("url")}
    added = 0
    for item in extra.get("results") or []:
        u = item.get("url")
        if not u or u in seen:
            continue
        seen.add(u)
        base.setdefault("results", []).append(item)
        added += 1
    if extra.get("fetched_content"):
        base["fetched_content"] = list(base.get("fetched_content") or []) + list(
            extra["fetched_content"])
    return added


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

        if name == "form_fill_submit":
            return _res(await _flows.form_fill_submit(
                fields=list(arguments.get("fields") or []),
                session_id=str(arguments.get("session_id") or ""),
                url=str(arguments.get("url") or ""),
                submit_selector=str(arguments.get("submit_selector") or ""),
                wait_state=str(arguments.get("wait_state") or "networkidle"),
                read_selector=str(arguments.get("read_selector") or ""),
                timeout=safe_float(arguments.get("timeout", 20.0), 20.0)))

        if name == "paginate":
            return _res(await _flows.paginate(
                next_selector=str(arguments.get("next_selector") or ""),
                session_id=str(arguments.get("session_id") or ""),
                url=str(arguments.get("url") or ""),
                max_pages=safe_int(arguments.get("max_pages", 5)),
                read_selector=str(arguments.get("read_selector") or ""),
                wait_state=str(arguments.get("wait_state") or "networkidle"),
                wait_ms=safe_int(arguments.get("wait_ms", 600)),
                stop_when_unchanged=bool(arguments.get("stop_when_unchanged", True)),
                timeout=safe_float(arguments.get("timeout", 20.0), 20.0)))

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
            _t_search = time.perf_counter()
            query = str(arguments.get("query", "")).strip()
            if not query:
                return _res({"success": False, "error": "query must not be empty"})
            # Depth default follows whether a model lane exists: fetching full
            # pages only pays off when something reads them. Explicit depth
            # always wins, and WEBSEARCH_DEFAULT_DEPTH pins the default.
            _model_ok = _groq_keys.has_keys or bool(_distill.available()["available"])
            _depth_default = default_depth(_model_ok)
            depth = max(DEPTH_MIN, min(DEPTH_MAX,
                        safe_int(arguments.get("depth", _depth_default),
                                 _depth_default)))
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

            # Name the sources that produced no text, instead of leaving the
            # caller to notice an empty `content` field halfway down a list.
            # They are excluded from synthesis, so the answer is not built on
            # them, but a caller reading the raw results deserves to know.
            unreadable = [
                {"url": x.get("url", ""), "title": x.get("title", ""),
                 "reason": (x.get("error") or ("dead link" if x.get("dead")
                            else "no text extracted"))[:120]}
                for x in (r.get("fetched_content") or [])
                if not (x.get("content") or "").strip()]
            if unreadable:
                r["unreadable"] = unreadable
            # Skip Groq synthesis when GAI already returned a full answer
            skip_synthesis = google_ai_only and r.get("ai_answer")
            if r.get("success") and bool(arguments.get("synthesize", True)) \
                    and (r.get("results") or r.get("fetched_content")) \
                    and not skip_synthesis:
                r["synthesis"] = await _synthesize(
                    query, r, depth,
                    length=str(arguments.get("synthesize_length", "short")))
                if r["synthesis"].get("error"):
                    r["synthesis_error"] = r["synthesis"]["error"]

                # ── one gap-driven round, if it is worth the wait ──────
                # Only when the first answer admits a weakness, only when there
                # is time left for a client to receive the result, and only once.
                _elapsed_ms = int((time.perf_counter() - _t_search) * 1000)
                if (bool(arguments.get("follow_up", True))
                        and not r["synthesis"].get("cached")
                        and _elapsed_ms < FOLLOW_UP_TIME_BUDGET_MS):
                    why = _gap_reason(r["synthesis"])
                    fq, picked = _follow_up_query(r) if why else ("", "")
                    if fq:
                        try:
                            # Same engines, narrower count: this is a top-up, not
                            # a second full search.
                            extra = await search_multi(
                                fq, count=min(count, 8), depth=depth,
                                search_type=str(arguments.get("search_type", "auto")),
                                timelimit=str(arguments.get("timelimit", "")),
                                safesearch=str(arguments.get("safesearch", "moderate")),
                                language=lang, cdp_url=HELIUM_CDP)
                            if extra.get("success"):
                                if depth >= 2 and extra.get("results"):
                                    try:
                                        got = await enrich(
                                            extra["results"], fq, depth=depth,
                                            cdp_url=HELIUM_CDP, count=8, language=lang)
                                        if got.get("fetched_content"):
                                            extra["fetched_content"] = got["fetched_content"]
                                    except Exception as e:  # noqa: BLE001
                                        logger.warning("follow-up enrich failed: %s", e)
                                added = _merge_round(r, extra)
                                r["follow_ups"] = [{
                                    "query": fq, "why": why, "picked_by": picked,
                                    "sources_added": added,
                                }]
                                if added:
                                    r["synthesis"] = await _synthesize(
                                        query, r, depth,
                                        length=str(arguments.get("synthesize_length", "short")),
                                        use_cache=False)
                                    r["synthesis"]["round"] = 2
                                    r["synthesis"]["rounds_reason"] = why
                        except Exception as e:  # noqa: BLE001
                            r["follow_up_error"] = f"{type(e).__name__}: {e}"
            elif r.get("success") and skip_synthesis:
                r["synthesis"] = {"skipped": "google_ai_only returned its own "
                                             "answer; not synthesising over it"}
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
                        cache_bust=bool(arguments.get("cache_bust", False)),
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
                            cache_bust=bool(arguments.get("cache_bust", False)),
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


async def _synthesis_selfcheck() -> None:
    """Prove the honesty fields, with the model replaced by stubs.

    These are the properties that make the answer trustworthy, and every one of
    them is something a refactor could quietly drop: no endpoint must say so
    rather than answer, an empty completion must be an error rather than an
    empty answer, and a cut-off answer must look cut off.
    """
    checks: list[tuple[str, bool]] = []
    fake = {"results": [
        {"url": "https://a.com/1", "title": "A1", "snippet": "alpha"},
        {"url": "https://a.com/2", "title": "A2", "snippet": "alpha two"},
        {"url": "https://a.com/3", "title": "A3", "snippet": "alpha three"},
        {"url": "https://b.com/1", "title": "B1", "snippet": "beta"},
        {"url": "https://b.com/2", "title": "B2", "snippet": "beta two"},
    ]}

    async def ok_chat(ep, messages, *, max_tokens, attempts=2, timeout=25):
        return "Answer [1].", "stop", "", 1

    async def stub_verify(answer, sources, ep, excerpts=None):
        return {"flagged": [], "ok": True, "checked_by": "stub", "advisory": True}

    real = {"_endpoints": _endpoints, "_chat": _chat,
            "verify_grounding": verify_grounding}
    one = [{"provider": "stub", "base_url": "http://stub/v1", "model": "m",
            "api_key": "k", "rotate": False}]
    try:
        globals()["_endpoints"] = lambda: []
        out = await _synthesize("selfcheck-q", fake, 2, use_cache=False)
        checks.append(("no endpoint: says skipped", bool(out.get("skipped"))))
        checks.append(("no endpoint: not marked available",
                       out.get("available") is False))

        globals()["_endpoints"] = lambda: one
        globals()["_chat"] = ok_chat
        globals()["verify_grounding"] = stub_verify
        out = await _synthesize("selfcheck-q", fake, 2, use_cache=False)
        answer = out.get("answer", "")
        checks.append(("answer has citations", "[1]" in answer))
        checks.append(("answer has sources block", "## Sources" in answer))
        checks.append(("prompt version attached",
                       out.get("prompt_version") == prompts.SYNTHESIS_VERSION))
        checks.append(("temperature pinned to 0", out.get("temperature") == 0.0))
        checks.append(("domain cap applied",
                       out["diversity"]["n_domains"] == 2 and
                       all(n <= 2 for n in out["diversity"]["duplicate_domains"].values())))
        checks.append(("not truncated when finish_reason=stop",
                       out.get("truncated") is False))
        checks.append(("grounding is advisory", out["grounded"].get("advisory") is True))

        async def empty_chat(ep, messages, *, max_tokens, attempts=2, timeout=25):
            return "", "length", "empty content", 2
        globals()["_chat"] = empty_chat
        out = await _synthesize("selfcheck-q", fake, 2, use_cache=False)
        checks.append(("empty completion is an error", bool(out.get("error"))))
        checks.append(("empty completion invents no answer",
                       "answer" not in out or not out.get("answer")))

        async def cut_chat(ep, messages, *, max_tokens, attempts=2, timeout=25):
            return "Half a sentence", "length", "", 1
        globals()["_chat"] = cut_chat
        out = await _synthesize("selfcheck-q", fake, 2, use_cache=False)
        checks.append(("truncated flag set", out.get("truncated") is True))
        checks.append(("truncation visible in the text",
                       "truncated at" in out.get("answer", "")))

        async def money_chat(ep, messages, *, max_tokens, attempts=2, timeout=25):
            return "It costs $9999 [1].", "stop", "", 1
        globals()["_chat"] = money_chat
        out = await _synthesize("selfcheck-q", fake, 2, use_cache=False)
        checks.append(("invented figure is detected", out.get("figures_unverified") == ["9999"]))
        checks.append(("invented figure is visible in the text",
                       "appear in no cited source" in out.get("answer", "")))
        checks.append(("citation markers are not treated as figures",
                       all(x not in ("1", "2") for x in out.get("figures_unverified", []))))

        async def one_cite_chat(ep, messages, *, max_tokens, attempts=2, timeout=25):
            return "Only one view [1].", "stop", "", 1
        globals()["_chat"] = one_cite_chat
        out = await _synthesize("selfcheck-q", fake, 2, use_cache=False)
        checks.append(("single cited source is flagged",
                       "single point of view" in out.get("answer", "")))
        checks.append(("citation count reported", out.get("citations_used") == 1))

        globals()["_chat"] = ok_chat
        with_dead = {"results": [
            {"url": "https://a.com/1", "title": "A1", "snippet": "alpha"},
            {"url": "https://b.com/x", "title": "Scholar stub", "snippet": "",
             "dead": True, "success": False},
        ]}
        out = await _synthesize("selfcheck-q", with_dead, 2, use_cache=False)
        checks.append(("unreadable sources are excluded", out.get("unreadable_sources") == 1))
        checks.append(("unreadable source is not cited",
                       "b.com/x" not in (out.get("answer") or "")))

        synd = {"results": [
            {"url": "https://a.com/x", "title": "Rust 1.90 released with new borrow checker", "snippet": "one"},
            {"url": "https://b.com/y", "title": "Rust 1.90 released with new borrow checker", "snippet": "two"},
            {"url": "https://c.com/z", "title": "A different story about Go", "snippet": "three"},
        ]}
        out = await _synthesize("selfcheck-q", synd, 2, use_cache=False)
        checks.append(("syndicated copies are dropped", out.get("duplicates_removed") == 1))

        # Verifier robustness: a model that answers in prose must not turn into a
        # crash or a silent "ok".
        async def prose_chat(ep, messages, *, max_tokens, attempts=2, timeout=25):
            return "Looks fine to me, no issues.", "stop", "", 1
        globals()["_chat"] = prose_chat
        # Call the captured original: the module global is stubbed above, so
        # using the name here would test the stub against itself.
        g = await real["verify_grounding"](
            "Claim [1].", [{"n": 1, "title": "t", "url": "u"}], one[0], ["source text"])
        checks.append(("prose verifier reply yields ok=None, not a crash",
                       g.get("ok") is None))
        checks.append(("prose verifier reply is reported as unparseable",
                       bool(g.get("error"))))

        globals()["_chat"] = ok_chat
        out = await _synthesize("selfcheck-q", {"results": []}, 2, use_cache=False)
        checks.append(("no results: skipped, no crash", bool(out.get("skipped"))))

    finally:
        globals().update(real)

    # ── follow-up selection: pure functions, no model, no network ──
    good = {"available": True, "citations_used": 4, "answer": "It is 3.4 [1].",
            "grounded": {"ok": True, "flagged": []}}
    weak = {"available": True, "citations_used": 1, "answer": "Something [1].",
            "grounded": {"ok": True, "flagged": []}}
    vague = {"available": True, "citations_used": 4,
             "answer": "Part of this is unsupported by the sources [1][2].",
             "grounded": {"ok": True, "flagged": []}}
    flagged = {"available": True, "citations_used": 4, "answer": "X [1].",
               "grounded": {"ok": False, "flagged": [{"sentence": "X"}]}}
    checks.append(("a sufficient answer asks for no follow-up", _gap_reason(good) == ""))
    checks.append(("a single-source answer asks for one", bool(_gap_reason(weak))))
    checks.append(("an answer admitting a gap asks for one", bool(_gap_reason(vague))))
    checks.append(("a flagged claim asks for one", bool(_gap_reason(flagged))))
    checks.append(("no synthesis means no follow-up", _gap_reason({}) == ""))
    checks.append(("an error asks for one", bool(_gap_reason({"available": True, "error": "x"}))))

    subs = {"subqueries": [
        {"query": "what is uv", "researchGoal": "establish background"},
        {"query": "known limitations of uv", "researchGoal": "identify limitations"},
    ]}
    q, why = _follow_up_query(subs)
    checks.append(("gap-targeting subquery preferred", "limitations" in q))
    checks.append(("picking reason is stated", "limitation" in why.lower()))
    checks.append(("falls back to engine follow_up",
                   _follow_up_query({"follow_up": "next thing"})[0] == "next thing"))
    checks.append(("no suggestion means no query", _follow_up_query({})[0] == ""))

    base = {"results": [{"url": "a"}], "fetched_content": [{"url": "a", "content": "x"}]}
    added = _merge_round(base, {"results": [{"url": "a"}, {"url": "b"}],
                                "fetched_content": [{"url": "b", "content": "y"}]})
    checks.append(("merge adds only new urls", added == 1))
    checks.append(("merge keeps both result sets", len(base["results"]) == 2))
    checks.append(("merge keeps both fetched sets", len(base["fetched_content"]) == 2))

    for name, ok in checks:
        print(f"{'OK' if ok else 'FAIL'}  {name}")
    print(f"server synthesis selfcheck: {sum(1 for _, v in checks if v)}/{len(checks)} OK")


async def main():
    asyncio.create_task(_parent_watchdog())
    asyncio.create_task(_warmup_gai())
    try:
        async with stdio_server() as (rs, ws):
            await server.run(rs, ws, server.create_initialization_options())
    finally:
        await close_http_client()

if __name__ == "__main__":
    import sys
    if "--selfcheck" in sys.argv:
        asyncio.run(_synthesis_selfcheck())
    else:
        asyncio.run(main())


def cli_main():
    """Console-script entry point (sync wrapper around the async main).

    Used by the packaged distribution: `websearch-mcp` on PATH, and by MCP
    clients that launch the module as a stdio server.
    """
    import asyncio
    asyncio.run(main())
