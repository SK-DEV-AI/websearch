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

_groq_keys = _KeyRotator("GROQ_API_KEYS")

INSTRUCTIONS = """# WebSearch MCP

Multi-engine search + content extraction.

## When to use what

- **search** for finding information (9 engines, dedup+rerank+synthesis). **synthesize=false** when you only need raw results. Set `domain` for vertical search (code, academic, finance, health, travel, legal).
- **fetch** to read a specific page (auto-CDP fallback on blocked pages. PDF/EPUB/DOCX support).
- **screenshot** for LLM-readable page text (ARIA snapshot).
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
        Tool(name="ping",
            description="Lightweight connectivity check — verifies internet and key search endpoints are reachable. Use before expensive calls when connectivity is uncertain. No params needed. e.g. ping()",
            input_schema={"type": "object", "properties": {}},
        ),
        Tool(name="search",
            description="Multi-engine web search with dedup, reranking, and synthesis. depth=1 returns snippets with relevance_score+fetch_relevance per result. depth>=2 fetches full pages + re-ranks. synthesize=True (default) returns Groq answer with [N] citations. google_ai_only skips all engines for Google AI Mode. e.g. search(query='latest AI models', depth=1)",
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
                "google_ai_only": {"type": "boolean", "description": "Skip all other search engines, only use Google AI Mode for an AI-generated answer"},
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
                    },
                "required": ["url"]}),
          Tool(name="screenshot",
            description="ARIA accessibility snapshot (LLM-readable page text). e.g. screenshot(url='https://example.com')",
            input_schema={"type": "object", "properties": {
                "url": {"type": "string"},
                "max_chars": {"type": "integer", "default": 10000},
                "depth": {"type": "integer", "description": "ARIA snapshot tree depth limit"},
                "verbose": {"type": "boolean", "default": False, "description": "Show all ARIA roles (not just interactive)"},
                "start_line": {"type": "integer", "description": "1-based start line for snapshot text range"},
                "end_line": {"type": "integer", "description": "1-based end line (inclusive) for snapshot text range"}},
                "required": ["url"]}),
         Tool(name="wikipedia",
            description="Search Wikipedia: articles, summaries, categories, links, pageviews. e.g. wikipedia(query='Python', action='summary')",
            input_schema={"type": "object", "properties": {
                "action": {"type": "string", "enum": ["search","summary","summary_action","categories","links","pageviews"], "default": "search", "description": "search=find articles, summary=REST fast extract, summary_action=Action API+images, categories=list page cats, links=page links, pageviews=traffic stats"},
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
            description="Discover all pages on a website via sitemap XML (primary) and HTML link extraction (fallback). Returns the base domain, source type, total count, and an array of discovered URLs with metadata (last_modified, priority, changefreq when available). e.g. map_site(url='https://example.com')",
            input_schema={"type": "object", "properties": {
                "url": {"type": "string", "description": "Full URL of the site to map (e.g. https://example.com)"},
                "max_urls": {"type": "integer", "default": 1000, "description": "Cap on returned URLs"},
                "max_depth": {"type": "integer", "default": 0, "description": "Recursive link extraction depth (0=homepage only). Only used when no sitemap exists."},
                "include_sitemap": {"type": "boolean", "default": True, "description": "Try sitemap discovery first"},
                "include_links": {"type": "boolean", "default": True, "description": "Fall back to HTML link extraction when no sitemap"},
                "same_domain": {"type": "boolean", "default": True, "description": "Only include URLs from the same domain"},
                 "exclude_patterns": {"type": "array", "items": {"type": "string"}, "description": "Regex patterns to exclude matching URLs"}},
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
            if r is None:
                should_retry = True
            elif not r.get("success"):
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
                ) or (len(content) < 100 and (
                    # tiny-but-real pages (small HTML, text extracted) don't
                    # benefit from a browser re-render — skip the wasted CDP
                    # round-trip; big HTML with no text is a JS shell → retry
                    not r.get("raw_html_len") or r.get("raw_html_len", 0) > 5000))

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
                exclude_patterns=arguments.get("exclude_patterns"))
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

async def main():
    asyncio.create_task(_warmup_gai())
    try:
        async with stdio_server() as (rs, ws):
            await server.run(rs, ws, server.create_initialization_options())
    finally:
        await close_http_client()

if __name__ == "__main__":
    asyncio.run(main())
