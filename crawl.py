from __future__ import annotations

import asyncio
from typing import Any

from crawl4ai import AsyncWebCrawler, BrowserConfig, CrawlerRunConfig, CacheMode, GeolocationConfig
from crawl4ai.deep_crawling import BFSDeepCrawlStrategy, DFSDeepCrawlStrategy
from crawl4ai.content_scraping_strategy import LXMLWebScrapingStrategy
from crawl4ai.markdown_generation_strategy import DefaultMarkdownGenerator
from crawl4ai.content_filter_strategy import PruningContentFilter, BM25ContentFilter
from crawl4ai.extraction_strategy import JsonCssExtractionStrategy
from crawl4ai import CosineStrategy
from embed import _embed, _cosine_sim


async def crawl_url(
    url: str, max_depth: int = 1, max_pages: int = 10, extract_links: bool = True,
    js_code: str = "", wait_for: str = "", session_id: str = "", locale: str = "",
    timezone_id: str = "", check_robots_txt: bool = False, strategy: str = "bfs",
    screenshot: bool = False, pdf: bool = False, proxy: str = "", magic: bool = True,
    flatten_shadow_dom: bool = True, process_iframes: bool = True,
    scan_full_page: bool = True, wait_for_images: bool = True,
    word_count_threshold: int = 30, page_timeout: int = 60000,
    max_retries: int = 2, browser_type: str = "chromium",
    user_agent: str = "", headers: dict | None = None,
    extra_args: list | None = None,
    geolocation_lat: float = 0, geolocation_lng: float = 0,
    delay_before_return_html: float = 0.1,
    stream: bool = False, capture_mhtml: bool = False,
    capture_network_requests: bool = False, capture_console_messages: bool = False,
    exclude_external_links: bool = False, exclude_social_media_links: bool = False,
    exclude_domains: list | None = None,
    css_selector: str = "", target_elements: list | None = None,
    memory_saving_mode: bool = False,
    remove_forms: bool = False, keep_data_attributes: bool = False,
    method: str = "GET", score_links: bool = False,
    simulate_user: bool = True, override_navigator: bool = True,
    cdp_url: str = "", adjust_viewport_to_content: bool = False,
    log_console: bool = False,
    content_filter: str = "", filter_query: str = "",
    css_extract: dict | None = None,
    bypass_cache: bool = False,
    exclude_all_images: bool = False,
    exclude_external_images: bool = False,
    min_content_chars: int = 0,
) -> dict:
    browser_kw: dict[str, Any] = {
        "headless": True, "verbose": False, "ignore_https_errors": True,
        "enable_stealth": True, "text_mode": True, "light_mode": True,
        "avoid_ads": True, "avoid_css": True,
        "viewport_width": 1920, "viewport_height": 1080,
        "browser_type": browser_type,
    }
    if cdp_url:
        browser_kw["cdp_url"] = cdp_url
    if proxy:
        browser_kw["proxy"] = {"server": proxy}
    if user_agent:
        browser_kw["user_agent"] = user_agent
    if headers:
        browser_kw["headers"] = headers
    if extra_args:
        browser_kw["extra_args"] = extra_args
    if memory_saving_mode:
        browser_kw["memory_saving_mode"] = True
    cfg = BrowserConfig(**browser_kw)

    run_kw: dict[str, Any] = {
        "scraping_strategy": LXMLWebScrapingStrategy(), "verbose": False,
        "cache_mode": CacheMode.ENABLED,
        "remove_overlay_elements": True, "remove_consent_popups": True,
        "simulate_user": simulate_user, "override_navigator": override_navigator,
        "magic": magic,
        "flatten_shadow_dom": flatten_shadow_dom,
        "process_iframes": process_iframes,
        "excluded_tags": ["script", "style", "nav", "footer"],
        "excluded_selector": "aside, .sidebar, .advertisement",
        "word_count_threshold": word_count_threshold,
        "page_timeout": page_timeout, "max_retries": max_retries,
        "scan_full_page": scan_full_page, "wait_for_images": wait_for_images,
        "screenshot": screenshot, "pdf": pdf,
        "delay_before_return_html": delay_before_return_html,
        "stream": stream, "method": method, "score_links": score_links,
        "markdown_generator": DefaultMarkdownGenerator(),
    }
    if css_extract:
        run_kw["extraction_strategy"] = JsonCssExtractionStrategy(
            schema=css_extract)
    if content_filter:
        if content_filter == "cosine":
            if filter_query:
                run_kw["extraction_strategy"] = CosineStrategy(
                    semantic_filter=filter_query)
        else:
            cf_map = {"pruning": PruningContentFilter(threshold=word_count_threshold / 100, threshold_type="dynamic"),
                      "bm25": BM25ContentFilter(user_query=filter_query, threshold=0.15),
                      "bm25_hq": BM25ContentFilter(user_query=filter_query, threshold=0.1, perform_stemming=True)}
            cf = cf_map.get(content_filter)
            if cf:
                run_kw["markdown_generator"] = DefaultMarkdownGenerator(content_filter=cf)
    if js_code:
        run_kw["js_code"] = js_code
    if wait_for:
        run_kw["wait_for"] = wait_for
    if session_id:
        run_kw["session_id"] = session_id
    if locale:
        run_kw["locale"] = locale
    if timezone_id:
        run_kw["timezone_id"] = timezone_id
    if check_robots_txt:
        run_kw["check_robots_txt"] = True
    if geolocation_lat or geolocation_lng:
        run_kw["geolocation"] = GeolocationConfig(latitude=geolocation_lat, longitude=geolocation_lng)
    if capture_mhtml:
        run_kw["capture_mhtml"] = True
    if capture_network_requests:
        run_kw["capture_network_requests"] = True
    if capture_console_messages:
        run_kw["capture_console_messages"] = True
    if exclude_external_links:
        run_kw["exclude_external_links"] = True
    if exclude_social_media_links:
        run_kw["exclude_social_media_links"] = True
    if exclude_domains:
        run_kw["exclude_domains"] = exclude_domains
    if css_selector:
        run_kw["css_selector"] = css_selector
    if target_elements:
        run_kw["target_elements"] = target_elements
    if remove_forms:
        run_kw["remove_forms"] = True
    if keep_data_attributes:
        run_kw["keep_data_attributes"] = True
    if max_depth > 0:
        sc = DFSDeepCrawlStrategy if strategy == "dfs" else BFSDeepCrawlStrategy
        run_kw["deep_crawl_strategy"] = sc(
            max_depth=max_depth, include_external=False, max_pages=max_pages)
    if adjust_viewport_to_content:
        run_kw["adjust_viewport_to_content"] = True
    if log_console:
        run_kw["log_console"] = True
    if bypass_cache:
        run_kw["cache_mode"] = CacheMode.BYPASS
    if min_content_chars > 0:
        # E4c: adaptive crawl — prune thin pages (nav shells, redirect
        # stubs, empty JS renders) from the result set; the crawl itself
        # is unchanged, only what counts as a deliverable narrows.
        pass
    if exclude_all_images:
        run_kw["exclude_all_images"] = True
    if exclude_external_images:
        run_kw["exclude_external_images"] = True
    run_config = CrawlerRunConfig(**run_kw)
    async with AsyncWebCrawler(config=cfg) as crawler:
        # ponytail: per-page timeout * page budget as an overall cap,
        # generous ceiling so legitimately long crawls still finish
        overall_timeout = max(60, int(max_pages * page_timeout / 1000))
        try:
            results = await asyncio.wait_for(
                crawler.arun(url, config=run_config), timeout=overall_timeout)
        except asyncio.TimeoutError:
            return {"success": False, "url": url,
                    "error": f"crawl timed out after {overall_timeout}s"}
        if isinstance(results, list):
            pages = []
            combined = []
            pruned = 0
            for r in results:
                md = r.markdown or ""
                if min_content_chars > 0 and len(md) < min_content_chars:
                    pruned += 1
                    continue
                combined.append(md)
                links = (r.links or {}) if extract_links else {}
                pe: dict[str, Any] = {"url": r.url, "markdown_len": len(md),
                                       "success": r.success}
                if links:
                    pe["links"] = {"internal": len(links.get("internal", [])),
                                   "external": len(links.get("external", []))}
                if r.screenshot:
                    pe["has_screenshot"] = True
                if r.pdf:
                    pe["has_pdf"] = True
                pages.append(pe)
            if len(pages) > 1:
                texts = [(p.get("url", "") or "")[:200] for p in pages]
                item_emb = await _embed(texts, "passage")
                if item_emb and all(e is not None for e in item_emb):
                    deduped_pages = []
                    deduped_indices = []
                    seen_emb: list[list[float]] = []
                    for i, (p, emb) in enumerate(zip(pages, item_emb)):
                        is_dup = any(_cosine_sim(emb, se) > 0.92 for se in seen_emb)
                        if not is_dup:
                            deduped_pages.append(p)
                            deduped_indices.append(i)
                            seen_emb.append(emb)
                    pages = deduped_pages
                    combined = [combined[i] for i in deduped_indices]
            return {"success": True, "url": url, "pages_crawled": len(pages),
                    "pages_pruned_thin": pruned,
                    "min_content_chars": min_content_chars,
                    "pages": pages,
                    "combined_markdown_len": sum(p["markdown_len"] for p in pages),
                    "combined_markdown": "\n\n---\n\n".join(combined)}
        md = results.markdown or ""
        pruned = 0
        if min_content_chars > 0 and len(md) < min_content_chars:
            pruned = 1
            md = ""
        links = getattr(results, 'links', None) or {}
        out: dict[str, Any] = {
            "success": True, "url": url, "status_code": results.status_code,
            "markdown": md,
            "pages_pruned_thin": pruned,
            "min_content_chars": min_content_chars,
            "links": {"internal": len(links.get("internal", [])),
                      "external": len(links.get("external", []))},
            "error": results.error_message or "",
        }
        if results.screenshot:
            out["has_screenshot"] = True
        if results.pdf:
            out["has_pdf"] = True
        if capture_console_messages and hasattr(results, 'console_messages'):
            out["console_messages"] = results.console_messages
        if capture_network_requests and hasattr(results, 'network_requests'):
            out["network_requests"] = results.network_requests
        return out
