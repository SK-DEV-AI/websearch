from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import trafilatura

from scrapling.fetchers import AsyncFetcher, AsyncStealthySession

from search_gai import _get_optimized_page, _cleanup_orphan_tabs
from security import SecurityError, validate_url as _validate_url
import cache as cache_mod
import focus as focus_mod

logger = logging.getLogger("fetch")

AsyncFetcher.configure(huge_tree=True)

# ── Cloudflare challenge detection ──────────────────────────────
# Covers old interstitial ("Checking your browser"), Managed Challenge
# (checkbox/Turnstile iframe), and Turnstile widget formats.

_CLOUDFLARE_DETECT_JS = """
() => {
    const t = document.title.toLowerCase();
    if (t.includes('just a moment') || t.includes('attention required') || t.includes('cloudflare')) return true;

    const u = window.location.href;
    if (u.includes('__cf_chl_') || u.includes('cf_chl_')) return true;

    if (window._cf_chl_opt || window._cf_chl_context || window.turnstile || window.__cfRLUnblockHandlers) return true;

    if (document.getElementById('cf-turnstile') || document.querySelector('.cf-turnstile')) return true;
    if (document.querySelector('form#challenge-form, [action*=\"__cf_chl_f_tk=\"], input[name=\"cf-turnstile-response\"]')) return true;
    if (document.querySelector('[id*=\"cf-challenge-\"], #challenge-spinner, #cf-challenge-running, .cf-browser-verification')) return true;

    for (let f of document.querySelectorAll('iframe')) {
        if (f.src && f.src.includes('challenges.cloudflare.com')) return true;
    }
    return false;
}
"""

_CLOUDFLARE_RESOLVED_JS = """
() => {
    const t = document.title.toLowerCase();
    if (t.includes('just a moment') || t.includes('attention required') || t.includes('cloudflare')) return false;

    const u = window.location.href;
    if (u.includes('__cf_chl_') || u.includes('cf_chl_')) return false;

    if (window._cf_chl_opt || window._cf_chl_context) return false;

    if (document.getElementById('cf-turnstile')) return false;
    if (document.querySelector('form#challenge-form, [action*=\"__cf_chl_f_tk=\"]')) return false;

    for (let f of document.querySelectorAll('iframe')) {
        if (f.src && f.src.includes('challenges.cloudflare.com')) return false;
    }

    const text = (document.body ? document.body.innerText || '' : '');
    if (text.includes('Checking your browser') || text.includes('cf-challenge')) return false;

    return true;
}
"""


async def _wait_cf_resolution(page, timeout: float = 120):
    """Detect and wait for Cloudflare challenge to resolve.

    Returns True if the page looks real (no challenge), False if
    the challenge is still up after *timeout* seconds.
    """
    try:
        cf = await page.evaluate(_CLOUDFLARE_DETECT_JS)
        if not cf:
            return True
    except Exception:
        return True  # Can't evaluate — assume page is fine

    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        await asyncio.sleep(1)
        try:
            done = await page.evaluate(_CLOUDFLARE_RESOLVED_JS)
            if done:
                return True
        except Exception:
            pass
    return False


def _extract_epub(content: bytes, max_chars: int = 50000) -> str:
    try:
        import ebooklib
        from ebooklib import epub
        import io
        from html.parser import HTMLParser

        class TextExtractor(HTMLParser):
            def __init__(self):
                super().__init__()
                self.result = []
            def handle_data(self, data):
                self.result.append(data)
            def get_text(self):
                return ''.join(self.result)

        book = epub.read_epub(io.BytesIO(content))
        texts = []
        for item in book.get_items():
            if item.get_type() == ebooklib.ITEM_DOCUMENT:
                parser = TextExtractor()
                parser.feed(item.get_content().decode('utf-8', errors='replace'))
                texts.append(parser.get_text())
        return '\n\n'.join(texts)[:max_chars]
    except ImportError:
        return "[EPUB support requires: pip install ebooklib]"
    except Exception as e:
        return f"[EPUB extraction error: {e}]"


def _extract_docx(content: bytes, max_chars: int = 50000) -> str:
    try:
        from docx import Document
        import io
        doc = Document(io.BytesIO(content))
        return '\n\n'.join(p.text for p in doc.paragraphs if p.text.strip())[:max_chars]
    except ImportError:
        return "[DOCX support requires: pip install python-docx]"
    except Exception as e:
        return f"[DOCX extraction error: {e}]"


async def fetch_url(url: str, max_chars: int = 5000, main_content_only: bool = True,
                    target_language: str = "", favor_precision: bool = False,
                    favor_recall: bool = False, fast: bool = False,
                    deduplicate: bool = True, output_format: str = "markdown",
                    include_images: bool = True, include_tables: bool = True,
                    include_comments: bool = True,
                    include_formatting: bool = True, include_links: bool = True,
                    prune_xpath: str = "", url_blacklist: str = "",
                    author_blacklist: str = "", min_output_size: int = 0,
                    raw: bool = False, offset: int = 0, focus: str = "",
                    cache_ttl: int = 3600) -> dict:
    """Fetch a URL with cache, focus filtering, and pagination via httpx + trafilatura.

    Cache keyed by URL+extraction_type (focus and offset are NOT part of the key).
    CDP-fetching (Cloudflare bypass, actions) should use ``scrapling_stealthy_fetch``.
    """
    # SSRF validation — reject internal/private/reserved URLs
    try:
        url = await _validate_url(url)
    except SecurityError as e:
        return {"success": False, "error": str(e), "url": url}
    try:
        url_lower = url.lower()
        extraction_type = output_format  # Backward compat

        # ── Cache check ─────────────────────────────────────────────
        if cache_ttl > 0:
            cached = await cache_mod.get_cached(
                url, extraction_type=extraction_type, ttl=cache_ttl)
            if cached:
                content = cached["content"]
                if focus:
                    content = focus_mod.filter_by_relevance(content, focus)
                return _build_paginated_response(url, content, cached.get("status", 200),
                                                  cached.get("title", ""),
                                                  cached.get("metadata", {}),
                                                  cached.get("content_type", ""),
                                                  offset, max_chars, method="cache")

        # ── Raw mode ────────────────────────────────────────────────
        if raw or any(url_lower.startswith(p) for p in
            ["https://raw.githubusercontent.com/", "https://raw.github.com/",
             "https://gitlab.com/", "https://bitbucket.org/",
             "https://gist.githubusercontent.com/"]):

            try:
                resp = await AsyncFetcher.get(url, timeout=15, stealthy_headers=True)
                content = resp.body if isinstance(resp.body, str) else resp.body.decode("utf-8", errors="replace")
                full_content = content.strip()
                if focus:
                    full_content = focus_mod.filter_by_relevance(full_content, focus)
                return _build_paginated_response(url, full_content, 200,
                                                  url.split("/")[-1], {}, "",
                                                  offset, max_chars, method="raw")
            except Exception as e:
                return {"success": False, "url": url, "error": f"Raw fetch failed: {e}"}

        # ── PDF / EPUB / DOCX ───────────────────────────────────────
        if url_lower.endswith('.pdf'):
            try:
                import os, shutil, tempfile, opendataloader_pdf  # noqa: F811
                from pathlib import Path
                resp = await AsyncFetcher.get(url, timeout=60, stealthy_headers=True)
                body = resp.body if isinstance(resp.body, bytes) else resp.body.encode()
                tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".pdf")
                tmp.write(body)
                tmp.close()
                out_dir = tempfile.mkdtemp()
                try:
                    await asyncio.to_thread(
                        opendataloader_pdf.convert,
                        input_path=[tmp.name], output_dir=out_dir,
                        format="markdown", quiet=True)
                    stem = Path(tmp.name).stem
                    md = next(Path(out_dir).glob(f"{stem}/output.md"), None)
                finally:
                    os.unlink(tmp.name)
                    shutil.rmtree(out_dir, ignore_errors=True)
                md_text = md.read_text("utf-8", errors="replace") if md else "[empty PDF]"
                full_content = md_text
                if focus:
                    full_content = focus_mod.filter_by_relevance(full_content, focus)
                return _build_paginated_response(url, full_content, 200,
                                                  url.split("/")[-1], {}, "",
                                                  offset, max_chars, method="pdf")
            except Exception:
                pass
        if url_lower.endswith('.epub'):
            resp = await AsyncFetcher.get(url, timeout=20, stealthy_headers=True)
            content = _extract_epub(
                resp.body if isinstance(resp.body, bytes) else resp.body.encode(), 100000)
            full_content = content
            if focus:
                full_content = focus_mod.filter_by_relevance(full_content, focus)
            return _build_paginated_response(url, full_content, 200,
                                              url.split("/")[-1], {}, "",
                                              offset, max_chars, method="epub")
        if url_lower.endswith(('.docx', '.doc')):
            resp = await AsyncFetcher.get(url, timeout=20, stealthy_headers=True)
            content = _extract_docx(
                resp.body if isinstance(resp.body, bytes) else resp.body.encode(), 100000)
            full_content = content
            if focus:
                full_content = focus_mod.filter_by_relevance(full_content, focus)
            return _build_paginated_response(url, full_content, 200,
                                              url.split("/")[-1], {}, "",
                                              offset, max_chars, method="docx")

        # ── httpx + trafilatura (primary path) ──────────────────────
        resp = await AsyncFetcher.get(url, timeout=15, stealthy_headers=True)
        raw_html = resp.body if isinstance(resp.body, str) else resp.body.decode("utf-8", errors="replace")
        if not main_content_only:
            content = (resp.get_all_text() or "").strip()
            full_content = content
            if focus:
                full_content = focus_mod.filter_by_relevance(full_content, focus)
            result = _build_paginated_response(url, full_content, resp.status,
                                              "", {}, "",
                                              offset, max_chars, method="httpx")
            result["raw_size"] = len(raw_html)
            return result

        kw: dict[str, Any] = {
            "output_format": output_format, "with_metadata": True,
            "include_links": include_links, "include_tables": include_tables,
            "include_images": include_images, "include_comments": include_comments,
            "include_formatting": include_formatting, "deduplicate": deduplicate,
            "url": url,
        }
        if target_language:
            kw["target_language"] = target_language
        if favor_precision:
            kw["favor_precision"] = True
        if favor_recall:
            kw["favor_recall"] = True
        if fast:
            kw["fast"] = True
        if prune_xpath:
            kw["prune_xpath"] = [x.strip() for x in prune_xpath.split(",") if x.strip()]
        if url_blacklist:
            kw["url_blacklist"] = set(x.strip() for x in url_blacklist.split(",") if x.strip())
        if author_blacklist:
            kw["author_blacklist"] = set(x.strip() for x in author_blacklist.split(",") if x.strip())
        result = trafilatura.extract(raw_html, **kw)
        title = None
        if isinstance(result, str) and result.startswith('{'):
            d = json.loads(result)
            full_content = d.get('text', '')
            title = d.get('title')
        else:
            full_content = result or ''
        if not full_content:
            full_content = trafilatura.extract(raw_html, output_format='txt',
                                                with_metadata=False, url=url) or ''
        if not full_content.strip():
            try:
                full_content = (resp.get_all_text() or '').strip()
            except Exception:
                full_content = raw_html.strip()
        full_content = full_content.strip()

        # Cloudflare challenge detection
        cf_keywords = ["just a moment", "checking your browser", "cf-challenge",
                       "cloudflare ray id", "__cf_chl_", "cf-turnstile",
                       "verify you are human", "attention required"]
        if full_content and sum(1 for kw in cf_keywords if kw in full_content[:600].lower()) >= 2:
            return {"success": False, "url": url,
                    "error": "Cloudflare challenge detected — auto-fallback to CDP in progress."}

        # Generic bot challenge detection (CreepJS, BotD, Anubis, PerimeterX, etc.)
        # These serve JS-heavy pages with minimal readable text — trafilatura extracts
        # very little despite a large raw HTML payload.
        raw_len = len(raw_html)
        if raw_len > 5000 and len(full_content) < 500:
            return {"success": False, "url": url,
                    "error": f"Bot challenge detected ({raw_len} bytes HTML, {len(full_content)} chars text)"}

        if min_output_size and len(full_content) < min_output_size:
            return {"success": False, "url": url,
                    "error": f"Content too short ({len(full_content)} < {min_output_size} chars)"}

        meta_str = trafilatura.extract(raw_html, output_format='json', with_metadata=True,
                                       include_links=False, include_tables=False,
                                       url=url) if full_content else None
        meta = {}
        if isinstance(meta_str, str) and meta_str.startswith('{'):
            try:
                meta = json.loads(meta_str).get("metadata", {})
            except Exception:
                pass

        # Cache the full content
        if cache_ttl > 0:
            asyncio.ensure_future(cache_mod.set_cached(
                url, full_content, extraction_type=extraction_type,
                status=resp.status, title=title or "",
                metadata={k: v for k, v in meta.items() if v}, ttl=cache_ttl))

        if focus:
            full_content = focus_mod.filter_by_relevance(full_content, focus)
        result = _build_paginated_response(url, full_content, resp.status,
                                          title or meta.get("title", ""),
                                          {k: v for k, v in meta.items() if v}, "",
                                          offset, max_chars, method="httpx")
        result["raw_size"] = raw_len
        return result
    except Exception as e:
        return {"success": False, "url": url, "error": str(e)}


async def _cdp_extract_content(page, css_selector: str | None, extraction_type: str,
                               page_url: str = "",
                               include_links: bool = True,
                               include_images: bool = True,
                               include_tables: bool = True,
                               deduplicate: bool = True) -> str:
    if css_selector:
        text = await page.evaluate(f"""
            (() => {{
                const el = document.querySelector({json.dumps(css_selector)});
                return el ? el.innerText : null;
            }})()
        """)
        if text:
            return text.strip()
        return await page.inner_text()
    if extraction_type == "html":
        return await page.document_html()
    if extraction_type == "markdown":
        html_c = await page.document_html()
        content = trafilatura.extract(html_c, output_format='markdown', fast=True,
                                      include_links=include_links, include_images=include_images,
                                      include_tables=include_tables, deduplicate=deduplicate,
                                      url=page_url or None)
        return content or await page.inner_text()
    return await page.inner_text()


async def _cdp_fetch_page(
    url: str,
    *,
    block_resources: bool = True,
    network_idle: bool = True,
    init_script: str = "",
    blocked_domains: list | None = None,
    wait_selector: str = "",
    css_selector: str | None = None,
    extraction_type: str = "markdown",
    retries: int = 3,
    timeout: int = 15,
    include_links: bool = True,
    include_images: bool = True,
    include_tables: bool = True,
    deduplicate: bool = True,
) -> dict:
    """Navigate to *url* via CDP, run actions or extract content, then close.

    Returns ``{"success": True, "url": ..., "title": ..., "content": ...}``
    on success, or ``{"success": False, "url": ..., "error": ...}``
    after all retries are exhausted.
    """
    last_err = None
    for attempt in range(max(retries, 1)):
        page = None
        try:
            page = await _get_optimized_page(block_resources=block_resources)
            if blocked_domains:
                try:
                    await page.set_blocked_resources(list(blocked_domains))
                except Exception:
                    pass
            if init_script:
                try:
                    await page.add_init_script(init_script)
                except Exception:
                    pass
            await page.goto(url, wait_until="commit", timeout=timeout)
            await page.wait_for_load_state("domcontentloaded", timeout=timeout)
            if network_idle:
                try:
                    await page.wait_for_load_state("networkidle", timeout=timeout)
                except Exception:
                    pass
            if wait_selector:
                try:
                    await _cdp_wait_for_selector(page, wait_selector, timeout=10)
                except Exception:
                    pass
            try:
                await _wait_cf_resolution(page)
            except Exception:
                pass

            content = await _cdp_extract_content(
                page, css_selector, extraction_type, page_url=url,
                include_links=include_links, include_images=include_images,
                include_tables=include_tables, deduplicate=deduplicate)

            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            title = await page.title()

            return {"success": True, "url": url,
                    "title": title or "", "content": (content or "")}
        except Exception as e:
            last_err = e
            if page:
                try:
                    await page.close()
                except Exception:
                    pass
            if attempt < retries - 1:
                await asyncio.sleep(min(0.5 * (attempt + 1), 2.0))
        finally:
            if page:
                try:
                    await page.close()
                except Exception:
                    pass
            asyncio.ensure_future(_cleanup_orphan_tabs())

    return {"success": False, "url": url,
            "error": str(last_err) if last_err else "CDP fetch failed"}


async def scrapling_stealthy_fetch(
    url: str, css_selector: str | None = None, extraction_type: str = "markdown",
    headless: bool = True, cdp_url: str | None = None, block_webrtc: bool = False,
    hide_canvas: bool = True, disable_resources: bool = True, google_search: bool = True,
    real_chrome: bool = False, proxy: str = "", locale: str = "", timezone_id: str = "",
    network_idle: bool = False, allow_webgl: bool = True, block_ads: bool = True,
    dns_over_https: bool = True, solve_cloudflare: bool = True, retries: int = 5,
    timeout: int = 30000, capture_xhr: str = "", wait_selector: str = "",
    wait_selector_state: str = "attached", blocked_domains: list | None = None,
    init_script: str = "", extra_headers: dict | None = None,
    useragent: str = "", load_dom: bool = False,
    page_setup=None) -> dict:
    # SSRF validation
    try:
        url = await _validate_url(url)
    except SecurityError as e:
        return {"success": False, "error": str(e), "url": url}
    # ── CDP-first path (primary) ──────────────────────────────────
    if cdp_url:
        result = await _cdp_fetch_page(
            url=url,
            block_resources=disable_resources,
            network_idle=network_idle,
            init_script=init_script,
            blocked_domains=blocked_domains,
            wait_selector=wait_selector,
            css_selector=css_selector,
            extraction_type=extraction_type,
            retries=retries,
            timeout=min(timeout, 15000) / 1000,
        )
        if result.get("success"):
            return {
                "success": True, "url": url,
                "title": result.get("title", ""),
                "content": (result.get("content", "") or "")[:50000],
                "method": "cdp", "attempt": 1,
            }
        # CDP failed — fall through to scrapling

    # ── Scrapling AsyncStealthySession (last resort fallback) ──────
    try:
        sk: dict[str, Any] = {
            "headless": headless, "timeout": timeout, "block_ads": block_ads,
            "dns_over_https": dns_over_https, "solve_cloudflare": solve_cloudflare,
            "retries": retries,
        }
        for k, v in [("block_webrtc", block_webrtc), ("hide_canvas", hide_canvas),
                     ("disable_resources", disable_resources), ("google_search", google_search),
                     ("real_chrome", real_chrome)]:
            if v:
                sk[k] = True
        if not allow_webgl:
            sk["allow_webgl"] = False
        if proxy:
            sk["proxy"] = proxy
        if locale:
            sk["locale"] = locale
        if timezone_id:
            sk["timezone_id"] = timezone_id
        if capture_xhr:
            sk["capture_xhr"] = capture_xhr
        if blocked_domains:
            sk["blocked_domains"] = set(blocked_domains)
        if init_script:
            sk["init_script"] = init_script
        if extra_headers:
            sk["extra_headers"] = extra_headers
        if useragent:
            sk["useragent"] = useragent
        async with AsyncStealthySession(**sk) as session:
            fk: dict[str, Any] = {"url": url, "network_idle": network_idle, "load_dom": load_dom}
            if wait_selector:
                fk["wait_selector"] = wait_selector
                fk["wait_selector_state"] = wait_selector_state
            if page_setup:
                fk["page_setup"] = page_setup
            p = await session.fetch(**fk)
            captured = getattr(p, "captured_xhr", None)
            if css_selector:
                el = p.css(css_selector)
                content = "\n".join(str(e.get_all_text()) for e in el) if el else (
                    p.get_all_text() if extraction_type != "html" else (
                        p.body if isinstance(p.body, str) else p.body.decode("utf-8", errors="replace")))
            else:
                if extraction_type == "html":
                    content = p.body if isinstance(p.body, str) else p.body.decode("utf-8", errors="replace")
                elif extraction_type == "markdown":
                    content = p.get_all_text()
                    html_c = p.body if isinstance(p.body, str) else p.body.decode("utf-8", errors="replace")
                    content = trafilatura.extract(html_c, output_format='markdown', fast=True, url=url) or content
                else:
                    content = p.get_all_text()
            if isinstance(content, bytes):
                content = content.decode("utf-8", errors="replace")
            result: dict[str, Any] = {"success": True, "url": url, "status": p.status,
                                      "content": content[:50000], "method": "scrapling"}
            if captured:
                result["captured_xhr"] = captured
            return result
    except Exception as e:
        return {"success": False, "url": url, "error": str(e)}


async def _cdp_wait_for_selector(page, css: str, timeout: float = 10):
    """Poll for a CSS selector to appear in the DOM."""
    import time as _time
    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        exists = await page.evaluate(f"!!document.querySelector({json.dumps(css)})")
        if exists:
            return True
        await asyncio.sleep(0.1)
    return False


# ── Pagination helper ──────────────────────────────────────────────

def _build_paginated_response(url: str, content: str, status: int | str,
                              title: str, metadata: dict, content_type: str,
                              offset: int, max_chars: int,
                              method: str = "httpx") -> dict:
    """Slice *content* at *offset* and return pagination metadata.

    The ``offset`` param is a 0-based char offset into the full content.
    Returns up to ``max_chars`` chars. Sets ``is_truncated`` and
    ``next_offset`` so the agent can page through with another call.
    """
    total = len(content)

    if offset > 0:
        content = content[offset:]
        if not content:
            return {
                "success": True, "url": url, "status": status,
                "title": title, "content": "",
                "content_type": content_type, "metadata": metadata,
                "total_extracted_chars": total,
                "offset": offset,
                "is_truncated": False,
                "next_offset": 0,
                "method": method,
            }

    if total > max_chars:
        sliced = content[:max_chars]
        next_off = offset + max_chars
        is_truncated = next_off < total
        return {
            "success": True, "url": url, "status": status,
            "title": title, "content": sliced,
            "content_type": content_type, "metadata": metadata,
            "total_extracted_chars": total,
            "offset": offset,
            "is_truncated": is_truncated,
            "next_offset": next_off if is_truncated else 0,
            "method": method,
        }
    else:
        return {
            "success": True, "url": url, "status": status,
            "title": title, "content": content,
            "content_type": content_type, "metadata": metadata,
            "total_extracted_chars": total,
            "offset": offset,
            "is_truncated": False,
            "next_offset": 0,
            "method": method,
        }
