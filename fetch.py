from __future__ import annotations

import asyncio
import json
import logging
import time
from typing import Any

import trafilatura

from scrapling.fetchers import AsyncFetcher, AsyncStealthySession

from urllib.parse import urlparse

from bot_detection import detect_antibot
from ghost_state import CHALLENGE, CONTENT_OK, TERMINAL, classify, ghost
import jsdata
from search_gai import _get_optimized_page, _cleanup_orphan_tabs
from cookies import CookieJar
from revalidate import RevalidationCache
from security import SecurityError, safe_fetch, validate_url as _validate_url
import cache as cache_mod
import focus as focus_mod

logger = logging.getLogger("fetch")

AsyncFetcher.configure(huge_tree=True)

# donsetch ports: per-host cookie jar + conditional revalidation cache
_jar = CookieJar()
_reval = RevalidationCache()

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

# Consent-banner + turnstile auto-dismiss (donsetch ops.rs DISMISS_MODALS_JS,
# turnstile click) — run once after the page settles so modals cannot
# wedge the challenge iframe and so cf-turnstile checkboxes get clicked.
_AUTO_DISMISS_JS = """
() => {
    const selectors = [
        'button[id*=\"accept\"]', 'button[class*=\"accept\"]', 'button[aria-label*=\"Accept\"]',
        'button[class*=\"consent\"]', '#onetrust-accept-btn-handler',
        'button[aria-label*=\"Agree\"]', '.fc-button.fc-cta-consent',
        'button[class*=\"cookie\"]', '#CybotCookiebotDialogBodyButtonAccept',
        '[class*=\"cookie-banner\"] button', '[id*=\"cmpbntyestxt\"]',
        'button[aria-label*=\"Got it\"]',
    ];
    for (let s of selectors) {
        const el = document.querySelector(s);
        if (el) { try { el.click(); } catch (e) {} }
    }
    const ts = document.querySelector('input[type=\"checkbox\"][name*=\"turnstile\"], .cf-turnstile input[type=\"checkbox\"]');
    if (ts) { try { ts.click(); } catch (e) {} }
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


async def _try_wayback(original_url: str) -> dict | None:
    """Check Wayback Machine for an archived copy of original_url.

    Returns fetch-like dict with cached_from set, or None if no snapshot exists.
    """
    try:
        from config import get_http_client
        c = get_http_client()
        r = await c.get(
            "https://archive.org/wayback/available",
            params={"url": original_url},
            timeout=10,
        )
        if r.status_code != 200:
            return None
        data = r.json()
        snap = data.get("archived_snapshots", {}).get("closest", {})
        if not snap.get("available"):
            return None
        ts = snap.get("timestamp", "")
        snap_url = snap.get("url", "")
        if not snap_url:
            return None
        # Strip the Wayback banner by appending id_ modifier to timestamp
        if not snap_url.startswith("https://web.archive.org/"):
            return None
        raw_url = snap_url.replace(f"/web/{ts}/", f"/web/{ts}id_/")
        wr = await safe_fetch(c, raw_url, timeout=15)
        if wr.status_code != 200:
            return None
        content = wr.text
        if not content or len(content.strip()) < 50:
            return None
        # Run through trafilatura like the normal path — model gets clean text, not raw HTML
        extracted = trafilatura.extract(content, output_format="markdown", with_metadata=True,
                                         include_links=False, include_tables=False,
                                         url=original_url)
        title = ""
        final_content = extracted or trafilatura.extract(content, output_format="txt",
                                                          with_metadata=False, url=original_url) or ""
        if isinstance(extracted, str) and extracted.startswith("{"):
            try:
                d = json.loads(extracted)
                final_content = d.get("text", "")
                title = d.get("title", "")
            except Exception:
                pass
        final_content = final_content.strip()
        # Format a human-readable date from the timestamp
        date_str = f"{ts[:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ts
        return {
            "success": True,
            "content": final_content,
            "url": original_url,
            "status": 200,
            "cached_from": f"web.archive.org ({date_str})",
            "snapshot_url": snap_url,
            "snapshot_timestamp": ts,
            "title": title,
            "method": "wayback",
        }
    except Exception:
        return None


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
                    cache_ttl: int = 3600, cookies: dict | None = None) -> dict:
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
            from pdf_extract import extract_pdf
            try:
                r = await extract_pdf(url, format="markdown")
                if r.get("success"):
                    full_content = "\n\n---\n\n".join(
                        v["content"] for v in r["results"].values())
                else:
                    full_content = f"[PDF extraction failed: {r.get('error')}]"
                if focus:
                    full_content = focus_mod.filter_by_relevance(full_content, focus)
                return _build_paginated_response(url, full_content, 200,
                                                  url.split("/")[-1], {}, "",
                                                  offset, max_chars, method="pdf")
            except Exception as e:
                logger.warning("PDF extraction failed for %s: %s", url, e)
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
        # 304 revalidation + cookie jar (donsetch revalidate.rs/cookies.rs
        # port): browser-true freshness windows, conditional GETs, and a
        # per-host jar fed from Set-Cookie. Fresh entries skip the request.
        parsed = urlparse(url)
        host = (parsed.hostname or "").lower()
        reval = _reval.check(url)
        resp = None
        raw_html = None
        if reval and reval[0] == "fresh":
            raw_html = reval[1].decode("utf-8", errors="replace")
            status_used = reval[2]
        else:
            cond = dict(reval[1]) if reval and reval[0] == "revalidate" else None
            jar_cookies = _jar.dict_for(host, parsed.path or "/")
            merged = {**jar_cookies, **(cookies or {})}
            try:
                resp = await AsyncFetcher.get(url, timeout=15, stealthy_headers=True,
                                              cookies=merged or None,
                                              headers=cond or None)
            except Exception as e:
                wayback = await _try_wayback(url)
                if wayback:
                    return wayback
                return {"success": False, "url": url, "error": f"Fetch failed: {e}"}
            for hop in list(getattr(resp, "history", []) or []) + [resp]:
                _jar.store_from_headers(host, hop.headers)
            if resp.status == 304:
                stored = _reval.stored(url)
                if stored:
                    raw_html = stored[0].decode("utf-8", errors="replace")
                    status_used = stored[1]
                else:
                    raw_html = ""
                    status_used = 304
            else:
                body = resp.body if isinstance(resp.body, bytes) else resp.body.encode("utf-8", errors="replace")
                _reval.store(url, resp.status, resp.headers, body)
                raw_html = resp.body if isinstance(resp.body, str) else body.decode("utf-8", errors="replace")
                status_used = resp.status

        # If the page is dead (404, 410, 5xx), try Wayback Machine
        if status_used in (404, 410) or status_used >= 500:
            wayback = await _try_wayback(url)
            if wayback:
                return wayback
            # No snapshot — let content extraction continue for error page info
        # fresh reval entries have no live response — run the full
        # extraction path (works off raw_html + status_used) instead
        if not main_content_only and resp is not None:
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
            try:
                d = json.loads(result)
            except (json.JSONDecodeError, ValueError):
                d = None
            if d:
                full_content = d.get('text', '')
                title = d.get('title')
            else:
                full_content = result
        else:
            full_content = result or ''
        if not full_content:
            full_content = trafilatura.extract(raw_html, output_format='txt',
                                                with_metadata=False, url=url) or ''
        if not full_content.strip():
            if resp is not None:
                try:
                    full_content = (resp.get_all_text() or '').strip()
                except Exception:
                    pass
            if not full_content.strip():
                full_content = raw_html.strip()
        full_content = full_content.strip()

        # Bot challenge detection via vendor-specific patterns (is-antibot port)
        # Covers Cloudflare, Akamai, DataDome, PerimeterX, Anubis, reCAPTCHA,
        # Turnstile, hCaptcha, and 25+ more — all from static HTTP response data.
        headers_dict = getattr(resp, "headers", {}) if resp else {}
        set_cookie = headers_dict.get("set-cookie", headers_dict.get("Set-Cookie"))
        detected, provider, detection_type = detect_antibot(
            html=raw_html,
            url=url,
            status_code=status_used,
            headers=headers_dict,
            set_cookie=set_cookie,
        )
        if detected:
            return {"success": False, "url": url,
                    "error": f"{provider} challenge detected ({detection_type}) — auto-fallback to CDP in progress."}

        # ── SPA JSON rescue (donsetch jsdata.rs port) ───────────────
        # A script-heavy page that yielded thin text is usually a
        # client-rendered shell with the real content embedded in a
        # JSON blob (Next.js __next_f RSC frames, __NEXT_DATA__,
        # GitHub react-app embeddedData, ytInitialData, ld+json ...).
        # Mine it BEFORE the density fallback so shells are rescued
        # instead of misread as unknown bot challenges. Challenge
        # pages are config-noise shaped and get rejected by the
        # scorer, so they still fall through to the gate below.
        if len(full_content) < 800:
            try:
                mined = jsdata.extract(raw_html, url)
            except Exception as e:
                logger.debug("jsdata rescue failed for %s: %s", url, e)
                mined = None
            if mined and len(mined) > len(full_content) + 50:
                full_content = mined

        # Generic density-based fallback for unknown challenge vendors
        # (is-antibot patterns only cover known vendors — new/obscure challenge
        # pages also serve massive JS payloads with near-empty extracted text.)
        raw_len = len(raw_html)
        if raw_len > 5000 and len(full_content) < 500:
            return {"success": False, "url": url,
                    "error": f"Unknown bot challenge detected ({raw_len} bytes HTML, {len(full_content)} chars text)"}

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
            _cache_task = asyncio.ensure_future(cache_mod.set_cached(
                url, full_content, extraction_type=extraction_type,
                status=status_used, title=title or "",
                metadata={k: v for k, v in meta.items() if v}, ttl=cache_ttl))
            _cache_task.add_done_callback(
                lambda t: t.exception() and logger.warning(
                    f"fetch: cache write failed for {url}: {t.exception()}"))

        if focus:
            full_content = focus_mod.filter_by_relevance(full_content, focus)
        result = _build_paginated_response(url, full_content, status_used,
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
                await page.evaluate(_AUTO_DISMISS_JS)
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

            # ── Terminal-verdict gate (donsetch server.rs:798-820) ──
            # Only content that actually looks like the page may be
            # served; a rendered 404/paywall/auth shell is an error,
            # not content — and an unsolved challenge is a failed
            # solve, not a page.
            verdict = classify(0, content, title)
            if verdict == CHALLENGE:
                return {"success": False, "url": url,
                        "error": "Challenge wall still up after browser render",
                        "verdict": verdict,
                        "next_action": "tier=2 (manual browser)"}
            if verdict in TERMINAL:
                return {"success": False, "url": url,
                        "error": f"Rendered page is a {verdict} shell",
                        "verdict": verdict, "next_action": "none"}

            # ── Clearance harvest (solve-and-bounce handoff) ─────────
            cookies: list[dict] = []
            try:
                cookies = await page.cookies()
            except Exception:
                pass

            return {"success": True, "url": url,
                    "title": title or "", "content": (content or ""),
                    "cookies": cookies, "verdict": verdict}
        except Exception as e:
            last_err = e
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
                "cookies": result.get("cookies", []),
                "verdict": result.get("verdict", CONTENT_OK),
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
            # terminal-verdict gate for the scrapling path too (same
            # shell-laundering protection as the CDP path)
            verdict = classify(result.get("status", 0), result.get("content", ""), "")
            if verdict == CHALLENGE or verdict in TERMINAL:
                return {"success": False, "url": url,
                        "error": f"Rendered page is a {verdict} shell",
                        "verdict": verdict, "next_action": "none"}
            result["verdict"] = verdict
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
