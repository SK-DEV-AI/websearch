from __future__ import annotations

import asyncio
import json
import logging
import os
import random
import datetime
import re
import time
import difflib
from typing import Any

import trafilatura

from scrapling_shim import AsyncFetcher, AsyncStealthySession

from urllib.parse import urlparse

from bot_detection import detect_antibot
from ghost_state import CHALLENGE, CONTENT_OK, TERMINAL, classify, ghost
import jsdata
import feed_extract
import hn_extract
import mathml
import md_convert
from search_gai import _get_optimized_page, _cleanup_orphan_tabs
from cookies import CookieJar
from revalidate import RevalidationCache
from security import SecurityError, safe_fetch_capped, validate_url as _validate_url
import cache as cache_mod
import focus as focus_mod
from config import QUALITY_FLOOR
import config
import fast_paths
from fast_paths import fast_path_fetch

logger = logging.getLogger("fetch")

def _configure_scrapling() -> None:
    """Apply scrapling's parser settings once, on first use.

    The call is deferred because scrapling is an optional dependency: doing it
    at import time would make the whole server unstartable without the extras.
    """
    try:
        AsyncFetcher.configure(huge_tree=True)
    except ImportError:
        pass  # optional extra not installed; the scrapling lanes degrade


_configure_scrapling()


async def _revalidate_redirect(requested: str, resp) -> str | None:
    """E3a: the SSRF validator guards the requested URL, but the
    fetcher follows redirects internally — re-validate the final hop
    so a redirect into an internal/private range is refused.
    Returns an error message, or None when the final URL is fine."""
    final = getattr(resp, "url", None)
    if not final or final == requested:
        return None
    try:
        await _validate_url(final)
    except SecurityError as e:
        return f"Redirect to blocked URL ({final}): {e}"
    return None

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
MAX_DECOMPRESSED = 64 * 1024 * 1024  # donsetch decompress.rs bomb cap

# donsetch inline.rs:212 — tracker params dropped from markdown link targets
TRACKER_PARAMS = re.compile(
    r"[?&](?:utm_[a-z0-9_]*|fbclid|gclid|dclid|msclkid|mc_cid|mc_eid|igshid|ref_src|spm|scm|_ga)[^&]*",
    re.IGNORECASE,
)
_LINK_TARGET = re.compile(r"\]\(([^()\s]+)\)")

# donsetch inline.rs:156 — wiki citation markers: [1], [12], [a] (≤3 digits or 1 lowercase letter)
# Negative lookahead keeps real links: [1](https://…) stays, bare [1] goes.
_CITATION_MARKER = re.compile(r"\[(?:\d{1,3}|[a-z])\](?!\()")

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


def _archive_extract(content: str, original_url: str) -> tuple[str, str]:
    """Shared tail for archived-page fallbacks: trafilatura → markdown.
    Returns (content, title); either may be empty."""
    html_ext = trafilatura.extract(content, output_format="html", with_metadata=False,
                                    include_links=False, include_tables=False,
                                    url=original_url)
    final_content = md_convert.to_markdown(
        html_ext, include_links=False, include_tables=False) if html_ext else ""
    if not final_content:
        final_content = trafilatura.extract(content, output_format="txt",
                                              with_metadata=False, url=original_url) or ""
    m = re.search(r"<title[^>]*>(.*?)</title>", content, re.S | re.I)
    title = m.group(1).strip()[:200] if m else ""
    return final_content.strip(), title


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
        meta: dict[str, Any] = {}
        body_bytes, _trunc = await safe_fetch_capped(c, raw_url, timeout=15,
                                                      meta_out=meta)
        if meta.get("status") != 200:
            return None
        content = body_bytes.decode("utf-8", errors="replace")
        if _bomb_capped(content):
            return None
        if not content or len(content.strip()) < 50:
            return None
        # Run through trafilatura like the normal path — model gets clean text, not raw HTML
        # steal C1: trafilatura extracts, html-to-markdown serializes
        final_content, title = _archive_extract(content, original_url)
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


async def _try_archive_today(original_url: str) -> dict | None:
    """Second dead-page source: newest archive.today snapshot.
    /newest/<url> 302s to the latest capture; no-capture lands on a
    near-empty search page whose extraction stays under our length floor."""
    try:
        from config import get_http_client
        c = get_http_client()
        meta: dict[str, Any] = {}
        body_bytes, _trunc = await safe_fetch_capped(
            c, f"https://archive.ph/newest/{original_url}", timeout=15,
            meta_out=meta)
        # archive.today serves full snapshot bodies even under its
        # aggressive 429 rate-limiting — only hard-fail on real errors
        if meta.get("status") not in (200, 429):
            return None
        content = body_bytes.decode("utf-8", errors="replace")
        if _bomb_capped(content):
            return None
        final_content, _title = _archive_extract(content, original_url)
        if not final_content or len(final_content) < 50:
            return None
        m = re.search(r"<title[^>]*>(.*?)</title>", content, re.S | re.I)
        title = m.group(1).strip()[:200] if m else ""
        # capture timestamp rides in the final redirect URL: .../YYYYMMDDHHMMSS/...
        ts_m = re.search(r"/((?:19|20)\d{12})/", str(meta.get("url", "")))
        date_str = (f"{ts_m.group(1)[:4]}-{ts_m.group(1)[4:6]}-{ts_m.group(1)[6:8]}"
                    if ts_m else "snapshot")
        return {
            "success": True,
            "content": final_content,
            "url": original_url,
            "status": 200,
            "cached_from": f"archive.today ({date_str})",
            "snapshot_url": str(meta.get("url", "")),
            "title": title,
            "method": "archive_today",
        }
    except Exception:
        return None


async def _pdf_fetch_response(url: str, offset: int, max_chars: int,
                              focus: str = "", data: bytes | None = None) -> dict:
    """Shape a PDF fetch the same way for both the .pdf branch and the
    content-type sniff (servers that serve PDFs at extension-less URLs).

    `data` is the body already downloaded by the caller: used by the sniff path
    so a mismatched URL does not cost a second download.
    """
    import tempfile
    from pdf_extract import extract_pdf

    tmp_path = ""
    target = url
    if data:
        try:
            fd, tmp_path = tempfile.mkstemp(prefix="pdfsniff_", suffix=".pdf")
            with os.fdopen(fd, "wb") as fh:
                fh.write(data)
            target = tmp_path
        except Exception as e:
            return {"success": False, "url": url,
                    "error": f"PDF extraction failed: tempfile: {e}"}
    try:
        r = await extract_pdf(target, format="markdown")
    except Exception as e:
        r = {"success": False, "error": str(e)}
    finally:
        if tmp_path:
            try:
                os.remove(tmp_path)
            except OSError:
                pass
    if not r.get("success"):
        # A failed PDF used to fall out of this branch into the HTML pipeline,
        # which happily parsed the raw PDF bytes as text and returned garbage
        # as a successful page. Fail honestly instead.
        err = str(r.get("error") or "unknown error")
        if tmp_path:
            # Never leak the scratch filename at the caller; it means nothing to
            # them and hides the real URL.
            err = err.replace(tmp_path, url).replace(os.path.basename(tmp_path), url)
        if not err.startswith("PDF extraction failed"):
            err = f"PDF extraction failed: {err}"
        out = {"success": False, "url": url, "method": "pdf", "error": err}
        if r.get("ocr_skipped"):
            out.update(ocr_skipped=r["ocr_skipped"], free_mb=r.get("free_mb"),
                       need_mb=r.get("need_mb"))
        return out
    full_content = "\n\n---\n\n".join(v["content"] for v in r["results"].values())
    if focus:
        full_content = focus_mod.filter_by_relevance(full_content, focus)
    out = _build_paginated_response(url, full_content, 200,
                                    url.split("/")[-1], {}, "",
                                    offset, max_chars, method="pdf")
    # A source-level cut (the per-document char cap) is NOT pagination: the
    # agent must be able to tell "the document ends here" from "we stopped
    # reading here".
    for v in r["results"].values():
        if v.get("truncated"):
            out["source_truncated"] = True
            out["source_truncated_at_chars"] = v.get("truncated_at_chars", 0)
            break
    return out


def _looks_like_pdf(raw: str, resp) -> bool:
    """True when the response is really a PDF despite its URL."""
    try:
        headers = getattr(resp, "headers", None) or {}
        items = headers.items() if hasattr(headers, "items") else []
        for k, v in items:
            if str(k).lower() == "content-type" and "application/pdf" in str(v).lower():
                return True
    except Exception:
        pass
    return raw[:5] == "%PDF-"


# ── Page-fetch concurrency ceiling ─────────────────────────────────
# `ops.py` admits up to 8 whole tool calls, but each `research.enrich` fans out
# to `depth*2` pages: at depth 5 that is 8 x 8 = 64 simultaneous downloads, each
# allowed a 64 MiB decompressed body. That is the configuration that made the
# machine crawl under concurrent searches (page cache + socket pressure + RSS
# spikes), even though the event loop itself measured healthy. Slots cap the
# in-flight download set without serialising whole calls; `0` disables the cap.
_PAGE_FETCH_SLOTS = max(0, int(os.environ.get("WEBSEARCH_PAGE_FETCH_CONCURRENCY", "16") or 0))
_PAGE_FETCH_SEM: asyncio.Semaphore | None = None


def _page_sem() -> asyncio.Semaphore | None:
    global _PAGE_FETCH_SEM
    if _PAGE_FETCH_SLOTS <= 0:
        return None
    if _PAGE_FETCH_SEM is None:
        _PAGE_FETCH_SEM = asyncio.Semaphore(_PAGE_FETCH_SLOTS)
    return _PAGE_FETCH_SEM


async def fetch_url(*args, **kwargs):
    """Public fetch entry: bounded by the page-fetch slot pool.

    Thin wrapper so every caller (search enrichment, extract, webtools,
    site_mapper) shares one global ceiling — a per-caller limit would let 8
    searches each open their own 8-wide fan-out and land right back at 64.
    """
    sem = _page_sem()
    if sem is None:
        return await _fetch_url(*args, **kwargs)
    async with sem:
        return await _fetch_url(*args, **kwargs)


async def _fetch_url(url: str, max_chars: int = 5000, main_content_only: bool = True,
                    target_language: str = "", favor_precision: bool = False,
                    favor_recall: bool = False, fast: bool = False,
                    deduplicate: bool = True, output_format: str = "markdown",
                    include_images: bool = True, include_tables: bool = True,
                    include_comments: bool = True,
                    include_formatting: bool = True, include_links: bool = True,
                    prune_xpath: str = "", url_blacklist: str = "",
                    author_blacklist: str = "", min_output_size: int = 0,
                    raw: bool = False, offset: int = 0, focus: str = "",
                    cache_ttl: int = 3600, cookies: dict | None = None,
                    quality_floor: float = 0.0, mineru: bool = False,
                    cache_bust: bool = False) -> dict:
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
        if cache_ttl > 0 and not cache_bust:
            cached = await cache_mod.get_cached(
                url, extraction_type=extraction_type, ttl=cache_ttl)
            if cached:
                content = cached["content"]
                if focus:
                    content = focus_mod.filter_by_relevance(content, focus)
                result = _build_paginated_response(
                    url, content, cached.get("status", 200),
                    cached.get("title", ""), cached.get("metadata", {}),
                    cached.get("content_type", ""),
                    offset, max_chars, method="cache")
                result["cache"] = _cache_state(cached, cache_ttl)
                return result

        # ── Reddit pre-gate probe ───────────────────────────────────
        # Threads resolve inside the user's own Helium session
        # (same-origin .json with their cookies) — user browsing, not
        # crawling. Runs before the robots gate: reddit serves
        # `User-agent: * / Disallow: /`, which would otherwise neuter
        # this probe for the one site that needs it most.
        if config.FAST_PATHS_ENABLED and fast_paths._REDDIT_RE.search(url):
            try:
                fp = await fast_paths._reddit_json(url)
            except Exception:
                fp = None
            if fp:
                if cache_ttl > 0:
                    asyncio.ensure_future(cache_mod.set_cached(
                        url, fp["content"], extraction_type=extraction_type,
                        status=200, content_type=fp.get("content_type", ""),
                        title=fp.get("title", ""), metadata=fp.get("metadata") or {},
                        ttl=cache_ttl))
                return _build_paginated_response(
                    url, fp["content"], 200, fp.get("title", ""),
                    fp.get("metadata") or {}, fp.get("content_type", ""),
                    offset, max_chars, method=fp.get("method", "fast-path"))

        # ── Robots.txt politeness gate ──────────────────────────────
        # Runs before every unattended fetch path so a Disallowed URL
        # is refused regardless of which fetcher would have handled it
        # (reddit-json user-session probe above excepted — see above).
        if config.ROBOTS_POLITENESS:
            import robots as robots_mod
            if not await robots_mod.allowed(url):
                return {"success": False, "url": url, "verdict": "blocked",
                        "error": "blocked by robots.txt"}

        # ── Fast paths (llms.txt / YouTube / Reddit / GitHub) ──────
        # Shape-gated probes that skip the generic pipeline entirely;
        # responses are cached so repeat fetches don't re-probe (GitHub
        # API is 60 req/hr unauthenticated).
        if config.FAST_PATHS_ENABLED:
            fp = await fast_path_fetch(url)
            if fp:
                if cache_ttl > 0:
                    asyncio.ensure_future(cache_mod.set_cached(
                        url, fp["content"], extraction_type=extraction_type,
                        status=200, content_type=fp.get("content_type", ""),
                        title=fp.get("title", ""), metadata=fp.get("metadata") or {},
                        ttl=cache_ttl))
                return _build_paginated_response(
                    url, fp["content"], 200, fp.get("title", ""),
                    fp.get("metadata") or {}, fp.get("content_type", ""),
                    offset, max_chars, method=fp.get("method", "fast-path"))

        # ── Raw mode ────────────────────────────────────────────────
        if raw or any(url_lower.startswith(p) for p in
            ["https://raw.githubusercontent.com/", "https://raw.github.com/",
             "https://gitlab.com/", "https://bitbucket.org/",
             "https://gist.githubusercontent.com/"]):

            try:
                resp = await AsyncFetcher.get(url, timeout=15, stealthy_headers=True)
                redir_err = await _revalidate_redirect(url, resp)
                if redir_err:
                    return {"success": False, "url": url, "error": redir_err}
                if _bomb_capped(resp.body):
                    return {"success": False, "url": url, "error": "decompressed body exceeds 64 MiB cap"}
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
            return await _pdf_fetch_response(url, offset, max_chars, focus)
        elif url_lower.endswith('.epub'):
            resp = await AsyncFetcher.get(url, timeout=20, stealthy_headers=True)
            redir_err = await _revalidate_redirect(url, resp)
            if redir_err:
                return {"success": False, "url": url, "error": redir_err}
            if _bomb_capped(resp.body):
                return {"success": False, "url": url, "error": "decompressed body exceeds 64 MiB cap"}
            content = _extract_epub(
                resp.body if isinstance(resp.body, bytes) else resp.body.encode(), 100000)
            full_content = content
            if focus:
                full_content = focus_mod.filter_by_relevance(full_content, focus)
            return _build_paginated_response(url, full_content, 200,
                                              url.split("/")[-1], {}, "",
                                              offset, max_chars, method="epub")
        elif url_lower.endswith(('.docx', '.doc')):
            resp = await AsyncFetcher.get(url, timeout=20, stealthy_headers=True)
            redir_err = await _revalidate_redirect(url, resp)
            if redir_err:
                return {"success": False, "url": url, "error": redir_err}
            if _bomb_capped(resp.body):
                return {"success": False, "url": url, "error": "decompressed body exceeds 64 MiB cap"}
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
                # Rate-limit / transient 5xx: exponential backoff with jitter,
                # honoring Retry-After when present (capped — never sleep for
                # an hour because a server said so). tenacity pattern, hand-rolled.
                for attempt in range(2):
                    if resp.status != 429 and resp.status < 500:
                        break
                    ra = resp.headers.get("retry-after")
                    if ra and ra.isdigit():
                        delay = min(float(ra), 10.0)
                    else:
                        delay = 1.5 * (2 ** attempt) + random.uniform(0, 1.0)
                    await asyncio.sleep(delay)
                    resp = await AsyncFetcher.get(url, timeout=15, stealthy_headers=True,
                                                  cookies=merged or None,
                                                  headers=cond or None)
                redir_err = await _revalidate_redirect(url, resp)
                if redir_err:
                    return {"success": False, "url": url, "error": redir_err}
            except Exception as e:
                wayback = await _try_wayback(url) or await _try_archive_today(url)
                if wayback:
                    return wayback
                return {"success": False, "url": url, "error": f"Fetch failed: {e}"}
            # Per-hop cookie scope: a redirect hop's Set-Cookie belongs to
            # THAT hop's host, not the original — storing all under `host`
            # replays b.com's cookies to a.com (and vice versa).
            for hop in list(getattr(resp, "history", []) or []) + [resp]:
                hop_url = getattr(hop, "url", None)
                hop_host = urlparse(str(hop_url)).hostname if hop_url else None
                _jar.store_from_headers(hop_host or host, hop.headers)
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
                if _bomb_capped(body):
                    return {"success": False, "url": url, "error": "decompressed body exceeds 64 MiB cap"}
                _reval.store(url, resp.status, resp.headers, body)
                raw_html = resp.body if isinstance(resp.body, str) else body.decode("utf-8", errors="replace")
                status_used = resp.status

        # A server can hand back a PDF at an extension-less URL (arxiv.org/pdf/
        # 2201.xxxx). Before this check those bytes went through trafilatura and
        # came back as mojibake that still looked like a successful fetch.
        if status_used < 400 and raw_html and _looks_like_pdf(raw_html, resp):
            body_bytes = None
            if resp is not None and isinstance(getattr(resp, "body", None), (bytes, bytearray)):
                body_bytes = bytes(resp.body)
            elif status_used == 304:
                stored_pdf = _reval.stored(url)
                if stored_pdf:
                    body_bytes = stored_pdf[0]
            if body_bytes is None:
                body_bytes = raw_html.encode("utf-8", errors="replace")
            return await _pdf_fetch_response(url, offset, max_chars, focus, body_bytes)

        # If the page is dead (404, 410, 5xx), try Wayback
        if status_used in (404, 410) or status_used >= 500:
            wayback = await _try_wayback(url) or await _try_archive_today(url)
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

        # ── Dedicated extractors (donsetch feed.rs / hn.rs ports) ──
        # Feeds are raw XML bodies trafilatura would render as soup;
        # HN comment threads are table layouts the generic pipeline
        # mangles into pipe rows. Both run BEFORE the generic pipeline.
        dedicated = None
        try:
            ct = ((getattr(resp, "headers", {}) or {}).get("content-type", "")
                  if resp is not None else "")
            dedicated = feed_extract.try_extract(raw_html, url, ct)
        except Exception as e:
            logger.debug("fetch: feed extract failed for %s: %s", url, e)
        if not dedicated:
            try:
                dedicated = hn_extract.try_extract(raw_html, url, focus or None)
            except Exception as e:
                logger.debug("fetch: HN extract failed for %s: %s", url, e)

        if dedicated:
            full_content = dedicated
            title = None
        else:
            # MathML → LaTeX pre-transform (donsetch math.rs port):
            # trafilatura flattens <math> away, gutting formulas.
            # Rewrite first so they survive as $..$ / $$..$$ text.
            try:
                traf_input = mathml.transform(raw_html)
            except Exception:
                traf_input = raw_html
            if output_format == 'markdown':
                # steal C1: trafilatura extracts, html-to-markdown serializes.
                # with_metadata must be off for html output (trafilatura 2.1.0
                # crashes on list-valued metadata); title comes from the
                # meta pass below.
                full_content = md_convert.extract_and_convert(
                    traf_input, url=url, fast=fast,
                    include_links=include_links, include_images=include_images,
                    include_tables=include_tables, deduplicate=deduplicate,
                    target_language=target_language,
                    favor_precision=favor_precision,
                    favor_recall=favor_recall,
                    prune_xpath=kw.get('prune_xpath'),
                    url_blacklist=kw.get('url_blacklist'),
                    author_blacklist=kw.get('author_blacklist'))
                title = None
                if not full_content:
                    result = trafilatura.extract(traf_input, **kw)
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
            else:
                result = trafilatura.extract(traf_input, **kw)
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
                full_content = trafilatura.extract(traf_input, output_format='txt',
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
        # Dedicated extractor output is authoritative: HN thread pages are
        # dense HTML with genuinely thin text (small threads) — never gate.
        raw_len = len(raw_html)
        if dedicated is None and raw_len > 5000 and len(full_content) < 500:
            # JS shells (GTM/Next.js) look like bot walls (big HTML, thin text) but are not — don't burn CDP on them
            if "gtm.start" in raw_html[:2000] or "__NEXT_DATA__" in raw_html[:2000]:
                pass  # fall through with thin content, don't trigger CDP
            else:
                return {"success": False, "url": url,
                        "error": f"Unknown bot challenge detected ({raw_len} bytes HTML, {len(full_content)} chars text)"}

        if min_output_size and len(full_content) < min_output_size:
            # Page fetched and parsed fine — genuinely small content.
            # raw_html_len lets server-side escalation tell this apart
            # from a JS shell that could yield more after rendering.
            return {"success": False, "url": url,
                    "error": f"Content too short ({len(full_content)} < {min_output_size} chars)",
                    "content": full_content, "raw_html_len": len(raw_html)}

        meta_str = trafilatura.extract(raw_html, output_format='json', with_metadata=True,
                                       include_links=False, include_tables=False,
                                       url=url) if full_content else None
        meta = {}
        meta_title = ""
        if isinstance(meta_str, str) and meta_str.startswith('{'):
            try:
                d = json.loads(meta_str)
                meta = d.get("metadata", {}) or {}
                meta_title = d.get("title", "") or ""
            except Exception:
                pass

        # extruct secondary pass: microdata/RDFa (trafilatura has neither) +
        # broader JSON-LD/OpenGraph surface, merged under "structured".
        if config.EXTRACT_METADATA_ENABLED and full_content:
            try:
                from extruct.w3cmicrodata import MicrodataExtractor
                from extruct.rdfa import RDFaExtractor
                from extruct.jsonld import JsonLdExtractor
                from extruct.opengraph import OpenGraphExtractor
                _mde, _rde, _jle, _oge = (MicrodataExtractor(), RDFaExtractor(),
                                          JsonLdExtractor(), OpenGraphExtractor())
                structured = {
                    "microdata": _mde.extract(raw_html, base_url=url) or [],
                    "rdfa": _rde.extract(raw_html, base_url=url) or [],
                    "jsonld": _jle.extract(raw_html, base_url=url) or [],
                    "opengraph": _oge.extract(raw_html, base_url=url) or [],
                }
                structured = {k: v for k, v in structured.items() if v}
                if structured:
                    meta["structured"] = structured
                    og = structured.get("opengraph") or []
                    if og and not meta.get("title"):
                        og_props = dict((og[0].get("properties") or []))
                        if og_props.get("og:title"):
                            meta["title"] = og_props["og:title"]
            except Exception as e:
                logger.warning(f"fetch: extruct pass failed: {e}")

        # Resolve the title once, from the best available source.
        _best = _best_title(meta.get("title", ""), meta_title, title,
                            content=full_content)

        if focus:
            full_content = focus_mod.filter_by_relevance(full_content, focus)
        full_content = _strip_trackers_md(full_content)
        full_content = _drop_citation_markers(full_content)
        full_content = _token_polish(full_content)

        # Change tracking: diff against whatever we cached last time
        changed_info = None
        try:
            prev = await cache_mod.get_previous(url, extraction_type=extraction_type)
            if prev and prev.get("content") is not None \
                    and prev["content"] != full_content:
                ratio = difflib.SequenceMatcher(
                    None, prev["content"][:20000], full_content[:20000]).ratio()
                changed_info = {"changed_pct": round((1 - ratio) * 100, 1),
                                "previous_fetch": time.strftime(
                                    "%Y-%m-%dT%H:%M:%SZ",
                                    time.gmtime(prev["fetched_at"]))}
        except Exception as e:
            logger.debug("fetch: change-tracking diff failed for %s: %s", url, e)

        # Cache the full content
        if cache_ttl > 0:
            _cache_task = asyncio.ensure_future(cache_mod.set_cached(
                url, full_content, extraction_type=extraction_type,
                status=status_used, title=_best,
                metadata={k: v for k, v in meta.items() if v}, ttl=cache_ttl))
            def _cache_done(t, _url=url):
                # A cancelled task raises CancelledError from .exception(), which
                # surfaced as a scary traceback on every clean shutdown. Check
                # cancelled() first, and call exception() once, not twice.
                if t.cancelled():
                    return
                exc = t.exception()
                if exc:
                    logger.warning("fetch: cache write failed for %s: %s", _url, exc)

            _cache_task.add_done_callback(_cache_done)

        result = _build_paginated_response(url, full_content, status_used,
                                          _best,
                                          {k: v for k, v in meta.items() if v}, "",
                                          offset, max_chars, method="httpx")
        if changed_info:
            result["changed"] = changed_info
        result["raw_size"] = raw_len
        q = _extraction_quality(full_content, title=_best,
                                url=url, raw_html=(raw_html or "").encode("utf-8", errors="replace") if isinstance(raw_html, str) else (raw_html or b""))
        result["quality"] = q
        result["low_quality"] = q < QUALITY_FLOOR
        # ── C3: MinerU-HTML rescue (auto-escalate) ─────────────────
        # The SLM main-content extractor is heavy (model ~1.2 GB
        # resident, seconds per page on CPU) — but the quality scorer
        # already knows when extraction failed, so escalate automatically
        # when enabled instead of waiting for the model to pass the flag.
        # Explicit mineru=True forces it even at decent quality; the
        # default path fires on low quality alone. MinerU never overrides
        # a decent extraction unless explicitly asked.
        if config.MINERU_ENABLED and (mineru or q < (quality_floor or QUALITY_FLOOR)):
            try:
                from mineru_extract import extract_with_mineru
                mr = await extract_with_mineru(raw_html, output_format=output_format)
                if mr.get("success") and len(mr["content"]) > len(full_content):
                    full_content = mr["content"].strip()
                    result = _build_paginated_response(
                        url, full_content, status_used,
                        _best,
                        {k: v for k, v in meta.items() if v}, "",
                        offset, max_chars, method="mineru_html")
                    result["raw_size"] = raw_len
                    result["quality"] = _extraction_quality(full_content, title=title or "", url=url)
                    result["low_quality"] = result["quality"] < QUALITY_FLOOR
            except Exception as e:
                logger.warning("mineru rescue failed for %s: %s", url, e)
        return _attach_published(result, meta)
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
        content = md_convert.extract_and_convert(
            html_c, url=page_url or "", fast=True,
            include_links=include_links, include_images=include_images,
            include_tables=include_tables, deduplicate=deduplicate)
        if content:
            content = _strip_trackers_md(content)
            content = _drop_citation_markers(content)
            content = _token_polish(content)
        return content or await page.inner_text()
    return await page.inner_text()


# UI furniture that metadata extractors pick up as a page title on some sites.
# mdbook (the Rust books) hands out "Keyboard shortcuts" from a hidden help
# panel, which is how a search result ends up labelled that instead of the
# chapter name - and a caller reading "Title: Keyboard shortcuts" learns nothing.
_JUNK_TITLES = {
    "keyboard shortcuts", "menu", "navigation", "home", "search", "untitled",
    "index", "404", "page not found", "not found", "log in", "sign in",
    "javascript is disabled", "loading", "untitled document", "document",
}


# A title that describes how the page was fetched is worse than no title: the
# caller reads it as the page's subject. These are produced by our own
# fast-path extractors, so they are detectable by shape rather than by site.
_JUNK_TITLE_PATTERNS = (
    re.compile(r"^markdown via\b", re.I),
    re.compile(r"\bvia accept header\b", re.I),
    re.compile(r"^(keyboard shortcuts|untitled|document|loading)\b", re.I),
    re.compile(r"^\W*$"),
)


def _is_junk_title(t: str) -> bool:
    t = (t or "").strip().lower()
    if not t or len(t) < 3 or t in _JUNK_TITLES:
        return True
    return any(p.search(t) for p in _JUNK_TITLE_PATTERNS)


def _best_title(*candidates: str, content: str = "") -> str:
    """First usable title, else the document's own first heading.

    Order matters: a page-specific og:title beats a generic <title>, which beats
    a heading, which beats nothing. Returning "" is correct when nothing is
    usable - an invented label is worse than an empty one.
    """
    for c in candidates:
        c = (c or "").strip()
        # A candidate can arrive as a markdown heading; its markers are not
        # part of the title.
        c = re.sub(r"^#{1,6}\s*", "", c)
        c = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", c).strip()
        if not _is_junk_title(c):
            return c[:200]
    for m in re.finditer(r"^#{1,2}\s+(.+)$", content or "", re.M):
        h = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", m.group(1)).strip()
        if not _is_junk_title(h):
            return h[:200]
    return ""



# Dates that a page states about itself. Search engines return `published` for
# only some results, so before this the freshness of most sources was unknown -
# `as_of` came back empty on a live run even though several pages carried a
# date in their own structured data.
_DATE_KEYS = (
    "datePublished", "dateCreated", "uploadDate",
    "article:published_time", "article:modified_time", "dateModified",
    "published_time", "pubdate",
)
_PUBLISHED_KEYS = {"datepublished", "datecreated", "uploaddate",
                   "article:published_time", "published_time", "pubdate"}


def _norm_date(raw: str) -> str:
    """Return YYYY-MM-DD for a parseable date, else ''.

    Partial dates are rejected rather than guessed: a page that only says "2024"
    must not become 2024-01-01, because a precise-looking date that nobody stated
    is a fabricated fact.
    """
    raw = (raw or "").strip()
    if not raw:
        return ""
    m = re.match(r"^(\d{4})-(\d{2})-(\d{2})", raw)
    if m:
        try:
            dt = datetime.date(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        except ValueError:
            return ""
    else:
        m2 = re.match(r"^(\d{4})/(\d{2})/(\d{2})$", raw)
        if not m2:
            return ""
        try:
            dt = datetime.date(int(m2.group(1)), int(m2.group(2)), int(m2.group(3)))
        except ValueError:
            return ""
    # A date in the future is a parsing artefact (or a scheduled post), and
    # treating it as evidence of freshness would be wrong.
    if dt > datetime.date.today():
        return ""
    if dt.year < 1990:
        return ""
    return dt.isoformat()


def _walk_for_dates(obj, found: dict) -> None:
    """Collect date-ish keys from nested jsonld / microdata / rdfa structures."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            kl = str(k).lower()
            if kl in {d.lower() for d in _DATE_KEYS} and isinstance(v, str):
                found.setdefault(kl, v)
            else:
                _walk_for_dates(v, found)
    elif isinstance(obj, list):
        for it in obj[:20]:
            _walk_for_dates(it, found)


def _published_from_meta(meta: dict) -> tuple[str, str]:
    """(published, modified) as ISO dates, either may be ''.

    Both are returned because either alone misleads. A wiki reports its article
    creation date as datePublished, so showing only that makes a page edited this
    week look twenty years old; showing only "last modified" makes a typo fix
    look like fresh reporting. The caller gets both and the prompt shows both.
    """
    found: dict = {}
    _walk_for_dates((meta or {}).get("structured") or {}, found)
    pub = ""
    mod = ""
    for key, raw in found.items():
        iso = _norm_date(raw)
        if not iso:
            continue
        if key in _PUBLISHED_KEYS and not pub:
            pub = iso
        elif key not in _PUBLISHED_KEYS and not mod:
            mod = iso
    return pub, mod


def _attach_published(result: dict, meta: dict) -> dict:
    pub, mod = _published_from_meta(meta)
    if pub:
        result["published"] = pub
    if mod:
        result["modified"] = mod
    return result



def _extraction_quality(content: str, *, title: str = "", url: str = "",
                        raw_html: bytes = b"") -> float:
    """Lightweight 0-1 extraction-quality score (rs-trafilatura
    extraction_quality concept). Signals: log-scaled text length,
    text/markup density, title presence, structural markup richness.
    Zero new deps — everything is already computed by the callers."""
    if not content or not content.strip():
        return 0.0
    n = len(content.strip())
    if n < 200:
        return 0.05
    len_sig = min(n / 2000.0, 1.0)
    if raw_html:
        raw_len = max(len(raw_html), 1)
        text_bytes = len(content.encode("utf-8", errors="replace"))
        density = min(text_bytes / raw_len * 8.0, 1.0)
    else:
        density = 0.5
    title_sig = 1.0 if title and len(title.strip()) >= 4 else 0.0
    struct = 0.0
    if "```" in content:
        struct += 0.15
    if "| " in content and "\n|-" in content.replace("\n\n", "\n"):
        struct += 0.15
    score = 0.45 * len_sig + 0.25 * density + 0.15 * title_sig + struct
    return round(min(max(score, 0.0), 1.0), 3)


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
                    # capped: chat widgets / long-poll pages never settle,
                    # burning the full budget (cs.rin.ru: +15s for nothing)
                    await page.wait_for_load_state(
                        "networkidle", timeout=min(timeout, 8))
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

            q = _extraction_quality(content or "", title=title or "", url=url)
            return {"success": True, "url": url,
                    "title": title or "", "content": (content or ""),
                    "cookies": cookies, "verdict": verdict,
                    "quality": q, "low_quality": q < QUALITY_FLOOR}
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


def _ghost_host(url: str) -> str:
    """Hostname for ghost-state fingerprint keying (like ghost.host_of)."""
    try:
        return (urlparse(url).hostname or "").lower()
    except ValueError:
        return ""


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
        # Camoufox coherence model: no explicit knobs + non-CDP path → fill
        # locale/timezone/UA from one per-host fingerprint bundle so the
        # stealth knobs never contradict each other.
        if not cdp_url and not (locale or timezone_id or useragent):
            from fingerprint import bundle_fills
            fills = bundle_fills(ghost.fp_seed(_ghost_host(url)),
                                 locale, timezone_id, useragent)
            locale, timezone_id, useragent = (fills["locale"],
                                              fills["timezone_id"],
                                              fills["useragent"])
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
            final = getattr(p, "url", None)
            if final and final != url:
                try:
                    await _validate_url(final)
                except SecurityError:
                    return {"success": False, "url": url,
                            "error": f"Redirect to blocked URL ({final})"}
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
                    content = md_convert.extract_and_convert(html_c, url=url, fast=True) or content
                    content = _strip_trackers_md(content)
                    content = _drop_citation_markers(content)
                    content = _token_polish(content)
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
    except ImportError as e:
        # scrapling itself is missing -> an install problem, not a fetch failure.
        return {"success": False, "url": url, "error": str(e),
                "lane": "scrapling_stealth", "lane_status": "not_installed",
                "fix": "pip install 'skdev-websearch-mcp[scraping]'"}
    except Exception as e:
        msg = str(e)
        # The stealth tier needs its own browser binary (patchright/playwright
        # download). That is an optional-dependency gap, NOT a bug in this
        # server, and reporting it as a raw traceback made agents mark the whole
        # tool as broken. Say what it is and how to fix it.
        low = msg.lower()
        if "executable doesn't exist" in low or "please run the following command" in low:
            return {"success": False, "url": url, "error": msg.splitlines()[0],
                    "lane": "scrapling_stealth", "lane_status": "browser_missing",
                    "note": ("The optional stealth tier needs its own browser build; "
                             "this is a setup gap, not a server error. Every other "
                             "fetch tier still works."),
                    "fix": "patchright install   # or: playwright install chromium"}
        if "executable" in low and "not exist" in low:
            return {"success": False, "url": url, "error": msg.splitlines()[0],
                    "lane": "scrapling_stealth", "lane_status": "browser_missing",
                    "fix": "patchright install"}
        return {"success": False, "url": url, "error": msg,
                "lane": "scrapling_stealth"}


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

def _paginate(text: str, offset: int, max_chars: int) -> tuple[str, int | None]:
    """Char-budget slice at a block boundary (donsetch paginate port).

    Resuming at offset>0 seeks forward to the next ``\n\n`` block so the
    agent starts at a clean paragraph/heading, not mid-sentence. Cutting
    prefers a block boundary in the last quarter of the window, and an
    inline truncation marker carries the resume offset inside the content
    itself (agents read content, not metadata). Returns (slice, next_off).
    """
    total = len(text)
    if offset >= total:
        return "", None
    start = offset
    if offset > 0:
        pos = text.find("\n\n", start, min(start + 500, total))
        if pos != -1:
            start = pos + 2
    end = min(start + max_chars, total)
    if end < total:
        window_start = start + (end - start) * 3 // 4
        pos = text.rfind("\n\n", window_start, end)
        if pos != -1:
            end = pos
    next_off = end if end < total else None
    slice_ = text[start:end]
    if next_off is not None:
        slice_ += f"\n\n*[truncated — continue with offset={next_off}]*"
    return slice_, next_off


def _bomb_capped(body: bytes | str) -> bool:
    """True if a (already decompressed) body exceeds the 64 MiB cap."""
    return len(body) > MAX_DECOMPRESSED


def _strip_trackers_md(md: str) -> str:
    """Drop tracker params from markdown link targets (donsetch inline.rs:212).

    Rewrites ``[text](url?utm_...=x&fbclid=y)`` to ``[text](url)`` — the
    markdown equivalent of stripping at link-render time.
    """
    def _clean(m: re.Match) -> str:
        target = TRACKER_PARAMS.sub("", m.group(1))
        if "?" in m.group(1) and "?" not in target and "&" in target:
            target = target.replace("&", "?", 1)
        return f"]({target})"
    return _LINK_TARGET.sub(_clean, md)


def _drop_citation_markers(md: str) -> str:
    """Drop wiki citation markers [1], [12], [a] (donsetch inline.rs:156).

    Removes bracket-only tokens that are <=3 digits or a single lowercase
    letter — the shape of wiki sup citations. Real links (``[text](url)``)
    and code spans are untouched because the regex only matches bare brackets.
    """
    return _CITATION_MARKER.sub("", md)


def _token_polish(md: str) -> str:
    """Token-war markdown filters (donsetch render.rs port).

    Drops bare-link lines, bare-number lines (vote counts, ranks), wiki
    ``[edit]`` junk, link-farm runs (>=7 consecutive bare links — the
    block is dropped entirely, matching render.rs's list-density drop),
    and exact-duplicate prose paragraphs (badge dupes, repeated teasers).
    """
    lines = md.split("\n")
    out: list[str] = []
    seen: set[str] = set()
    farm = 0
    farm_start = -1
    for ln in lines:
        s = ln.strip()
        if not s:
            out.append(ln)
            continue  # blank lines don't break a farm run
        bare = False
        if s.startswith("!["):
            pass
        elif len(s) < 80 and (
            re.fullmatch(r"\[[^\]]*\]\(https?://[^)]*\)", s)
            or re.fullmatch(r"<https?://[^\s<>]+>", s)
        ):
            bare = True
        elif len(s) < 8 and re.fullmatch(r"[\d,]+", s):
            bare = True
        elif len(s) < 14 and s.startswith("[") and s.endswith("]")\
                and s[1:-1].strip().replace(" ", "").isalpha():
            bare = True
        if bare:
            if farm == 0:
                farm_start = len(out)
            farm += 1
            out.append(ln)
            continue
        if farm >= 7:
            del out[farm_start:]
        farm = 0
        key = re.sub(r"\s+", " ", s).lower()
        if key and s[0] not in "#-*|>`":
            if key in seen:
                continue
            seen.add(key)
        out.append(ln)
    if farm >= 7:
        del out[farm_start:]
    return "\n".join(out)


def _cache_state(cached: dict, cache_ttl: int) -> dict:
    """Describe a cache hit in a way a caller can act on.

    "cached: true" answers the wrong question. What a caller needs is how old
    the content is and how long it stays valid, because those decide whether to
    trust it or re-fetch with cache_bust.
    """
    fetched_at = float(cached.get("fetched_at") or 0)
    age = max(0, int(time.time() - fetched_at)) if fetched_at else 0
    ttl = int(cached.get("ttl") or cache_ttl or 0)
    return {
        "hit": True,
        "age_s": age,
        "ttl_s": ttl,
        "ttl_remaining_s": max(0, min(ttl, int(cache_ttl)) - age),
        "stale_risk": "high" if age > max(60, int(cache_ttl) * 0.75) else "low",
    }


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

    sliced, next_off = _paginate(content, offset, max_chars)
    is_truncated = next_off is not None
    return {
        "success": True, "url": url, "status": status,
        "title": title, "content": sliced,
        # Default: this content was fetched now. A cache hit overrides it.
        "cache": {"hit": False},
        "content_type": content_type, "metadata": metadata,
        "total_extracted_chars": total,
        "offset": offset,
        "is_truncated": is_truncated,
        "next_offset": next_off if is_truncated else 0,
        "method": method,
    }


def _selfcheck() -> None:
    """Date and title extraction, with no network.

    Both functions shape what the synthesis model is told about a source, and
    both have already been wrong once: a fetch label became a page title, and
    publication dates were dropped entirely while sitting in the page's own
    structured data.
    """
    checks: list[tuple[str, bool]] = []

    checks.append(("iso date is kept", _norm_date("2026-10-15T02:20:26+00:00") == "2026-10-15"))
    checks.append(("year-only date is refused, not guessed", _norm_date("2024") == ""))
    checks.append(("future date is refused", _norm_date("2099-01-01") == ""))
    checks.append(("pre-1990 date is refused", _norm_date("1970-01-01") == ""))
    checks.append(("slash date is accepted", _norm_date("2024/03/25") == "2024-03-25"))
    checks.append(("garbage is refused", _norm_date("last tuesday") == ""))
    checks.append(("impossible date is refused", _norm_date("2024-02-31") == ""))

    pub, mod = _published_from_meta({"structured": {"jsonld": [{
        "datePublished": "2024-03-25T13:00:41+00:00", "dateModified": "2026-01-02"}]}})
    checks.append(("both dates extracted", (pub, mod) == ("2024-03-25", "2026-01-02")))
    checks.append(("published preferred as the publication date", pub == "2024-03-25"))

    pub2, mod2 = _published_from_meta({"structured": {"jsonld": [{
        "dateModified": "2026-10-23T21:58:19Z"}]}})
    checks.append(("modified-only page has no published date", pub2 == "" and mod2 == "2026-10-23"))
    checks.append(("empty metadata is safe", _published_from_meta({}) == ("", "")))

    checks.append(("mdbook UI label rejected", _is_junk_title("Keyboard shortcuts")))
    checks.append(("fetch label rejected", _is_junk_title("markdown via Accept header for x")))
    checks.append(("real title kept", not _is_junk_title("Creatine - Wikipedia")))
    checks.append(("heading is the fallback",
                   _best_title("", "Keyboard shortcuts", content="# MIR borrow check\n\nx") == "MIR borrow check"))
    checks.append(("no title beats a fake title", _best_title("", "", content="no heading") == ""))

    for name, ok in checks:
        print(f"{'OK' if ok else 'FAIL'}  {name}")
    print(f"fetch selfcheck: {sum(1 for _, v in checks if v)}/{len(checks)} OK")


if __name__ == "__main__":
    _selfcheck()
