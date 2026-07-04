from __future__ import annotations

import asyncio
import json
from typing import Any

import trafilatura

from scrapling.fetchers import AsyncFetcher, AsyncStealthySession

from config import cached
from search_gai import _get_optimized_page, _cleanup_orphan_tabs

AsyncFetcher.configure(huge_tree=True)


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


@cached(ttl=120)
async def fetch_url(url: str, max_chars: int = 5000, main_content_only: bool = True,
                    target_language: str = "", favor_precision: bool = False,
                    favor_recall: bool = False, fast: bool = False,
                    deduplicate: bool = True, output_format: str = "markdown",
                    include_images: bool = True, include_tables: bool = True,
                    include_comments: bool = True,
                    include_formatting: bool = True, include_links: bool = True,
                    prune_xpath: str = "", url_blacklist: str = "",
                    author_blacklist: str = "", cdp_url: str = "",
                    min_output_size: int = 0, raw: bool = False) -> dict:
    try:
        url_lower = url.lower()

        # Raw mode: skip CDP and trafilatura, return raw text directly
        if raw or any(url_lower.startswith(p) for p in
            ["https://raw.githubusercontent.com/", "https://raw.github.com/",
             "https://gitlab.com/", "https://bitbucket.org/",
             "https://gist.githubusercontent.com/"]):
            try:
                resp = await AsyncFetcher.get(url, timeout=15, stealthy_headers=True)
                content = resp.body if isinstance(resp.body, str) else resp.body.decode("utf-8", errors="replace")
                return {"success": True, "url": url, "title": url.split("/")[-1],
                        "content": content.strip()[:max_chars],
                        "method": "raw"}
            except Exception as e:
                return {"success": False, "url": url, "error": f"Raw fetch failed: {e}"}

        # CDP-first: use Helium browser when available (real cookies, no CAPTCHA)
        if cdp_url:
            try:
                page = await _get_optimized_page(block_resources=True)
                try:
                    await page.goto(url, wait_until="commit", timeout=15000)
                    await page.wait_for_load_state("domcontentloaded", timeout=15000)
                    try:
                        await page.wait_for_load_state("networkidle", timeout=15000)
                    except Exception:
                        pass
                    # Detect & wait for Cloudflare challenge
                    cf_detected = await page.evaluate("document.title.includes('Just a moment') || document.body.innerText.includes('Checking your browser') || document.body.innerText.includes('Just a moment') || document.querySelector('#challenge-spinner, #cf-challenge-running, .cf-browser-verification') !== null")
                    if cf_detected:
                        for _ in range(120):
                            await asyncio.sleep(1)
                            done = await page.evaluate("document.title !== 'Just a moment' && !document.body.innerText.includes('Checking your browser') && !document.body.innerText.includes('Just a moment') && document.querySelector('#challenge-spinner, #cf-challenge-running, .cf-browser-verification') === null")
                            if done:
                                break
                    text = await page.evaluate(_CDP_EXTRACT_MARKDOWN_JS)
                    if text and text.strip():
                        content = text.strip()
                    else:
                        html_c = await page.evaluate("document.documentElement.outerHTML")
                        content = trafilatura.extract(
                            html_c, output_format='markdown', include_links=include_links,
                            include_images=include_images, include_tables=include_tables,
                            deduplicate=deduplicate, fast=True, url=url,
                        ) or await page.evaluate("document.body.innerText || ''")
                    if isinstance(content, bytes):
                        content = content.decode("utf-8", errors="replace")
                    title = await page.evaluate("document.title")
                    return {"success": True, "url": url, "title": title or "",
                            "content": (content or "")[:max_chars],
                            "method": "cdp"}
                finally:
                    try:
                        await page.close()
                    except Exception:
                        pass
                    asyncio.ensure_future(_cleanup_orphan_tabs())
            except Exception:
                pass  # Fall through to httpx

        if url_lower.endswith('.pdf'):
            try:
                import asyncio, os, shutil, tempfile, opendataloader_pdf
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
                md_text = md.read_text("utf-8", errors="replace")[:max_chars] if md else "[empty PDF]"
                return {"success": True, "url": url, "title": url.split("/")[-1], "content": md_text}
            except Exception:
                pass  # Fall through to httpx
        if url_lower.endswith('.epub'):
            resp = await AsyncFetcher.get(url, timeout=20, stealthy_headers=True)
            content = _extract_epub(
                resp.body if isinstance(resp.body, bytes) else resp.body.encode(), max_chars)
            return {"success": True, "url": url, "title": url.split("/")[-1],
                    "content": content}
        if url_lower.endswith(('.docx', '.doc')):
            resp = await AsyncFetcher.get(url, timeout=20, stealthy_headers=True)
            content = _extract_docx(
                resp.body if isinstance(resp.body, bytes) else resp.body.encode(), max_chars)
            return {"success": True, "url": url, "title": url.split("/")[-1],
                    "content": content}
        resp = await AsyncFetcher.get(url, timeout=15, stealthy_headers=True)
        raw_html = resp.body if isinstance(resp.body, str) else resp.body.decode("utf-8", errors="replace")
        if not main_content_only:
            content = (resp.get_all_text() or "").strip()
            return {"success": True, "url": url, "status": resp.status,
                    "content": content[:max_chars] + ("\n[...truncated]" if len(content) > max_chars else "")}
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
            content = d.get('text', '')
            title = d.get('title')
        else:
            content = result or ''
        if not content:
            content = trafilatura.extract(raw_html, output_format='txt',
                                          with_metadata=False, url=url) or ''
        if not content.strip():
            try:
                content = (resp.get_all_text() or '').strip()
            except Exception:
                content = raw_html.strip()
        clean = content.strip()[:max_chars]
        # Cloudflare challenge detection in httpx path
        cf_keywords = ["just a moment", "checking your browser", "cf-challenge", "cloudflare ray id"]
        if clean and sum(1 for kw in cf_keywords if kw in clean[:600].lower()) >= 2:
            return {"success": False, "url": url, "error": "Cloudflare challenge detected — auto-fallback to CDP in progress."}
        if len(clean) == max_chars:
            clean += "\n[...truncated]"
        if min_output_size and len(clean) < min_output_size:
            return {"success": False, "url": url, "error": f"Content too short ({len(clean)} < {min_output_size} chars)"}
        meta_str = trafilatura.extract(raw_html, output_format='json', with_metadata=True,
                                       include_links=False, include_tables=False,
                                       url=url) if clean else None
        meta = {}
        if isinstance(meta_str, str) and meta_str.startswith('{'):
            try:
                meta = json.loads(meta_str).get("metadata", {})
            except Exception:
                pass
        return {"success": True, "url": url, "status": resp.status,
                "title": title or meta.get("title", ""),
                "content": clean, "metadata": {k: v for k, v in meta.items() if v}}
    except Exception as e:
        return {"success": False, "url": url, "error": str(e)}


_CDP_EXTRACT_MARKDOWN_JS = """
() => {
    const clone = document.cloneNode(true);
    clone.querySelectorAll('script, style, noscript, iframe[src*="ads"], nav, footer, aside').forEach(e => e.remove());
    const article = clone.querySelector('article') || clone.querySelector('[role="main"]') || clone.querySelector('main') || clone.body;
    return article ? article.innerText : '';
}
"""


async def _cdp_extract_content(page, css_selector: str | None, extraction_type: str) -> str:
    if css_selector:
        el = await page.query_selector(css_selector)
        if el:
            return await el.inner_text()
        return await page.evaluate("document.body.innerText || ''")
    if extraction_type == "html":
        return await page.evaluate("document.documentElement.outerHTML")
    if extraction_type == "markdown":
        text = await page.evaluate(_CDP_EXTRACT_MARKDOWN_JS)
        if text and text.strip():
            return text.strip()
        html_c = await page.evaluate("document.documentElement.outerHTML")
        content = trafilatura.extract(html_c, output_format='markdown', fast=True, url=url)
        return content or await page.evaluate("document.body.innerText || ''")
    return await page.evaluate("document.body.innerText || ''")


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
    page_action=None, page_setup=None) -> dict:
    # ── CDP-first path (primary) ──────────────────────────────────
    if cdp_url:
        last_err = None
        for attempt in range(max(retries, 1)):
            page = None
            try:
                page = await _get_optimized_page(block_resources=disable_resources)
                if blocked_domains:
                    try:
                        cdp_session = await page.context.new_cdp_session(page)
                        await cdp_session.send("Network.enable")
                        await cdp_session.send("Network.setBlockedURLs", {"urls": list(blocked_domains)})
                        await cdp_session.detach()
                    except Exception:
                        pass
                if init_script:
                    try:
                        await page.add_init_script(init_script)
                    except Exception:
                        pass
                goto_timeout = min(timeout, 15000)
                await page.goto(url, wait_until="commit", timeout=goto_timeout)
                dc_timeout = min(timeout, 15000)
                await page.wait_for_load_state("domcontentloaded", timeout=dc_timeout)
                if network_idle:
                    try:
                        ni_timeout = min(timeout, 15000)
                        await page.wait_for_load_state("networkidle", timeout=ni_timeout)
                    except Exception:
                        pass
                if wait_selector:
                    try:
                        state_map = {"attached": "attached", "detached": "detached",
                                     "visible": "visible", "hidden": "hidden"}
                        await page.wait_for_selector(
                            wait_selector,
                            state=state_map.get(wait_selector_state, "attached"),
                            timeout=10000)
                    except Exception:
                        pass
                # Detect & wait for Cloudflare challenge
                try:
                    cf = await page.evaluate("document.title.includes('Just a moment') || document.body.innerText.includes('Checking your browser') || document.body.innerText.includes('Just a moment') || document.querySelector('#challenge-spinner, #cf-challenge-running, .cf-browser-verification') !== null")
                    if cf:
                        for _ in range(120):
                            await asyncio.sleep(1)
                            done = await page.evaluate("document.title !== 'Just a moment' && !document.body.innerText.includes('Checking your browser') && !document.body.innerText.includes('Just a moment') && document.querySelector('#challenge-spinner, #cf-challenge-running, .cf-browser-verification') === null")
                            if done:
                                break
                except Exception:
                    pass
                content = await _cdp_extract_content(page, css_selector, extraction_type)
                if isinstance(content, bytes):
                    content = content.decode("utf-8", errors="replace")
                title = await page.evaluate("document.title")
                return {"success": True, "url": url, "title": title or "",
                        "content": (content or "")[:50000], "method": "cdp",
                        "attempt": attempt + 1}
            except Exception as e:
                last_err = e
                if page:
                    try:
                        await page.close()
                    except Exception:
                        pass
                if attempt < retries - 1:
                    await asyncio.sleep(min(0.5 * (attempt + 1), 2.0))
                    continue
                break
            finally:
                if page:
                    try:
                        await page.close()
                    except Exception:
                        pass
                asyncio.ensure_future(_cleanup_orphan_tabs())
        # CDP failed after all retries — fall through to scrapling

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
            if page_action:
                fk["page_action"] = page_action
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
