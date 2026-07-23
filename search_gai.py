"""Google AI Mode search via hidden CDP page to the user's Helium browser session.

Architecture:
- Single CDP connection to Helium (http://127.0.0.1:9222), shared across calls
- Hidden pages via Target.createTarget(background=true) — no tab flashing
- 4-stage completion detection (SVG → aria-label → text → timeout)
- SERPO citation extraction with [CITE-N] markers → sequential footnotes

Reference: https://github.com/PleasePrompto/google-ai-mode-mcp
"""

from __future__ import annotations

import asyncio
import base64
import logging
import os
import re
import time
import urllib.parse
from typing import Any

from config import GOOGLE_AI_URL, HELIUM_CDP, get_http_client
from cdp_client import CDPPage, get_cdp_session, close_cdp

logger = logging.getLogger("gai")

# ── Multi-language constants ─────────────────────────────────────

CITATION_SELECTORS = [
    '[aria-label="View related links"]', '[aria-label*="Related links"]',
    '[aria-label*="Sources"]', '[aria-label="Zugehörige Links anzeigen"]',
    '[aria-label*="Zugehörige Links"]', '[aria-label*="Quellen"]',
    '[aria-label*="Liens associés"]', '[aria-label*="Sources"]',
    '[aria-label*="Enlaces relacionados"]', '[aria-label*="Fuentes"]',
    '[aria-label*="Gerelateerde links"]', '[aria-label*="Bronnen"]',
    '[aria-label*="Link correlati"]', '[aria-label*="Fonti"]',
    'button[aria-label*="links" i]',
]

AI_COMPLETION_TEXT_INDICATORS = [
    "AI-generated", "AI Overview", "Generative AI is experimental",
    "KI-Antworten", "KI-generiert", "Generative KI",
    "AI-gegenereerd", "AI-overzicht",
    "Las respuestas de la IA", "Resumen de IA",
    "Réponses IA", "Aperçu de l'IA",
    "Risposte IA", "Panoramica IA",
]

GAI_ERROR_TEXT_INDICATORS = [
    "something went wrong and an ai response wasn't generated",
]

CUTOFF_MARKERS = [
    "AI-generated answers may contain mistakes", "AI can make mistakes",
    "Generative AI is experimental", "AI overviews are experimental",
    "KI-Antworten können Fehler enthalten",
    "AI-reacties kunnen fouten bevatten",
    "Las respuestas de la IA pueden contener errores",
    "Les réponses de l'IA peuvent contenir des erreurs",
    "Le risposte dell'IA possono contenere errori",
    # Google AI Mode post-answer UI noise (matched in body text)
    "Good response", "Bad response",
    "Share public link", "This public link shares",
    "Thanks for letting us know", "A copy of this chat",
    "Make a legal removal request",
]

CUTOFF_MARKERS_LOWER = [m.lower() for m in CUTOFF_MARKERS]

CAPTCHA_INDICATORS = [
    "/sorry/index", "unusual traffic", "captcha", "recaptcha",
    "are you a robot", "verify you are human", "automated queries",
]

AI_MODE_BLOCKED = [
    "not available in your country", "not available in your region",
    "not available in your language", "ai mode is not available",
    "ai mode isn't available", "der ki-modus ist in ihrem land",
    "le mode ia n'est pas disponible",
]

# GAI only accepts images + PDF — other formats silently fail on the backend
MIME_MAP = {
    '.avif': 'image/avif', '.bmp': 'image/bmp',
    '.heic': 'image/heic', '.heif': 'image/heif',
    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg',
    '.png': 'image/png', '.webp': 'image/webp',
    '.pdf': 'application/pdf',
}

ANTI_DETECT_JS = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'plugins', {get: () => [1,2,3,4,5]});
window.chrome = {runtime: {}, csi: function(){}, loadTimes: function(){}};
Object.defineProperty(navigator, 'permissions', {
    get: () => ({ query: (params) => Promise.resolve({state: 'granted', onchange: null}) })
});
Object.defineProperty(navigator, 'maxTouchPoints', {get: () => 0});
"""

RESOURCE_BLOCK_PATTERNS = [
    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.webp", "*.ico",
    "*.woff", "*.woff2", "*.ttf", "*.eot", "*.otf",
    "*.mp4", "*.webm", "*.ogg", "*.mp3", "*.wav",
    "*/ads/*", "*/analytics/*", "*/tracking/*",
]

# ── Page lifecycle semaphore ──────────────────────────────────────

_page_semaphore = asyncio.Semaphore(5)


async def _get_optimized_page(block_resources: bool = True) -> CDPPage:
    """Create a hidden CDP page with anti-detection JS and optional resource blocking.

    Used by fetch.py, screenshot.py, and crawl.py for stealth CDP operations.
    Also applies protective defaults: disable images, deny downloads,
    ignore cert errors, and (optionally) block resource URLs.
    """
    async with _page_semaphore:
        session = await get_cdp_session()
        if not session:
            raise ConnectionError("Cannot connect to Helium CDP")
        page = await session.create_page()
        try:
            await page.add_init_script(ANTI_DETECT_JS)
            # Protective defaults — save RAM, prevent leaks, avoid hangs
            await page.disable_images()
            await page.set_download_behavior("deny")
            await page.ignore_certificate_errors(True)
            if block_resources:
                try:
                    await page.set_blocked_resources(RESOURCE_BLOCK_PATTERNS)
                except Exception:
                    pass
            return page
        except Exception:
            await page.close()
            raise


async def _cleanup_orphan_tabs():
    """Close stale about:blank pages left by broken searches."""
    try:
        session = await get_cdp_session()
        if not session:
            return
        result = await session.send("Target.getTargets")
        for t in result.get("targetInfos", []):
            if t["url"] == "about:blank":
                await session.send("Target.closeTarget", {"targetId": t["targetId"]})
    except Exception as e:
        logger.warning("orphan tab cleanup: %s", e)


# ── GoogleAIClient ────────────────────────────────────────────────

class CompletionResult:
    """Result from _wait_for_completion, letting the caller know which stage fired."""
    def __init__(self, success: bool, method: str):
        self.success = success
        self.method = method  # "svg" | "aria" | "text" | "timeout" | "captcha" | "blocked"


class GoogleAIClient:
    """One-shot search via Google AI Mode, connecting to the user's Helium CDP session."""

    def __init__(self, cdp_url: str | None = None):
        self._cdp_url = cdp_url or HELIUM_CDP
        self._page: CDPPage | None = None

    async def search(self, query: str, search_prompt: str = "",
                     gl: str = "", hl: str = "en", tbs: str = "", pws: str = "",
                     upload_urls: list[str] | None = None) -> dict:
        """Execute a single search via Google AI Mode.

        Returns {"success": bool, "result": {"answer": str, "sources": list, "followUp": str}}
        or {"success": False, "error": str}.
        """
        try:
            p = await _get_optimized_page(block_resources=False)
        except ConnectionError:
            return {"success": False, "error": "Cannot connect to Helium CDP"}
        self._page = p
        try:

            # ── Navigate ──
            final_query = " ".join(filter(None, [query, search_prompt]))
            if upload_urls:
                bare_url = f"{GOOGLE_AI_URL}?udm=50"
                for k, v in [("hl", hl), ("gl", gl), ("tbs", tbs), ("pws", pws)]:
                    if v:
                        bare_url += f"&{k}={v}"
                await p.goto(bare_url, wait_until="domcontentloaded", timeout=20,
                            referrer="https://www.google.com/")
                await p.wait_for_load_state("networkidle", timeout=15)
                cap = await self._detect_captcha(p)
                if cap:
                    return {"success": False, "error": cap}
                uploaded = await self._upload_files(p, upload_urls)
                if uploaded:
                    await asyncio.sleep(1)
                    await p.type_text(final_query, selector="textarea")
                    await asyncio.sleep(0.5)
                    await p.evaluate('document.querySelector(\'[aria-label="Send"]\')?.click()')
                    await asyncio.sleep(0.3)
            else:
                params = [f"q={urllib.parse.quote_plus(final_query)}", "udm=50"]
                for k, v in [("hl", hl), ("gl", gl), ("tbs", tbs), ("pws", pws)]:
                    if v:
                        params.append(f"{k}={v}")
                url = f"{GOOGLE_AI_URL}?{'&'.join(params)}"
                await p.goto(url, wait_until="domcontentloaded", timeout=20,
                            referrer="https://www.google.com/")
                cap = await self._detect_captcha(p)
                if cap:
                    return {"success": False, "error": cap}

            # ── Wait for AI completion ──
            cr = await self._wait_for_completion(p, 240 if not upload_urls else 300)
            if not cr.success:
                return {"success": False, "error": f"Completion detection failed: {cr.method}"}

            # ── Expand "show more" sections ──
            try:
                await asyncio.wait_for(p.evaluate("""() => {
                    for (const btn of document.querySelectorAll('[aria-expanded="false"]')) {
                        const t = btn.innerText.toLowerCase();
                        if (t.includes('show more') || t.includes('mehr anzeigen')
                            || t.includes('meer weergeven')) btn.click();
                    }
                }"""), timeout=3)
                await asyncio.sleep(0.5)
            except Exception:
                pass

            # ── SERPO: Click citation buttons, insert [CITE-N] markers, extract sources ──
            try:
                await p.call_function("""(selectors) => {
                    function isVisible(el) {
                        if (!el) return false;
                        try {
                            const s = window.getComputedStyle(el);
                            const r = el.getBoundingClientRect();
                            return s.display!=='none' && s.visibility!=='hidden'
                                && s.opacity!=='0' && el.offsetParent!==null
                                && r.width > 0 && r.height > 0;
                        } catch(e) { return false; }
                    }
                    const turns = document.querySelectorAll('[data-subtree=aimc]');
                    const lastTurn = turns[turns.length - 1];
                    if (!lastTurn) return;
                    const mainCol = lastTurn.closest('[data-container-id="main-col"]');
                    const container = mainCol || lastTurn;
                    let buttons = [];
                    for (const sel of selectors) {
                        buttons = Array.from(container.querySelectorAll(sel));
                        if (buttons.filter(b => isVisible(b)).length > 0) break;
                    }
                    buttons = buttons.filter(b => isVisible(b));
                    for (let i = 0; i < buttons.length; i++) {
                        const marker = document.createElement('span');
                        marker.className = 'citation-marker';
                        marker.innerHTML = '<code>[CITE-' + i + ']</code>';
                        buttons[i].parentNode.insertBefore(marker, buttons[i].nextSibling);
                        try { buttons[i].click(); } catch(e) {}
                    }
                }""", CITATION_SELECTORS)
                await asyncio.sleep(0.5)
            except Exception:
                pass

            # ── Extract results (no inline args — all self-contained) ──
            serpo = await p.evaluate("""() => {
                const turns = document.querySelectorAll('[data-subtree=aimc]');
                const lastTurn = turns[turns.length - 1];
                const mainCol = lastTurn ? lastTurn.closest('[data-container-id="main-col"]') : null;
                const container = mainCol || lastTurn;
                const ansHtml = (() => {
                    if (!lastTurn) return '';
                    const c = lastTurn.cloneNode(true);
                    for (const e of c.querySelectorAll('button, style, script')) e.remove();
                    for (const e of c.querySelectorAll('[role=button]')) {
                        const txt = document.createTextNode(' ' + e.textContent + ' ');
                        e.parentNode.replaceChild(txt, e);
                    }
                    for (const m of c.querySelectorAll('.citation-marker')) m.remove();
                    return c.innerHTML;
                })();
                const seen = new Set();
                const srcs = [];
                for (const a of document.querySelectorAll('a[href]')) {
                    let h = a.href;
                    try {
                        const u = new URL(h);
                        if (u.hostname.includes('google.com') && u.pathname === '/url') {
                            const t = u.searchParams.get('q') || u.searchParams.get('url');
                            if (t) h = t;
                        }
                    } catch(e) {}
                    if (!h.startsWith('http') || seen.has(h)) continue;
                    if (['google.com','gstatic.com'].some(d => h.includes(d))) continue;
                    seen.add(h);
                    let t = a.innerText.trim().split('\\n')[0];
                    if (!t || t.length < 4) { try { t = new URL(h).hostname.replace('www.',''); } catch(e) {} }
                    let s = '';
                    const card = a.closest('li, [class]');
                    if (card) {
                        const ls = card.innerText.trim().split('\\n').filter(l => l.length > 30);
                        for (const lx of ls) { if (lx !== t && lx.length > 30) { s = lx.substring(0, 200); break; } }
                    }
                    srcs.push({ title: t.substring(0, 200), url: h.split('#')[0], snippet: s });
                }
                let fu = '';
                if (lastTurn) {
                    const blocks = lastTurn.innerText.trim().split('\\n').map(l => l.trim()).filter(l => l.length > 15);
                    if (blocks.length > 0) { const lb = blocks[blocks.length - 1];
                        if (lb.includes('?') && lb.length < 150) fu = lb; }
                }
                return { html: (container ? container.innerHTML : ''), answerHtml: ansHtml, sources: srcs.slice(0, 20), followUp: fu };
            }""") or {}

            answer_html = serpo.get("answerHtml", "")
            if not answer_html:
                return {"success": False, "error": "Could not extract AI response"}

            md = self._html_to_markdown(answer_html)
            # [CITE-N] → sequential footnotes
            cite_count = 0
            def replace_cite(m):
                nonlocal cite_count
                cite_count += 1
                return f"[{cite_count}]"
            md = re.sub(r'\[CITE-\d+\]', replace_cite, md or "")
            citation_sources = serpo.get("sources", [])
            if md and citation_sources:
                md += "\n\n---\n\n## Sources\n\n"
                for i, src in enumerate(citation_sources):
                    t = src.get("title", src.get("source", "") or src.get("url", ""))
                    u = src.get("url", "")
                    md += f"[{i+1}] {t}  \n{u}\n\n"

            return {
                "success": True,
                "result": {
                    "answer": md or "No content extracted",
                    "sources": citation_sources,
                    "followUp": serpo.get("followUp", ""),
                },
            }
        except Exception as e:
            logger.exception("GAI search failed")
            try:
                await p.terminate_execution()
            except Exception:
                pass
            return {"success": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            try:
                await p.close()
            except Exception:
                pass
            asyncio.ensure_future(_cleanup_orphan_tabs())

    # ── Completion detection ──────────────────────────────────────

    async def _wait_for_completion(self, p: CDPPage, deadline_seconds: float) -> CompletionResult:
        """4-stage detection: SVG thumbs-up → aria-label → text indicators → timeout.

        Also detects GAI error messages and returns early with failure.
        Returns at the deadline or when detection succeeds.
        """
        deadline = time.monotonic() + deadline_seconds
        while time.monotonic() < deadline:
            try:
                has_svg = await p.evaluate(
                    "!!document.querySelector('button svg[viewBox=\"3 3 18 18\"]')")
                if has_svg:
                    has_aimc = await p.evaluate(
                        "!!document.querySelector('[data-subtree=aimc]')")
                    if has_aimc:
                        return CompletionResult(True, "svg")
            except Exception:
                pass
            try:
                body = await p.evaluate("document.body.innerText")
                for err in GAI_ERROR_TEXT_INDICATORS:
                    if err in body.lower():
                        await p.terminate_execution()
                        return CompletionResult(False, err)
                if any(ind in body for ind in AI_COMPLETION_TEXT_INDICATORS):
                    has_aimc = await p.evaluate(
                        "!!document.querySelector('[data-subtree=aimc]')")
                    if has_aimc:
                        return CompletionResult(True, "text")
            except Exception:
                pass
            await asyncio.sleep(0.5)
        await p.terminate_execution()
        return CompletionResult(False, "timeout")

    # ── HTML → markdown ───────────────────────────────────────────

    def _html_to_markdown(self, html: str) -> str:
        try:
            import markdownify
            md = markdownify.markdownify(
                html, heading_style="ATX", bullets="-",
                strip=["script", "style", "noscript"],
            )
            md = re.sub(r'==+([^=]+)==+', r'\1', md)
            md = re.sub(r'!\[[^\]]*\]\(data:image\/[^)]+\)', '', md)
            md = re.sub(r'\[\]\([^)]*\)', '', md)
            body = md
            sources = ""
            si = md.rfind("## Sources")
            if si >= 0:
                body = md[:si].strip()
                sources = md[si:]
            lines = body.split("\n")
            cut_idx = None
            for i, line in enumerate(lines):
                sl = line.strip().lower()
                if any(m in sl for m in CUTOFF_MARKERS_LOWER):
                    cut_idx = i
                    break
            if cut_idx is not None:
                lines = lines[:cut_idx]
            noise_lines = {
                "Copy", "# Share public link", "Share public link",
                "Good response", "Bad response", "More",
                "Share", "Facebook", "Gmail", "X", "Reddit", "WhatsApp",
                "Thanks for letting us know",
            }
            while lines and (lines[-1].strip() in noise_lines or not lines[-1].strip()):
                lines.pop()
            while lines and any(lines[-1].strip().startswith(p) for p in
                ["This public link shares", "A copy of this chat",
                 "Google may use account", "Can't copy the link",
                 "Make a legal removal"]):
                lines.pop()
            body = "\n".join(lines).strip()
            md = body + "\n\n" + sources if sources else body
            md = re.sub(r'(?<![.!?\n])\n(?![*\-#\d\n])', ' ', md)
            md = re.sub(r'\n{3,}', '\n\n', md)
            return md.strip()
        except Exception:
            return html

    # ── CAPTCHA detection ─────────────────────────────────────────

    async def _detect_captcha(self, p: CDPPage) -> str | None:
        """Returns a reason string if CAPTCHA or blocking is detected, None otherwise."""
        try:
            if any(i in p.url.lower() for i in CAPTCHA_INDICATORS):
                return "CAPTCHA URL"
            body = (await p.evaluate("document.body.innerText")).lower()
            if any(i in body for i in AI_MODE_BLOCKED):
                return "AI Mode blocked in this region/language"
            if any(i in body for i in CAPTCHA_INDICATORS):
                return "CAPTCHA detected"
        except Exception:
            pass
        return None

    # ── File upload ───────────────────────────────────────────────

    async def _upload_files(self, p: CDPPage, upload_urls: list[str]) -> bool:
        """Upload files via DragEvent drop through the browser."""
        if not upload_urls:
            return False
        for _ in range(20):
            if await p.evaluate("!!document.querySelector('textarea')"):
                break
            await asyncio.sleep(0.5)
        uploaded = False
        for url in upload_urls:
            try:
                path = url.replace('file://', '')
                is_local = os.path.isfile(path)
                if is_local:
                    with open(path, 'rb') as f:
                        content = f.read()
                    if len(content) > 10_000_000:
                        continue
                    b64 = base64.b64encode(content).decode('ascii')
                    ext = os.path.splitext(path.lower())[1]
                    name = os.path.basename(path)
                    mime = MIME_MAP.get(ext, 'application/octet-stream')
                    ok = await p.call_function("""async ({b64, mime, name}) => {
                        const r = await fetch(`data:${mime};base64,${b64}`);
                        const blob = await r.blob();
                        const f = new File([blob], name, {type: mime});
                        const dt = new DataTransfer(); dt.items.add(f);
                        const ta = document.querySelector('textarea');
                        if (!ta) return false;
                        ta.dispatchEvent(new DragEvent('dragenter',
                            {bubbles: true, cancelable: true, dataTransfer: dt}));
                        ta.dispatchEvent(new DragEvent('dragover',
                            {bubbles: true, cancelable: true, dataTransfer: dt}));
                        ta.dispatchEvent(new DragEvent('drop',
                            {bubbles: true, cancelable: true, dataTransfer: dt}));
                        await new Promise(r => setTimeout(r, 2000));
                        return {ok: true, size: blob.size, type: mime};
                    }""", {"b64": b64, "mime": mime, "name": name})
                    if isinstance(ok, dict) and ok.get("ok"):
                        uploaded = True
                        await asyncio.sleep(1)
                    continue
                # Remote URL
                ok = await p.call_function("""async (url) => {
                    const resp = await fetch(url);
                    if (!resp.ok) return {error: `HTTP ${resp.status}`};
                    const blob = await resp.blob();
                    const name = url.split('/').pop()?.split('?')[0] || 'upload';
                    const f = new File([blob], name, {type: blob.type});
                    const dt = new DataTransfer(); dt.items.add(f);
                    const ta = document.querySelector('textarea');
                    if (!ta) return {error: 'no textarea'};
                    ta.dispatchEvent(new DragEvent('dragenter',
                        {bubbles: true, cancelable: true, dataTransfer: dt}));
                    ta.dispatchEvent(new DragEvent('dragover',
                        {bubbles: true, cancelable: true, dataTransfer: dt}));
                    ta.dispatchEvent(new DragEvent('drop',
                        {bubbles: true, cancelable: true, dataTransfer: dt}));
                    await new Promise(r => setTimeout(r, 2000));
                    return {ok: true, size: blob.size, type: blob.type};
                }""", url)
                if isinstance(ok, dict) and ok.get("ok"):
                    uploaded = True
                    await asyncio.sleep(1)
                    continue
                # Fallback: server-side download
                c = get_http_client()
                resp = await c.get(url, headers={
                    "User-Agent": "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36"},
                    follow_redirects=True)
                if resp.status_code == 200 and len(resp.content) <= 10_000_000:
                    b64 = base64.b64encode(resp.content).decode('ascii')
                    ext = os.path.splitext(urllib.parse.urlparse(url).path)[1].lower() or '.bin'
                    mime = MIME_MAP.get(ext, resp.headers.get('content-type', 'application/octet-stream'))
                    ok2 = await p.call_function("""async ({b64, mime, name}) => {
                        const r = await fetch(`data:${mime};base64,${b64}`);
                        const blob = await r.blob();
                        const f = new File([blob], name, {type: mime});
                        const dt = new DataTransfer(); dt.items.add(f);
                        const ta = document.querySelector('textarea');
                        if (!ta) return false;
                        ta.dispatchEvent(new DragEvent('dragenter',
                            {bubbles: true, cancelable: true, dataTransfer: dt}));
                        ta.dispatchEvent(new DragEvent('dragover',
                            {bubbles: true, cancelable: true, dataTransfer: dt}));
                        ta.dispatchEvent(new DragEvent('drop',
                            {bubbles: true, cancelable: true, dataTransfer: dt}));
                        await new Promise(r => setTimeout(r, 2000));
                        return true;
                    }""", {"b64": b64, "mime": mime, "name": f"upload{ext}"})
                    if ok2:
                        uploaded = True
                        await asyncio.sleep(1)
            except Exception:
                pass
        return uploaded

    async def close(self):
        """Cleanup GAI client (page cleanup only — CDP is shared)."""
        if self._page:
            try:
                await self._page.close()
            except Exception:
                pass
            self._page = None


# ── Singleton accessor ────────────────────────────────────────────

_gaiclient: GoogleAIClient | None = None
_gai_lock = asyncio.Lock()


async def get_gai_client(cdp_url: str | None = None) -> GoogleAIClient | None:
    """Get or create the singleton GAI client. No availability check — handled by search()."""
    global _gaiclient
    if _gaiclient is not None:
        return _gaiclient
    async with _gai_lock:
        if _gaiclient is not None:
            return _gaiclient
        url = cdp_url or HELIUM_CDP
        if not url:
            return None
        # Quick liveness check — connect to CDP, create one page, close it
        try:
            session = await get_cdp_session(url)
            if not session:
                return None
            p = await session.create_page()
            try:
                await p.add_init_script(ANTI_DETECT_JS)
                await p.evaluate("1+1")
            finally:
                await p.close()
            _gaiclient = GoogleAIClient(cdp_url=url)
            return _gaiclient
        except Exception as e:
            logger.warning("GAI CDP check failed: %s", e)
            return None


async def gai_shutdown():
    """Shutdown the GAI client and CDP connection."""
    global _gaiclient
    if _gaiclient:
        _gaiclient = None
    await close_cdp()
