"""Google AI Mode search via hidden CDP page to the user's Helium browser session.

Architecture:
- Single CDP connection to Helium (http://127.0.0.1:9222), shared across calls
- Hidden pages via Target.createTarget(hidden=True) — no tab flashing
- 4-stage completion detection (SVG → aria-label → text → timeout)
- SERPO citation extraction with [CITE-N] markers → sequential footnotes

Reference: https://github.com/PleasePrompto/google-ai-mode-mcp
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
import os
import re
import time
import urllib.parse
from typing import Any

from config import GOOGLE_AI_URL, HELIUM_CDP, get_http_client

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

CUTOFF_MARKERS = [
    "AI-generated answers may contain mistakes", "AI can make mistakes",
    "Generative AI is experimental", "AI overviews are experimental",
    "KI-Antworten können Fehler enthalten",
    "AI-reacties kunnen fouten bevatten",
    "Las respuestas de la IA pueden contener errores",
    "Les réponses de l'IA peuvent contenir des erreurs",
    "Le risposte dell'IA possono contenere errori",
    # Google AI Mode post-answer UI noise
    "Good response", "Bad response",
    "Share public link", "This public link shares",
    "Thanks for letting us know", "A copy of this chat",
    "Make a legal removal request",
]

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

MIME_MAP = {
    '.jpg': 'image/jpeg', '.jpeg': 'image/jpeg', '.png': 'image/png',
    '.gif': 'image/gif', '.webp': 'image/webp', '.pdf': 'application/pdf',
    '.svg': 'image/svg+xml', '.bmp': 'image/bmp', '.avif': 'image/avif',
    '.mp4': 'video/mp4', '.mov': 'video/mp4', '.mp3': 'audio/mpeg',
    '.wav': 'audio/wav', '.ogg': 'audio/ogg',
    '.txt': 'text/plain', '.md': 'text/markdown', '.csv': 'text/csv',
    '.json': 'application/json', '.xml': 'text/xml', '.html': 'text/html',
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

# ── Shared CDP browser singleton ─────────────────────────────────

_pw = None
_browser = None
_browser_lock = asyncio.Lock()

async def _get_browser():
    """Get or create a shared CDP connection to Helium."""
    global _pw, _browser
    async with _browser_lock:
        if _browser:
            try:
                ctx = _browser.contexts
                if ctx and ctx[0].pages:
                    await asyncio.wait_for(ctx[0].pages[0].title(), timeout=3)
                    return _browser
            except Exception:
                logger.debug("CDP browser stale, reconnecting")
                await _shutdown()
        from playwright.async_api import async_playwright
        _pw = await async_playwright().start()
        _browser = await _pw.chromium.connect_over_cdp(HELIUM_CDP)
        logger.info("Connected to Helium CDP")
        return _browser

async def _shutdown():
    """Close everything — CDP, Playwright."""
    global _pw, _browser
    if _browser:
        try:
            await _browser.close()
        except Exception:
            pass
        _browser = None
    if _pw:
        try:
            await _pw.stop()
        except Exception:
            pass
        _pw = None

async def _create_hidden_page(ctx):
    """Create a page without stealing focus.

    Uses ctx.new_page() — same approach as the reference implementation.
    Pages are closed after use; with hidden-target CDP being unreliable
    (Playwright doesn't discover hidden CDP targets as Page objects),
    a normal new page is simpler and always works.
    """
    return await ctx.new_page()

async def _cleanup_orphan_tabs(exclude_page=None):
    """Close stale about:blank pages left by broken searches."""
    global _browser
    if not _browser:
        return
    try:
        ctx = _browser.contexts[0]
        for p in list(ctx.pages):
            try:
                if p.url == "about:blank" and p is not exclude_page:
                    await p.close()
            except Exception:
                pass
    except Exception:
        pass


_page_semaphore = asyncio.Semaphore(5)

async def _get_optimized_page(block_resources: bool = True):
    """Create a hidden page with anti-detection JS and optional resource blocking.

    Used by fetch.py and screenshot.py for stealth CDP operations.
    """
    async with _page_semaphore:
        b = await _get_browser()
        ctx = b.contexts[0]
        page = await _create_hidden_page(ctx)
        await page.add_init_script(ANTI_DETECT_JS)
        if block_resources:
            try:
                cdp = await ctx.new_cdp_session(page)
                await cdp.send("Network.enable")
                await cdp.send("Network.setBlockedURLs", {"urls": [
                    "*.png", "*.jpg", "*.jpeg", "*.gif", "*.svg", "*.webp", "*.ico",
                    "*.woff", "*.woff2", "*.ttf", "*.eot", "*.otf",
                    "*.mp4", "*.webm", "*.ogg", "*.mp3", "*.wav",
                    "*/ads/*", "*/analytics/*", "*/tracking/*",
                ]})
                await cdp.detach()
            except Exception:
                pass
        return page

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
        self._page = None

    async def _get_context(self):
        b = await _get_browser()
        return b.contexts[0]

    async def search(self, query: str, search_prompt: str = "", pro_mode: bool = False,
                     gl: str = "", hl: str = "en", tbs: str = "", pws: str = "",
                     upload_urls: list[str] | None = None) -> dict:
        """Execute a single search via Google AI Mode.

        Returns {"success": bool, "result": {"answer": str, "sources": list, "followUp": str}}
        or {"success": False, "error": str}.
        """
        b = await _get_browser()
        ctx = b.contexts[0]
        p = await _create_hidden_page(ctx)
        self._page = p
        try:
            await p.add_init_script(ANTI_DETECT_JS)

            # ── Navigate ──
            final_query = " ".join(filter(None, [query, search_prompt]))
            if upload_urls:
                bare_url = f"{GOOGLE_AI_URL}?udm=50"
                for k, v in [("hl", hl), ("gl", gl), ("tbs", tbs), ("pws", pws)]:
                    if v:
                        bare_url += f"&{k}={v}"
                await p.goto(bare_url, wait_until="domcontentloaded", timeout=20000,
                             referer="https://www.google.com/")
                await p.wait_for_load_state("networkidle", timeout=15000)
                cap = await self._detect_captcha(p)
                if cap:
                    return {"success": False, "error": cap}
                uploaded = await self._upload_files(p, upload_urls)
                if uploaded:
                    await p.evaluate(f"""() => {{
                        const ta = document.querySelector('textarea');
                        if (ta) {{ ta.focus(); ta.value = {json.dumps(final_query)};
                            ta.dispatchEvent(new Event('input', {{bubbles: true}})); }}
                    }}""")
                    await asyncio.sleep(0.5)
                    await p.evaluate("""() => {
                        const btn = document.querySelector('[aria-label="Send"]');
                        if (btn) { btn.disabled = false; btn.click(); }
                    }""")
                    await asyncio.sleep(0.3)
            else:
                params = [f"q={urllib.parse.quote_plus(final_query)}", "udm=50"]
                for k, v in [("hl", hl), ("gl", gl), ("tbs", tbs), ("pws", pws)]:
                    if v:
                        params.append(f"{k}={v}")
                url = f"{GOOGLE_AI_URL}?{'&'.join(params)}"
                await p.goto(url, wait_until="domcontentloaded", timeout=20000,
                             referer="https://www.google.com/")
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
                await p.evaluate("""(selectors) => {
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
                        ref = buttons[i].nextSibling ? buttons[i] : buttons[i];
                        ref.parentNode.insertBefore(marker, ref.nextSibling);
                        try { buttons[i].click(); } catch(e) {}
                    }
                }""", CITATION_SELECTORS)
                await asyncio.sleep(0.5)
            except Exception:
                pass

            # ── Extract results ──
            serpo = await p.evaluate("""() => {
                const turns = document.querySelectorAll('[data-subtree=aimc]');
                const lastTurn = turns[turns.length - 1];
                const mainCol = lastTurn ? lastTurn.closest('[data-container-id="main-col"]') : null;
                const container = mainCol || lastTurn;
                const ansHtml = (() => {
                    if (!lastTurn) return '';
                    const c = lastTurn.cloneNode(true);
                    for (const e of c.querySelectorAll('[role=button], button, style, script')) e.remove();
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
            return {"success": False, "error": f"{type(e).__name__}: {e}"}
        finally:
            try:
                await p.close()
            except Exception:
                pass
            asyncio.ensure_future(_cleanup_orphan_tabs(exclude_page=None))

    # ── Completion detection ──────────────────────────────────────

    async def _wait_for_completion(self, p, deadline_seconds: float) -> CompletionResult:
        """4-stage detection: SVG thumbs-up → aria-label → text indicators → timeout.

        Returns at the deadline or when detection succeeds.
        """
        deadline = time.monotonic() + deadline_seconds
        # Stage 1-2: SVG button + aria-label (polled together)
        while time.monotonic() < deadline:
            try:
                svg = await p.query_selector('button svg[viewBox="3 3 18 18"]')
                if svg:
                    has_aimc = await p.evaluate("!!document.querySelector('[data-subtree=aimc]')")
                    if has_aimc:
                        return CompletionResult(True, "svg")
            except Exception:
                pass
            try:
                body = await p.evaluate("document.body.innerText")
                if any(ind in body for ind in AI_COMPLETION_TEXT_INDICATORS):
                    has_aimc = await p.evaluate("!!document.querySelector('[data-subtree=aimc]')")
                    if has_aimc:
                        return CompletionResult(True, "text")
            except Exception:
                pass
            await asyncio.sleep(0.5)
        return CompletionResult(True, "timeout")

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
            for marker in CUTOFF_MARKERS:
                idx = md.find(marker)
                if idx >= 0:
                    md = md[:idx].strip()
                    break
            md = re.sub(r'(?<![.!?\n])\n(?![*\-#\d\n])', ' ', md)
            md = re.sub(r'\n{3,}', '\n\n', md)
            return md.strip()
        except Exception:
            return html

    # ── CAPTCHA detection ─────────────────────────────────────────

    async def _detect_captcha(self, p) -> str | None:
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

    async def _upload_files(self, p, upload_urls: list[str]) -> bool:
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
                    ok = await p.evaluate("""async ({b64, mime, name}) => {
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
                ok = await p.evaluate("""async (url) => {
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
                    ok2 = await p.evaluate("""async ({b64, mime, name}) => {
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
            b = await _get_browser()
            ctx = b.contexts[0]
            p = await _create_hidden_page(ctx)
            try:
                await p.add_init_script(ANTI_DETECT_JS)
                await p.goto("about:blank", timeout=10000)
            finally:
                await p.close()
                asyncio.ensure_future(_cleanup_orphan_tabs())
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
    await _shutdown()
