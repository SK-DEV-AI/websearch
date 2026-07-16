"""Direct CDP client for Helium browser — replaces Playwright CDP connections.

Architecture:
- Single shared WebSocket connection to Helium's browser-level CDP endpoint
- Multiple page sessions multiplexed over the same connection
- Page operations (navigate, evaluate, screenshot, ARIA snapshot, etc.)
  through CDP protocol, without Playwright's Node.js driver overhead

CDP protocol reference: https://chromedevtools.github.io/devtools-protocol/
"""

from __future__ import annotations

import asyncio
import base64
import json
import logging
from typing import Any, Callable

import httpx
import websockets

from config import HELIUM_CDP

logger = logging.getLogger("cdp")


class CDPError(Exception):
    """CDP protocol error from browser."""

    def __init__(self, error: dict):
        self.code = error.get("code")
        self.message = error.get("message")
        super().__init__(f"CDP error {self.code}: {self.message}")


class CDPSession:
    """Persistent CDP connection to the browser, multiplexing page sessions.

    One WebSocket to the browser-level endpoint handles commands and events
    for all page sessions. Each page gets a sessionId via Target.attachToTarget.
    """

    def __init__(self):
        self._ws: websockets.WebSocketClientProtocol | None = None
        self._msg_id = 0
        self._pending: dict[int, asyncio.Future] = {}
        self._listeners: dict[str, list[Callable]] = {}
        self._reader_task: asyncio.Task | None = None
        self._closed = False

    async def connect(self, cdp_url: str | None = None) -> bool:
        """Connect to Helium's browser CDP WebSocket.

        Discovers the WS URL from the HTTP debug endpoint automatically,
        because the browser UUID changes on restart.
        """
        if self._ws and not self._closed:
            return True

        base = (cdp_url or HELIUM_CDP).rstrip("/")
        try:
            async with httpx.AsyncClient() as c:
                resp = await c.get(f"{base}/json/version", timeout=5)
                if resp.status_code != 200:
                    logger.error("CDP /json/version returned %s", resp.status_code)
                    return False
                data = resp.json()
                ws_url = data.get("webSocketDebuggerUrl")
                if not ws_url:
                    logger.error("No webSocketDebuggerUrl in /json/version")
                    return False
        except Exception as e:
            logger.error("CDP version fetch failed: %s", e)
            return False

        try:
            self._ws = await asyncio.wait_for(
                websockets.connect(ws_url, max_size=4 * 1024 * 1024, open_timeout=10),
                timeout=12,
            )
        except Exception as e:
            logger.error("CDP WebSocket connect failed: %s", e)
            return False

        self._closed = False
        self._reader_task = asyncio.create_task(self._reader_loop())
        logger.info("Connected to Helium CDP")
        return True

    async def _reader_loop(self):
        """Background: read WebSocket frames and dispatch to pending or listeners."""
        try:
            async for raw in self._ws:
                try:
                    data = json.loads(raw)
                except json.JSONDecodeError:
                    continue

                msg_id = data.get("id")
                method = data.get("method")

                if msg_id is not None:
                    fut = self._pending.pop(msg_id, None)
                    if fut and not fut.done():
                        if "error" in data:
                            fut.set_exception(CDPError(data["error"]))
                        else:
                            fut.set_result(data.get("result", {}))

                if method:
                    session_id = data.get("sessionId")
                    cbs = self._listeners.get(method, [])
                    for cb in cbs:
                        try:
                            cb(data.get("params", {}), session_id)
                        except Exception:
                            logger.exception("CDP listener error")

        except websockets.ConnectionClosed as e:
            logger.debug("CDP WS closed: %s", e.code)
        except Exception as e:
            logger.error("CDP reader error: %s", e)
        finally:
            self._closed = True
            for mid, fut in self._pending.items():
                if not fut.done():
                    fut.cancel()
            self._pending.clear()

    async def send(
        self,
        method: str,
        params: dict | None = None,
        session_id: str | None = None,
        timeout: float = 30,
    ) -> dict:
        """Send a CDP command and wait for the response."""
        if self._closed or not self._ws:
            raise ConnectionError("CDP not connected")

        self._msg_id += 1
        msg_id = self._msg_id
        msg: dict[str, Any] = {"id": msg_id, "method": method}
        if params:
            msg["params"] = params
        if session_id:
            msg["sessionId"] = session_id

        loop = asyncio.get_running_loop()
        future = loop.create_future()
        self._pending[msg_id] = future

        try:
            await self._ws.send(json.dumps(msg))
            return await asyncio.wait_for(future, timeout=timeout)
        except asyncio.TimeoutError as e:
            self._pending.pop(msg_id, None)
            raise TimeoutError(f"CDP {method} timed out after {timeout}s") from e

    def on(self, event: str, callback: Callable):
        """Register an event listener."""
        self._listeners.setdefault(event, []).append(callback)

    def off(self, event: str, callback: Callable):
        """Unregister an event listener."""
        cbs = self._listeners.get(event)
        if cbs:
            self._listeners[event] = [cb for cb in cbs if cb is not callback]

    async def ensure_connected(self, cdp_url: str | None = None) -> bool:
        """Reconnect if the connection is stale."""
        if self._ws and not self._closed:
            try:
                await self.send("Browser.getVersion", timeout=5)
                return True
            except Exception:
                await self.close()
        return await self.connect(cdp_url)

    async def create_page(self) -> "CDPPage":
        """Create a new browser tab and return a CDPPage handle."""
        result = await self.send(
            "Target.createTarget",
            {"url": "about:blank", "newWindow": False, "background": True},
        )
        target_id = result["targetId"]
        attach = await self.send(
            "Target.attachToTarget", {"targetId": target_id, "flatten": True}
        )
        page = CDPPage(self, target_id, attach["sessionId"])
        await page._init()
        return page

    async def close(self):
        """Shut down the connection and cancel pending work."""
        self._closed = True
        if self._reader_task:
            self._reader_task.cancel()
            try:
                await self._reader_task
            except (asyncio.CancelledError, Exception):
                pass
            self._reader_task = None
        if self._ws:
            try:
                await self._ws.close()
            except Exception:
                pass
            self._ws = None
        for mid, fut in list(self._pending.items()):
            if not fut.done():
                fut.cancel()
        self._pending.clear()


class CDPPage:
    """A browser tab managed through CDP.

    Wraps a target ID + session ID with Playwright-like convenience methods
    (goto, evaluate, screenshot, etc.) that translate to CDP protocol commands.
    """

    def __init__(self, session: CDPSession, target_id: str, session_id: str):
        self._session = session
        self._target_id = target_id
        self._session_id = session_id
        self._url = "about:blank"
        self._nav_listener = None

    async def _init(self):
        """Enable page-level CDP events and register navigation + lifecycle tracking."""
        # Track main frame URL changes
        def on_frame_navigated(params, sess_id):
            if sess_id == self._session_id:
                frame = params.get("frame", {})
                if frame.get("id") == frame.get("loaderId"):
                    self._url = frame.get("url", self._url)

        self._nav_listener = on_frame_navigated
        self._session.on("Page.frameNavigated", on_frame_navigated)

        # Lifecycle events: resolve futures per event name
        self._lifecycle_futures: dict[str, asyncio.Future] = {}

        def on_lifecycle(params, sess_id):
            if sess_id == self._session_id:
                name = params.get("name", "")
                fut = self._lifecycle_futures.get(name)
                if fut and not fut.done():
                    fut.set_result(True)

        self._lifecycle_listener = on_lifecycle
        self._session.on("Page.lifecycleEvent", on_lifecycle)

        await self._session.send("Page.enable", session_id=self._session_id)
        await self._session.send("Page.setLifecycleEventsEnabled",
                                 {"enabled": True}, session_id=self._session_id)
        await self._session.send("Runtime.enable", session_id=self._session_id)

    # ── Navigation ────────────────────────────────────────────────

    async def goto(
        self, url: str, wait_until: str = "load", timeout: float = 30,
        referrer: str = "",
    ) -> dict:
        """Navigate to a URL.

        Args:
            wait_until: ``"commit"`` (return immediately after navigation starts),
                        ``"domcontentloaded"`` (wait for DOM),
                        ``"load"`` (wait for full page load),
                        ``"networkidle"`` (wait for network to be idle).
            referrer: Optional HTTP Referer header (helps anti-bot).
        """
        params = {"url": url}
        if referrer:
            params["referrer"] = referrer
        result = await self._session.send(
            "Page.navigate", params, session_id=self._session_id, timeout=timeout
        )
        self._url = url

        error = result.get("errorText")
        if error:
            raise Exception(f"Navigation error: {error}")

        await self.wait_for_load_state(wait_until, timeout)
        return result

    async def _wait_lifecycle(self, event_name: str, timeout: float):
        """Wait for a *lifecycleEvent* (event-driven, no polling)."""
        # Clear any stale future from a prior navigation
        old = self._lifecycle_futures.pop(event_name, None)
        if old and not old.done():
            old.cancel()

        loop = asyncio.get_running_loop()
        fut = loop.create_future()
        self._lifecycle_futures[event_name] = fut
        try:
            await asyncio.wait_for(fut, timeout=timeout)
        except asyncio.TimeoutError:
            self._lifecycle_futures.pop(event_name, None)
        except asyncio.CancelledError:
            self._lifecycle_futures.pop(event_name, None)
            raise

    async def wait_for_load_state(
        self, state: str = "domcontentloaded", timeout: float = 30
    ):
        """Wait for a page load milestone via CDP events (no polling).

        Supports ``"commit"``, ``"domcontentloaded"``, ``"load"``, and ``"networkidle"``.
        ``"networkidle"`` uses ``networkAlmostIdle`` from lifecycle events.
        """
        LIFECYCLE_MAP = {
            "commit": "commit",
            "domcontentloaded": "DOMContentLoaded",
            "load": "load",
            "networkidle": "networkAlmostIdle",
        }
        name = LIFECYCLE_MAP.get(state)
        if name:
            await self._wait_lifecycle(name, timeout)
        else:
            raise ValueError(f"Unknown load state: {state}")

    # ── Input (keyboard / mouse) ──────────────────────────────────

    async def click(self, selector: str, button: str = "left",
                    click_count: int = 1) -> bool:
        """Click an element identified by *selector* via CDP Input domain.

        Uses real browser-level mouse events (not JS .click()).
        Returns True if the element was found and clicked.
        """
        box = await self.evaluate(f"""(s => {{
            const el = document.querySelector(s);
            if (!el) return null;
            const r = el.getBoundingClientRect();
            return {{x: r.left + r.width / 2, y: r.top + r.height / 2}};
        }})({json.dumps(selector)})""")
        if not box:
            return False
        await self._session.send(
            "Input.dispatchMouseEvent",
            {"type": "mousePressed", "x": box["x"], "y": box["y"],
             "button": button, "clickCount": click_count},
            session_id=self._session_id, timeout=10,
        )
        await self._session.send(
            "Input.dispatchMouseEvent",
            {"type": "mouseReleased", "x": box["x"], "y": box["y"],
             "button": button, "clickCount": click_count},
            session_id=self._session_id, timeout=10,
        )
        return True

    async def type_text(self, text: str, selector: str | None = None,
                        delay_ms: float = 20) -> bool:
        """Type *text* into an element via CDP Input domain.

        If *selector* is given, focuses the element first.
        Uses ``Input.insertText`` for reliable character insertion.
        ``delay_ms`` is preserved for API compatibility (unused — insertText is instant).
        """
        if selector:
            await self.evaluate(f"""(s => {{
                const el = document.querySelector(s);
                if (el) el.focus();
            }})({json.dumps(selector)})""")
            await asyncio.sleep(0.05)

        if text:
            await self._session.send(
                "Input.insertText", {"text": text},
                session_id=self._session_id, timeout=10,
            )
        return True

    async def press_key(self, key: str, code: str = ""):
        """Press and release a single key via CDP Input domain.

        *key* is the DOM key value (e.g., ``"Enter"``, ``"Escape"``).
        """
        if not code:
            code = key
        await self._session.send(
            "Input.dispatchKeyEvent",
            {"type": "keyDown", "key": key, "code": code},
            session_id=self._session_id, timeout=10,
        )
        await self._session.send(
            "Input.dispatchKeyEvent",
            {"type": "keyUp", "key": key, "code": code},
            session_id=self._session_id, timeout=10,
        )

    # ── Network utilities ─────────────────────────────────────────

    async def cookies(self, urls: list[str] | None = None) -> list[dict]:
        """Return all browser cookies (optional URL filter)."""
        params = {}
        if urls:
            params["urls"] = urls
        result = await self._session.send(
            "Network.getCookies", params,
            session_id=self._session_id, timeout=10,
        )
        return result.get("cookies", [])

    # ── JavaScript ─────────────────────────────────────────────────

    async def evaluate(self, expression: str) -> Any:
        """Evaluate JavaScript in the page context and return the result value.

        Function definitions (``() => {{ ... }}``, ``function() {{ ... }}``)
        are automatically wrapped in an IIFE so they execute immediately.
        """
        expr = expression.strip()
        # Auto-wrap function definitions in IIFE so they execute
        if self._is_function_def(expr):
            expr = f"({expr})()"

        result = await self._session.send(
            "Runtime.evaluate",
            {"expression": expr, "returnByValue": True, "awaitPromise": True},
            session_id=self._session_id,
            timeout=30,
        )
        if "exceptionDetails" in result:
            raise Exception(
                f"JS error: {result['exceptionDetails']['text']}; "
                f"{result['exceptionDetails'].get('exception', {}).get('description', '')}"
            )
        r = result.get("result", {})
        if r.get("type") in ("undefined", "function"):
            return None
        return r.get("value")

    @staticmethod
    def _is_function_def(expr: str) -> bool:
        """Heuristic: does *expr* look like a function/arrow definition needing IIFE wrap?"""
        if len(expr) < 6:
            return False
        # Already an IIFE or immediately invoked
        if expr.startswith("((") or expr.startswith("(async"):
            return False
        # Arrow functions: () =>, (a) =>, async () =>, async (a) =>
        # Matches both block-body () => { ... } and expression-body () => expr
        import re
        if re.match(
            r"^(async\s+)?(\([^)]*\)|\w+)\s*=>",
            expr,
        ):
            return True
        # function keyword at start: function foo() {
        if re.match(r"^function\s+\w*\s*\(", expr):
            return True
        return False

    # ── Init script ────────────────────────────────────────────────

    async def add_init_script(self, script: str):
        """Register a script that runs on every new document in this page."""
        await self._session.send(
            "Page.addScriptToEvaluateOnNewDocument",
            {"source": script},
            session_id=self._session_id,
        )

    # ── Metadata ───────────────────────────────────────────────────

    @property
    def url(self) -> str:
        return self._url

    async def title(self) -> str:
        return (await self.evaluate("document.title")) or ""

    async def call_function(self, func: str, *args: Any) -> Any:
        """Call a JavaScript function with JSON-serializable arguments.

        Maps to Playwright's ``page.evaluate(js, arg)`` pattern.
        Arguments are serialized inline into the expression string.

        Example: ``await page.call_function('(a, b) => a + b', 5, 3)``
        """
        serialized = ", ".join(json.dumps(a) for a in args)
        expression = f"({func})({serialized})"
        return await self.evaluate(expression)

    async def document_html(self) -> str:
        return (await self.evaluate("document.documentElement.outerHTML")) or ""

    # ── DOM queries ────────────────────────────────────────────────

    async def query_selector(self, selector: str) -> bool:
        """Check whether an element matching *selector* exists."""
        result = await self.evaluate(
            f"!!document.querySelector({json.dumps(selector)})"
        )
        return bool(result)

    async def inner_text(self) -> str:
        return (await self.evaluate("document.body.innerText")) or ""

    # ── Screenshot & ARIA ──────────────────────────────────────────

    async def aria_snapshot(self, **kwargs) -> str:
        """Capture the ARIA accessibility tree (Page.captureSnapshot).

        Known kwargs: ``mode`` ("ai" | "snapshot"), ``depth``, ``boxes``.
        """
        params: dict[str, Any] = {}
        for k, v in kwargs.items():
            if v is not None:
                params[k] = v
        result = await self._session.send(
            "Page.captureSnapshot", params, session_id=self._session_id, timeout=15
        )
        return result.get("data", "")

    async def screenshot(self, **kwargs) -> bytes:
        """Take a screenshot (Page.captureScreenshot). Returns raw PNG/JPEG bytes.

        Common kwargs: ``format`` ("png"/"jpeg"), ``quality`` (0-100), ``fullPage``,
        ``clip`` ({x, y, width, height}), ``omitBackground``, ``caret``, ``scale``.
        """
        params: dict[str, Any] = {}
        for k, v in kwargs.items():
            if v is not None:
                params[k] = v
        result = await self._session.send(
            "Page.captureScreenshot", params, session_id=self._session_id, timeout=30
        )
        data = result.get("data")
        if not data:
            return b""
        return base64.b64decode(data)

    # ── Resource blocking ──────────────────────────────────────────

    async def set_blocked_resources(self, patterns: list[str]):
        """Block network requests matching URL patterns.

        Uses zero buffer caps (``maxResourceBufferSize=0``) so Chrome
        does not keep response bodies in memory — only event metadata
        is tracked, saving RAM on concurrent pages.
        """
        await self._session.send(
            "Network.enable",
            {"maxTotalBufferSize": 0, "maxResourceBufferSize": 0, "maxPostDataSize": 0},
            session_id=self._session_id,
        )
        await self._session.send(
            "Network.setBlockedURLs",
            {"urls": patterns},
            session_id=self._session_id,
        )

    async def disable_images(self):
        """Prevent image bytes from entering the renderer at the engine level.

        Saves RAM and bandwidth on text-only fetches by telling Chrome
        to skip decoding all image types. Call before ``goto()``.
        """
        await self._session.send(
            "Emulation.setDisabledImageTypes",
            {"imageTypes": ["avif", "webp"]},
            session_id=self._session_id,
        )

    # ── Security & behavior ──────────────────────────────────────

    async def set_download_behavior(self, behavior: str = "deny"):
        """Prevent stray file downloads from consuming disk/RAM."""
        await self._session.send(
            "Browser.setDownloadBehavior", {"behavior": behavior},
        )

    async def ignore_certificate_errors(self, ignore: bool = True):
        """Ignore TLS certificate errors (for CDP fallback on HTTPS sites)."""
        await self._session.send(
            "Security.setIgnoreCertificateErrors", {"ignore": ignore},
        )

    # ── JS safety ─────────────────────────────────────────────────

    async def terminate_execution(self):
        """Kill running JS in the page.

        Use when a hung script blocks page interaction or extraction.
        """
        try:
            await self._session.send(
                "Runtime.terminateExecution", {},
                session_id=self._session_id, timeout=5,
            )
        except Exception:
            pass

    # ── Diagnostics ──────────────────────────────────────────────

    async def dom_counters(self) -> dict:
        """Return DOM node counters (documents, nodes, jsEventListeners)."""
        try:
            return (await self._session.send(
                "Memory.getDOMCounters", {},
                session_id=self._session_id, timeout=5,
            )).get("result", {})
        except Exception:
            return {"documents": 0, "nodes": 0, "jsEventListeners": 0}

    async def get_ax_tree(self, depth: int = 5) -> list[dict]:
        """Return the AX tree via **Accessibility.getFullAXTree**.

        Fallback when ``Page.captureSnapshot`` ``mode="ai"`` output
        is not usable. Enables/disables Accessibility per-call.
        """
        try:
            await self._session.send(
                "Accessibility.enable", {},
                session_id=self._session_id, timeout=5,
            )
            result = await self._session.send(
                "Accessibility.getFullAXTree", {"max_depth": depth},
                session_id=self._session_id, timeout=10,
            )
            return result.get("nodes", [])
        finally:
            await self._session.send(
                "Accessibility.disable", {},
                session_id=self._session_id, timeout=5,
            )

    # ── Close ──────────────────────────────────────────────────────

    async def close(self):
        """Close the tab."""
        if self._nav_listener:
            self._session.off("Page.frameNavigated", self._nav_listener)
        try:
            await self._session.send(
                "Target.closeTarget",
                {"targetId": self._target_id},
                timeout=5,
            )
        except Exception:
            pass


# ── Singleton management ──────────────────────────────────────────

_cdp_session: CDPSession | None = None
_cdp_lock = asyncio.Lock()


async def get_cdp_session(cdp_url: str | None = None) -> CDPSession | None:
    """Get or create the shared CDP session to Helium."""
    global _cdp_session
    if _cdp_session:
        ok = await _cdp_session.ensure_connected(cdp_url)
        if ok:
            return _cdp_session
        _cdp_session = None

    async with _cdp_lock:
        if _cdp_session:
            return _cdp_session
        session = CDPSession()
        ok = await session.connect(cdp_url)
        if not ok:
            logger.warning("Could not connect to Helium CDP")
            return None
        _cdp_session = session
        logger.info("CDP session established")
        return _cdp_session


async def close_cdp():
    """Shut down the shared CDP session."""
    global _cdp_session
    if _cdp_session:
        await _cdp_session.close()
        _cdp_session = None
