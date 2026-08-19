"""Cloudflare Turnstile solve on the shared CDP browser (theme A1).

CDP-only implementation of the turnstile_solver (MIT) / nodriver approach,
reimplemented without AGPL code or asset coupling:

1. detect the `.cf-turnstile` widget in the parent DOM (cross-origin iframe
   is opaque — we only need its bounding box),
2. geometry-click the checkbox (left-aligned inside the box), with a short
   human-like pre-move,
3. success via a MutationObserver injected on new document (attribute change
   on the turnstile input / #challenge-success-text visibility) OR the
   appearance of a cf_clearance cookie,
4. retry until deadline, then hand the clearance cookies back.

Trust caveat (from turnstile_solver README): CDP clicks are untrusted events;
they pass mainly with a clean IP + coherent fingerprint. Failure is soft —
the caller falls through to scrapling.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time

logger = logging.getLogger("turnstile_solve")

from cdp_client import get_cdp_session
from ghost_state import CLEARANCE_PREFIXES

_OBSERVER_JS = r"""
(() => {
  const mark = () => { try { sessionStorage.setItem('turnstile_verified', 'true'); } catch (e) {} };
  try {
    // body-level observer: catches widgets rendered after DOMContentLoaded
    const bodyObs = new MutationObserver(() => {
      const inp = document.querySelector('.cf-turnstile input');
      if (inp && inp.value) mark();
    });
    bodyObs.observe(document.documentElement, {childList: true, subtree: true});
    // attribute observer once the input exists
    const check = () => {
      const inp = document.querySelector('.cf-turnstile input');
      if (inp) {
        const obs = new MutationObserver(() => { if (inp.value) mark(); });
        obs.observe(inp, {attributes: true, attributeFilter: ['name', 'value', 'checked']});
      }
      const success = document.querySelector('#challenge-success-text');
      if (success) {
        const obs = new MutationObserver(() => mark());
        obs.observe(success.parentElement || document.body, {childList: true, subtree: true});
      }
    };
    if (document.readyState === 'loading') {
      document.addEventListener('DOMContentLoaded', check);
    } else {
      check();
    }
    setInterval(() => {
      const s = document.querySelector('#challenge-success-text');
      if (s && s.getClientRects().length > 0) mark();
      const inp = document.querySelector('.cf-turnstile input');
      if (inp && inp.value) mark();
    }, 500);
  } catch (e) {}
})();
"""

# checkbox sits in the left portion of the widget box
_CLICK_X_FRAC = 0.10
_CLICK_Y_FRAC = 0.5


async def _widget_box(page) -> tuple[float, float, float, float] | None:
    """Bounding box of the .cf-turnstile widget in viewport CSS px."""
    send = page._session.send
    sid = page._session_id
    await send("DOM.enable", session_id=sid)
    doc = await send("DOM.getDocument", {"depth": 0}, session_id=sid)
    root = doc["root"]["nodeId"]
    q = await send("DOM.querySelector",
                   {"nodeId": root, "selector": ".cf-turnstile[data-sitekey]"},
                   session_id=sid)
    node_id = q.get("nodeId")
    if not node_id:
        return None
    box = await send("DOM.getBoxModel", {"nodeId": node_id}, session_id=sid)
    quad = box.get("model", {}).get("content") or []
    if len(quad) < 4:
        return None
    xs = quad[0::2]
    ys = quad[1::2]
    x, y = min(xs), min(ys)
    w, h = max(xs) - x, max(ys) - y
    if w <= 2 or h <= 2:
        return None  # not painted yet
    return x, y, w, h


async def _click_at(page, x: float, y: float):
    send = page._session.send
    sid = page._session_id
    sx, sy = max(x - 60.0, 0.0), max(y - 30.0, 0.0)
    steps = 12
    for i in range(1, steps + 1):
        t = i / steps
        jx = x + (sx - x) * (1 - t) ** 2 * 0.3
        jy = y + (sy - y) * (1 - t) ** 2 * 0.3
        await send("Input.dispatchMouseEvent",
                   {"type": "mouseMoved", "x": jx, "y": jy, "buttons": 1},
                   session_id=sid)
        await asyncio.sleep(0.012)
    for ev in ("mousePressed", "mouseReleased"):
        await send("Input.dispatchMouseEvent",
                   {"type": ev, "x": x, "y": y, "button": "left",
                    "buttons": 1 if ev == "mousePressed" else 0,
                    "clickCount": 1},
                   session_id=sid)


async def _is_verified(page, url: str) -> bool:
    try:
        v = await page.evaluate(
            "sessionStorage.getItem('turnstile_verified') === 'true'")
        if v:
            return True
    except Exception:
        pass
    try:
        cks = await page.cookies([url])
        for c in cks or []:
            if str(c.get("name", "")).startswith("cf_clearance"):
                return True
    except Exception:
        pass
    return False


async def turnstile_solve(url: str, timeout: float = 35.0) -> list[dict] | None:
    """Solve a Cloudflare Turnstile wall on *url* via the shared CDP browser.

    Returns clearance cookies (filtered) on success, None if no Turnstile
    wall was found or the solve failed. Creates one background tab, closes
    it on exit — never touches other tabs, never opens a window.
    """
    deadline = time.monotonic() + timeout
    page = None
    try:
        session = await get_cdp_session()
        if not session:
            return None
        page = await session.create_page()
        await page.add_init_script(_OBSERVER_JS)
        await page.goto(url, wait_until="load", timeout=30)
        await asyncio.sleep(2.5)  # Cloudflare injects the widget async

        box = await _widget_box(page)
        if box is None:
            logger.info("turnstile_solve: no .cf-turnstile widget on %s", url)
            return None

        clicked = 0
        while time.monotonic() < deadline:
            box = await _widget_box(page) or box
            cx = box[0] + box[2] * _CLICK_X_FRAC
            cy = box[1] + box[3] * _CLICK_Y_FRAC
            await _click_at(page, cx, cy)
            clicked += 1
            poll_until = min(time.monotonic() + 5.0, deadline)
            while time.monotonic() < poll_until:
                if await _is_verified(page, url):
                    cks = await page.cookies([url])
                    filt = [c for c in (cks or [])
                            if str(c.get("name", "")).startswith(CLEARANCE_PREFIXES)]
                    logger.info("turnstile_solve: solved %s (%d click%s)",
                                url, clicked, "s" if clicked != 1 else "")
                    return filt or cks or None
                await asyncio.sleep(1.0)
        logger.info("turnstile_solve: timeout on %s (%d clicks)", url, clicked)
        return None
    except Exception as e:
        logger.info("turnstile_solve: failed on %s: %s", url, e)
        return None
    finally:
        if page is not None:
            try:
                await page.close()
            except Exception:
                pass


async def _selfcheck():
    """Reachability probe: connect, create tab, detect widget on a known
    Turnstile demo page, report detection without clicking."""
    session = await get_cdp_session()
    if not session:
        print("no CDP session")
        return
    page = await session.create_page()
    try:
        await page.add_init_script(_OBSERVER_JS)
        await page.goto("https://turnstile-demo.pages.dev", wait_until="load", timeout=30)
        await asyncio.sleep(3)
        print("widget box:", await _widget_box(page))
        print("cf page reached:", page.url.startswith("https://challenges.cloudflare.com"))
    finally:
        await page.close()


if __name__ == "__main__":
    asyncio.run(_selfcheck())
