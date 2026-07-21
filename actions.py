"""Page interaction actions for CDP fetches.

Enables agents to specify click/type/press/wait/scroll sequences to
reach content behind interactive elements (click-to-load, forms,
infinite scroll, paginated content).

Each action is a single-key dict:
    {"click": "selector"}       — click element via CDP Input domain
    {"type": "some text"}       — type into currently focused element
    {"fill": {"selector": "input#q", "text": "query"}} — fill form field
    {"press": "Enter"}          — press a key
    {"wait": 500}              — wait in ms
    {"scroll": 3}              — scroll N viewports (or {"amount": 500} pixels)
    {"wait_selector": ".res"}  — wait for element in DOM
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any

from cdp_client import CDPPage

logger = logging.getLogger("actions")


async def run_actions(page: CDPPage, actions: list[dict],
                      timeout: float = 30) -> str | None:
    """Run a sequence of page interactions, return page text after all complete.

    Returns None if any action fails irrecoverably.
    """
    if not actions:
        return None

    deadline = time.monotonic() + timeout

    for i, action in enumerate(actions):
        if time.monotonic() > deadline:
            logger.warning("actions timed out at step %d/%d", i, len(actions))
            break

        try:
            if "click" in action:
                sel = action["click"]
                if not await page.click(sel):
                    logger.warning("click target not found: %s", sel)
                await asyncio.sleep(0.3)

            elif "type" in action:
                await page.type_text(str(action["type"]))
                await asyncio.sleep(0.1)

            elif "fill" in action:
                f = action["fill"]
                if isinstance(f, dict):
                    await page.type_text(
                        f.get("text", ""),
                        selector=f.get("selector", ""),
                    )
                await asyncio.sleep(0.2)

            elif "press" in action:
                await page.press_key(str(action["press"]))
                await asyncio.sleep(0.2)

            elif "wait" in action:
                ms = int(action["wait"])
                await asyncio.sleep(min(ms / 1000, timeout))

            elif "scroll" in action:
                amt = action["scroll"]
                if isinstance(amt, dict):
                    px = amt.get("amount", 500)
                else:
                    px = int(amt) * 800
                await page.evaluate(f"window.scrollBy(0, {px})")
                await asyncio.sleep(0.3)

            elif "wait_selector" in action:
                sel = action["wait_selector"]
                rem = deadline - time.monotonic()
                if rem > 0:
                    await _wait_selector(page, sel, min(rem, 15))

        except Exception as e:
            logger.warning("action %d failed: %s", i, e)

    try:
        text = await page.inner_text()
        if text and text.strip():
            return text.strip()
    except Exception:
        pass
    return None


async def _wait_selector(page: CDPPage, selector: str, timeout: float = 10):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if await page.query_selector(selector):
            return True
        await asyncio.sleep(0.1)
    return False
