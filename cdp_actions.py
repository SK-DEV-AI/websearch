"""Browser action tool — expose the CDP/Helium browser capabilities over MCP.

The CDP client (``cdp_client.py``) already implements every browser
primitive this module needs (navigate, click, type_text, press_key,
evaluate, document_html, inner_text, AX tree, screenshot) — none of them
were reachable from MCP before.

Tab lifecycle (the rule that keeps this repo leak-free):
- Every tool call acquires a page through the SAME acquisition path the
  existing browser tools use (``search_gai._get_optimized_page``) — a hidden
  background tab, registered in the CDP owned-target registry.
- The tab is closed in a ``finally`` block (plus the usual orphan sweep),
  so a call can never leak a tab. Each call therefore starts from a clean
  page — no cross-call state leaks between agents.
- Multi-step flows stay in ONE tab by passing ``steps``: an ordered program
  (focus → type → press Enter → read the result) executed on the same page
  before it is closed.

SSRF discipline:
- Every URL goes through ``security.validate_url`` (public http/https only,
  private IP / DNS-rebinding blocked) BEFORE the browser is told to fetch it.
- The browser follows redirects itself, so the FINAL url is re-validated
  after navigation; a landing on a blocked address aborts the step and the
  page content is never returned. (Residual risk: the browser has already
  issued the internal request by then — same class as the existing CDP
  fetch/screenshot paths, not a new hole.)

Failure style: advisory only. Every failure returns
``{"success": false, "error": "..."}`` — the server never crashes and no
step raises out of the tool.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
from typing import Any

from cdp_client import CDPPage
from config import HELIUM_CDP
# _format_ax_nodes lives in screenshot.py — imported (not copied) so the AX
# tree rendering stays single-sourced across the tree.
from screenshot import _format_ax_nodes
from search_gai import _cleanup_orphan_tabs, _get_optimized_page
from security import SecurityError, validate_url

logger = logging.getLogger("cdp_actions")

# ── Surface ────────────────────────────────────────────────────────

ACTIONS = (
    "navigate",   # url            — go to a URL (SSRF-validated)
    "click",      # selector       — real mouse click on an element
    "type",       # text+selector  — real text insertion (focus first if selector given)
    "key",        # key            — press a key (Enter, Escape, Tab, ...)
    "evaluate",   # expression     — run JS in the page, return the value
    "html",       #                — page HTML
    "text",       # selector?      — inner text of the page or of one element
    "aria",       # depth?         — accessibility tree snapshot
    "screenshot", #                — PNG/JPEG written to a temp file, path returned
    "wait",       # wait_state     — wait for a load milestone
)

LOAD_STATES = ("domcontentloaded", "load", "networkidle")

MAX_STEPS = 25
MAX_STEP_TIMEOUT = 120.0
MAX_TOTAL_SECONDS = 300.0
DEFAULT_MAX_CHARS = 20000
MAX_MAX_CHARS = 200000
DEFAULT_STEP_TIMEOUT = 30.0


class _StepError(Exception):
    """A single step failed — carries the step index for the caller."""

    def __init__(self, index: int, message: str, action: str = "",
                 results: list[dict] | None = None):
        super().__init__(message)
        self.index = index
        self.message = message
        self.action = action
        self.results = results or []


# ── Step normalization + validation ────────────────────────────────

def _as_int(value: Any, default: int) -> int:
    try:
        return int(value)
    except (TypeError, ValueError):
        return default


def _normalize_step(raw: dict) -> dict:
    """Coerce one step (or the single-action call) into the internal shape."""
    text = raw.get("text")
    if text is None:
        text = ""
    elif not isinstance(text, str):
        text = str(text)
    return {
        "action": str(raw.get("action", "") or "").strip().lower(),
        "url": str(raw.get("url", "") or "").strip(),
        "selector": str(raw.get("selector", "") or "").strip(),
        "text": text,
        "key": str(raw.get("key", "") or "").strip(),
        "expression": str(raw.get("expression", "") or "").strip(),
        "wait_state": str(raw.get("wait_state", "") or "domcontentloaded").strip().lower(),
        "depth": max(1, min(100, _as_int(raw.get("depth", 5), 5))),
    }


def _validate_step(step: dict, index: int) -> None:
    """Raise ValueError with a human-readable reason when a step is malformed."""
    action = step["action"]
    if action not in ACTIONS:
        raise ValueError(
            f"unknown action {action!r} — expected one of {', '.join(ACTIONS)}")
    if action == "navigate" and not step["url"]:
        raise ValueError("action=navigate requires a url")
    if action == "click" and not step["selector"]:
        raise ValueError("action=click requires a selector")
    if action == "type" and not step["text"]:
        raise ValueError("action=type requires text")
    if action == "key" and not step["key"]:
        raise ValueError("action=key requires a key")
    if action == "evaluate" and not step["expression"]:
        raise ValueError("action=evaluate requires an expression")
    if step["wait_state"] not in LOAD_STATES:
        raise ValueError(
            f"unknown wait_state {step['wait_state']!r} — expected one of "
            f"{', '.join(LOAD_STATES)}")


# ── Navigation (SSRF-validated) ────────────────────────────────────

async def _navigate(page: CDPPage, url: str, wait_state: str,
                    timeout: float) -> None:
    """Validate + navigate, then re-validate the landing URL.

    The browser resolves redirects itself, so the address the page actually
    ended up on is checked against ``validate_url`` before any content is
    handed back to the caller.
    """
    try:
        await validate_url(url)
    except SecurityError as e:
        raise ValueError(f"URL rejected by SSRF validation: {e}") from e

    await page.goto(url, wait_until="commit", timeout=timeout)
    if wait_state and wait_state != "commit":
        await page.wait_for_load_state(wait_state, timeout=min(timeout, 20.0))
    await asyncio.sleep(0.2)

    final = page.url
    if final and final not in ("about:blank", "") and not final.startswith("data:"):
        try:
            await validate_url(final)
        except SecurityError as e:
            raise ValueError(
                f"navigation landed on a blocked URL ({final}) — "
                f"SSRF validation failed: {e}") from e


# ── Per-action execution ───────────────────────────────────────────

async def _run_step(page: CDPPage, step: dict, *, timeout: float,
                    max_chars: int, full_page: bool, image_type: str,
                    quality: int | None) -> dict:
    """Execute one action on *page*. Raises on failure."""
    action = step["action"]

    if action == "navigate":
        await _navigate(page, step["url"], step["wait_state"], timeout)
        return {"action": "navigate", "url": page.url,
                "requested_url": step["url"]}
    if step["url"]:
        # every other action may navigate first (fresh page per call)
        await _navigate(page, step["url"], step["wait_state"], timeout)

    if action == "click":
        sel = step["selector"]
        clicked = await page.click(sel)
        if not clicked:
            raise ValueError(f"click target not found: {sel}")
        return {"action": "click", "selector": sel, "clicked": True,
                "url": page.url}

    if action == "type":
        sel = step["selector"]
        if sel and not await page.query_selector(sel):
            raise ValueError(f"type target not found: {sel}")
        await page.type_text(step["text"], selector=sel or None)
        return {"action": "type", "selector": sel, "chars": len(step["text"]),
                "url": page.url}

    if action == "key":
        await page.press_key(step["key"])
        return {"action": "key", "key": step["key"], "url": page.url}

    if action == "evaluate":
        value = await page.evaluate(step["expression"])
        # Cap the payload: an expression can return an arbitrarily large
        # structure (unbounded read == memory-DoS class, same as the fetch
        # tier's streaming-cap problem). Oversized results come back as a
        # truncated string instead of the raw object.
        try:
            encoded = json.dumps(value, default=str)
        except (TypeError, ValueError):
            encoded = str(value)
        if len(encoded) > max_chars:
            return {"action": "evaluate", "result": encoded[:max_chars],
                    "result_chars": len(encoded), "truncated": True,
                    "url": page.url}
        return {"action": "evaluate", "result": value, "url": page.url}

    if action == "html":
        html = await page.document_html()
        total = len(html)
        return {"action": "html", "html": html[:max_chars], "html_chars": total,
                "truncated": total > max_chars, "url": page.url}

    if action == "text":
        sel = step["selector"]
        if sel:
            value = await page.call_function(
                "(s) => { const e = document.querySelector(s);"
                " if (!e) return null;"
                " const t = e.innerText != null && e.innerText !== ''"
                "   ? e.innerText : (e.textContent || '');"
                " return t; }", sel)
            found = value is not None
            text = value or ""
        else:
            text = await page.inner_text()
            found = True
        total = len(text)
        return {"action": "text", "selector": sel, "found": found,
                "text": text[:max_chars], "chars": total,
                "truncated": total > max_chars, "url": page.url}

    if action == "aria":
        nodes = await page.get_ax_tree(depth=step["depth"])
        if not nodes:
            nodes = await page.get_ax_tree(depth=100)
        lines = _format_ax_nodes(nodes) if nodes else []
        snap = "\n".join(lines)
        total = len(snap)
        return {"action": "aria", "snapshot": snap[:max_chars], "chars": total,
                "nodes": len(nodes), "truncated": total > max_chars,
                "url": page.url}

    if action == "screenshot":
        fmt = image_type if image_type in ("png", "jpeg") else "png"
        kwargs: dict[str, Any] = {"captureBeyondViewport": bool(full_page),
                                  "format": fmt}
        if fmt == "jpeg" and quality is not None:
            kwargs["quality"] = max(0, min(100, _as_int(quality, 80)))
        data = await page.screenshot(**kwargs)
        if not data:
            raise ValueError("no screenshot captured")
        fd, path = tempfile.mkstemp(prefix="browser_action_",
                                    suffix=f".{fmt}",
                                    dir=tempfile.gettempdir())
        try:
            with os.fdopen(fd, "wb") as f:
                f.write(data)
        except Exception:
            try:
                os.unlink(path)
            except OSError:
                pass
            raise
        return {"action": "screenshot", "path": path, "bytes": len(data),
                "image_type": fmt, "url": page.url}

    if action == "wait":
        await page.wait_for_load_state(step["wait_state"], timeout=timeout)
        return {"action": "wait", "wait_state": step["wait_state"],
                "url": page.url}

    raise ValueError(f"unknown action {action!r}")


async def _run_program(page: CDPPage, program: list[dict], *,
                       step_timeout: float, max_chars: int, full_page: bool,
                       image_type: str, quality: int | None) -> list[dict]:
    """Run the ordered program on one page; abort on the first failure."""
    results: list[dict] = []
    for index, step in enumerate(program):
        try:
            payload = await asyncio.wait_for(
                _run_step(page, step, timeout=step_timeout, max_chars=max_chars,
                          full_page=full_page, image_type=image_type,
                          quality=quality),
                timeout=step_timeout + 5.0)
        except asyncio.TimeoutError as e:
            raise _StepError(
                index,
                f"step {index} ({step['action']}) timed out after "
                f"{step_timeout:.0f}s",
                step["action"], results) from e
        except _StepError:
            raise
        except Exception as e:
            raise _StepError(
                index,
                f"step {index} ({step['action']}) failed: "
                f"{type(e).__name__}: {e}",
                step["action"], results) from e
        results.append(payload)
    return results


# ── Tool entry point ───────────────────────────────────────────────

async def browser_action(
    action: str = "",
    url: str = "",
    selector: str = "",
    text: str = "",
    key: str = "",
    expression: str = "",
    wait_state: str = "domcontentloaded",
    steps: list | None = None,
    timeout: float = DEFAULT_STEP_TIMEOUT,
    max_chars: int = DEFAULT_MAX_CHARS,
    full_page: bool = True,
    image_type: str = "png",
    quality: int | None = None,
    depth: int = 5,
) -> dict:
    """Drive a real browser page through the local Helium CDP endpoint.

    Returns ``{"success": True, ...}`` or ``{"success": False, "error": ...}``
    — never raises.
    """
    # ── Build the program (single action, or an ordered multi-step one) ──
    if steps is not None:
        if not isinstance(steps, list):
            return {"success": False,
                    "error": "steps must be a list of action objects"}
        if not steps:
            return {"success": False, "error": "steps must not be empty"}
        if len(steps) > MAX_STEPS:
            return {"success": False,
                    "error": f"too many steps (max {MAX_STEPS})"}
        raw_steps = []
        for i, s in enumerate(steps):
            if not isinstance(s, dict):
                return {"success": False,
                        "error": f"steps[{i}] must be an object"}
            raw_steps.append(s)
    else:
        if not str(action or "").strip():
            return {"success": False,
                    "error": f"action is required — one of {', '.join(ACTIONS)} "
                              "(or pass steps[] for a multi-step flow)"}
        raw_steps = [{
            "action": action, "url": url, "selector": selector, "text": text,
            "key": key, "expression": expression, "wait_state": wait_state,
            "depth": depth,
        }]

    program: list[dict] = []
    for i, raw in enumerate(raw_steps):
        step = _normalize_step(raw)
        try:
            _validate_step(step, i)
        except ValueError as e:
            return {"success": False, "error": f"step {i}: {e}",
                    "step_index": i}
        program.append(step)

    # ── Budgets ──
    try:
        step_timeout = float(timeout)
    except (TypeError, ValueError):
        step_timeout = DEFAULT_STEP_TIMEOUT
    step_timeout = max(1.0, min(MAX_STEP_TIMEOUT, step_timeout))
    total_budget = min(MAX_TOTAL_SECONDS, step_timeout * len(program))
    max_chars = max(200, min(MAX_MAX_CHARS, _as_int(max_chars,
                                                     DEFAULT_MAX_CHARS)))

    # ── Acquire the page (same path as the existing browser tools) ──
    try:
        page = await _get_optimized_page(block_resources=False)
    except ConnectionError:
        return {"success": False,
                "error": f"Helium CDP connection failed at {HELIUM_CDP} — "
                          f"is the browser running with --remote-debugging-port=9222?"}
    except Exception as e:
        return {"success": False,
                "error": f"Helium CDP connection failed at {HELIUM_CDP}: "
                          f"{type(e).__name__}: {e}"}

    results: list[dict] = []
    title = ""
    error: dict | None = None
    try:
        results = await asyncio.wait_for(
            _run_program(page, program, step_timeout=step_timeout,
                         max_chars=max_chars, full_page=full_page,
                         image_type=image_type, quality=quality),
            timeout=total_budget)
        try:
            title = await page.title()
        except Exception:
            title = ""
    except asyncio.TimeoutError:
        error = {"success": False,
                 "error": f"browser_action timed out after {total_budget:.0f}s"}
    except _StepError as e:
        error = {"success": False, "error": e.message, "step_index": e.index,
                 "action": e.action}
        if e.results:
            error["completed_steps"] = e.results
    except Exception as e:
        error = {"success": False, "error": f"{type(e).__name__}: {e}"}
    finally:
        # The tab is ALWAYS closed here — no leaks, ever.
        try:
            await page.close()
        except Exception:
            pass
        asyncio.ensure_future(_cleanup_orphan_tabs())

    if error:
        return error

    if len(results) == 1:
        return {"success": True, "title": title, **results[0]}
    return {"success": True, "title": title, "steps": results,
            "final": results[-1]}
