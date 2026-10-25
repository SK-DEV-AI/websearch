"""Named browser sessions — let browser state survive across tool calls.

The problem this solves: `browser_action` opens one fresh tab per call and
closes it afterwards. That is correct for a single read, but it makes any
interactive workflow impossible inside a tool call, so agents wrote scripts
instead: log in, click through, fill a form, follow pagination. Each of those
was a throwaway script.

A session is a named, long-lived tab. While it is open:

  * cookies, localStorage and login state persist
  * scroll position and open dialogs persist
  * several `browser_action(session_id=...)` calls act on the same page

Lifecycle rules, deliberately conservative:

  * Sessions are opt-in. No session_id means the old behaviour exactly:
    fresh tab, closed at the end of the call. Nothing changes for existing
    callers.
  * A session is capped at MAX_SESSIONS, oldest-idle evicted first, so a
    runaway caller cannot accumulate tabs.
  * Sessions are owned by this process. On restart they are gone; `get` will
    try to re-attach to the recorded target id once, then fail honestly.
  * `close` always runs the CDP close, so tabs do not leak.

Nothing here is used unless a caller passes a session_id or calls
`browser_session`.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import time
import uuid
from typing import Any

from cdp_client import CDPSession, CDPPage, get_cdp_session, _mark_owned

logger = logging.getLogger(__name__)

MAX_SESSIONS = 8
IDLE_CLOSE_SECONDS = 60 * 30          # a session unused for 30 min is evicted
_REGISTRY_PATH = os.path.expanduser("~/.cache/websearch/sessions.json")

# session_id -> {"page": CDPPage, "target_id": str, "created_at": float,
#                "last_used": float, "url": str, "label": str}
_SESSIONS: dict[str, dict[str, Any]] = {}
_LOCK = asyncio.Lock()


# ───────────────────────── persistence ─────────────────────────


def _load_registry() -> dict:
    try:
        with open(_REGISTRY_PATH, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_registry() -> None:
    """Best-effort metadata mirror so `list` can explain sessions after a
    restart. Never fatal: losing this file only loses history, not state."""
    try:
        os.makedirs(os.path.dirname(_REGISTRY_PATH), exist_ok=True)
        disk = {
            sid: {k: v for k, v in rec.items() if k != "page"}
            for sid, rec in _SESSIONS.items()
        }
        tmp = _REGISTRY_PATH + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(disk, fh, indent=1)
        os.replace(tmp, _REGISTRY_PATH)
    except Exception as e:  # noqa: BLE001
        logger.debug("session registry save failed: %s", e)


# ───────────────────────── helpers ─────────────────────────


def _new_id(label: str) -> str:
    base = "".join(c for c in (label or "").lower() if c.isalnum())[:16]
    return f"{base or 'sess'}-{uuid.uuid4().hex[:8]}"


async def _evict_if_needed() -> None:
    """Keep the session count bounded: drop the least recently used."""
    while len(_SESSIONS) >= MAX_SESSIONS:
        victim = min(_SESSIONS.items(), key=lambda kv: kv[1].get("last_used", 0))
        logger.info("session: evicting %s (cap %d)", victim[0], MAX_SESSIONS)
        await _close_one(victim[0])


async def _close_one(sid: str) -> bool:
    rec = _SESSIONS.pop(sid, None)
    if not rec:
        return False
    page = rec.get("page")
    if page is not None:
        try:
            await page.close()
        except Exception as e:  # noqa: BLE001
            logger.debug("session %s close failed: %s", sid, e)
    _save_registry()
    return True


async def _reattach(target_id: str) -> CDPPage | None:
    """Try to re-attach to a tab id recorded before a restart."""
    session: CDPSession | None = await get_cdp_session()
    if not session:
        return None
    try:
        res = await session.send(
            "Target.attachToTarget", {"targetId": target_id, "flatten": True}
        )
        page = CDPPage(session, target_id, res["sessionId"])
        await page._init()
        _mark_owned(target_id)
        return page
    except Exception as e:  # noqa: BLE001
        logger.debug("session reattach failed for %s: %s", target_id, e)
        return None


# ───────────────────────── public API ─────────────────────────


async def create(*, label: str = "", url: str = "") -> dict:
    """Open a named, persistent tab."""
    async with _LOCK:
        await _evict_if_needed()

        session: CDPSession | None = await get_cdp_session()
        if not session:
            return {
                "success": False,
                "error": (
                    "No CDP browser is reachable. Sessions need a Chromium-based "
                    "browser started with --remote-debugging-port on this host."
                ),
                "fix": "chromium --remote-debugging-port=9222 "
                       '--user-data-dir="$HOME/.mcp-browser"',
                "call": "doctor()",
            }
        try:
            page = await session.create_page()
        except Exception as e:  # noqa: BLE001
            return {"success": False,
                    "error": f"could not open a tab: {type(e).__name__}: {e}"}

        if url:
            try:
                await page.goto(url)
            except Exception as e:  # noqa: BLE001
                # Keep the session: the caller may still want to navigate later.
                logger.debug("session initial goto failed: %s", e)

        sid = _new_id(label)
        now = time.time()
        _SESSIONS[sid] = {
            "page": page,
            "target_id": getattr(page, "target_id", ""),
            "created_at": now,
            "last_used": now,
            "url": url,
            "label": label or sid,
        }
        _save_registry()
        return {
            "success": True,
            "session_id": sid,
            "label": label or sid,
            "url": url or "about:blank",
            "tabs": 1,
            "note": "Pass session_id to browser_action to keep using this tab.",
        }


async def get_page(session_id: str) -> tuple[CDPPage | None, dict | None]:
    """Return (page, error). Touches last_used on success."""
    sid = (session_id or "").strip()
    if not sid:
        return None, {"success": False, "error": "session_id is required"}

    rec = _SESSIONS.get(sid)
    if rec and rec.get("page") is not None:
        # Cheap liveness check: a closed tab raises on evaluate.
        try:
            await rec["page"].evaluate("1+1")
        except Exception:
            logger.info("session %s tab is gone; will try reattach", sid)
            rec["page"] = None
        else:
            rec["last_used"] = time.time()
            return rec["page"], None

    if not rec:
        disk = _load_registry()
        if sid not in disk:
            return None, {
                "success": False,
                "error": f"unknown session_id {sid!r}",
                "fix": "Call browser_session(action='create') first, "
                       "or browser_session(action='list') to see live sessions.",
            }
        rec = _SESSIONS.setdefault(sid, {**disk[sid], "page": None})

    page = await _reattach(rec.get("target_id", ""))
    if page is None:
        _SESSIONS.pop(sid, None)
        _save_registry()
        return None, {
            "success": False,
            "error": (
                f"session {sid!r} existed but its tab is no longer available "
                "(the browser restarted or the tab was closed)."
            ),
            "fix": "Call browser_session(action='create') to start a new session.",
        }
    rec["page"] = page
    rec["last_used"] = time.time()
    _save_registry()
    return page, None


async def info(session_id: str) -> dict:
    rec = _SESSIONS.get(session_id) or _load_registry().get(session_id)
    if not rec:
        return {"success": False, "error": f"unknown session_id {session_id!r}"}
    out = {
        "success": True,
        "session_id": session_id,
        "label": rec.get("label", ""),
        "url": rec.get("url", ""),
        "live": rec.get("page") is not None,
        "age_seconds": int(time.time() - rec.get("created_at", time.time())),
        "idle_seconds": int(time.time() - rec.get("last_used", time.time())),
    }
    page = (_SESSIONS.get(session_id) or {}).get("page")
    if page is not None:
        try:
            out["current_url"] = page.url
        except Exception:
            pass
    return out


async def list_all() -> dict:
    live = []
    for sid, rec in _SESSIONS.items():
        item = {
            "session_id": sid,
            "label": rec.get("label", ""),
            "url": rec.get("url", ""),
            "age_seconds": int(time.time() - rec.get("created_at", time.time())),
            "idle_seconds": int(time.time() - rec.get("last_used", time.time())),
        }
        page = rec.get("page")
        if page is not None:
            try:
                item["current_url"] = page.url
            except Exception:
                pass
        live.append(item)
    stale = [s for s in _load_registry() if s not in _SESSIONS]
    return {
        "success": True,
        "count": len(live),
        "sessions": live,
        "stale_on_disk": stale,
        "cap": MAX_SESSIONS,
    }


async def close(session_id: str) -> dict:
    sid = (session_id or "").strip()
    if sid:
        closed = await _close_one(sid)
        return ({"success": True, "closed": sid} if closed
                else {"success": False, "error": f"unknown session_id {sid!r}"})

    n = 0
    for s in list(_SESSIONS):
        n += 1 if await _close_one(s) else 0
    try:
        os.remove(_REGISTRY_PATH)
    except Exception:
        pass
    return {"success": True, "closed_count": n}


async def shutdown() -> int:
    """Close every tab on server exit so nothing leaks."""
    return (await close("")).get("closed_count", 0)


def active_count() -> int:
    return len(_SESSIONS)


# ───────────────────────── selfcheck ─────────────────────────


async def _selfcheck() -> None:
    checks: list[tuple[str, bool]] = []
    checks.append(("new id is unique", _new_id("") != _new_id("")))
    checks.append(("new id carries label", _new_id("login").startswith("login-")))
    checks.append(("registry path is under home", _REGISTRY_PATH.startswith(os.path.expanduser("~"))))
    checks.append(("empty session_id rejected", (await get_page(""))[1] is not None))
    checks.append(("unknown session rejected", (await get_page("nope-123"))[1] is not None))
    for name, okv in checks:
        print(f"{'OK' if okv else 'FAIL'}  {name}")
    print(f"session selfcheck: {sum(1 for _, v in checks if v)}/{len(checks)} OK")


if __name__ == "__main__":
    asyncio.run(_selfcheck())
