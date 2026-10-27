"""Request admission control: per-call deadline, concurrency bound, partials.

Why this module exists (2026-10-27 incident, proven from logs):

* The jcode MCP client cuts every tool call at 300 s and reports a bare
  ``Request timeout after 300s`` with no payload. Today one process served
  314 tool calls with median 2.7 s, p90 123 s, max 225 s, and 33 hard
  client timeouts. The tail is what kills swarms: several agents searching
  at once queue behind one another's rerank/fetch/synthesis work, and every
  call that crosses 300 s is wasted work that keeps burning CPU, which
  slows the next call. Restarting the server "fixed" it only by dropping
  that in-flight work.
* So a request must (a) be bounded by a deadline we control, below the
  client's, (b) not multiply without limit when many agents call at once,
  and (c) return whatever it already gathered instead of nothing.

Env knobs (all optional):

  WEBSEARCH_TOOL_DEADLINE   seconds a single tool call may run (default 240,
                            0 disables). Keep it under the client timeout.
  WEBSEARCH_MAX_CONCURRENT  simultaneous tool calls allowed (default 8,
                            0 = unlimited).
  WEBSEARCH_QUEUE_WAIT      seconds a call may wait for a slot before it is
                            answered "busy" instead (default 20).

The partial payload is opt-in per request: a long tool publishes its
in-progress result with :func:`hold` as soon as it has something usable, and
a deadline hit returns that instead of an empty failure.
"""

from __future__ import annotations

import asyncio
import contextvars
import json
import logging
import os
import time

logger = logging.getLogger("ops")


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, "") or default)
    except (TypeError, ValueError):
        return default


DEADLINE = _env_float("WEBSEARCH_TOOL_DEADLINE", 240.0)
MAX_CONCURRENT = _env_int("WEBSEARCH_MAX_CONCURRENT", 8)
QUEUE_WAIT = _env_float("WEBSEARCH_QUEUE_WAIT", 20.0)
SLOW_CALL_S = _env_float("WEBSEARCH_SLOW_CALL_S", 30.0)

_partial: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "websearch_partial", default=None)

_slots: asyncio.Semaphore | None = None
_slots_loop: object = None
_slots_cap = -1
_in_flight = 0


def hold(payload: dict) -> None:
    """Publish the in-progress result of this request (same dict object)."""
    _partial.set(payload)


def partial() -> dict | None:
    return _partial.get()


def _semaphore() -> asyncio.Semaphore | None:
    """One semaphore per event loop, rebuilt if the limit changes.

    The server reuses a single loop, so this normally builds once; rebuilding
    on a capacity change keeps the knob honest when MAX_CONCURRENT is set
    after import (tests, or a config reload).
    """
    global _slots, _slots_loop, _slots_cap
    if MAX_CONCURRENT <= 0:
        return None
    loop = asyncio.get_running_loop()
    if _slots is None or _slots_loop is not loop or _slots_cap != MAX_CONCURRENT:
        _slots = asyncio.Semaphore(MAX_CONCURRENT)
        _slots_loop = loop
        _slots_cap = MAX_CONCURRENT
    return _slots


def busy_payload(name: str, waited: float) -> dict:
    return {
        "success": False,
        "error": (
            f"Server busy: {_in_flight} tool calls already running "
            f"(WEBSEARCH_MAX_CONCURRENT={MAX_CONCURRENT}); {name} waited "
            f"{waited:.0f}s for a slot. Retry shortly, or raise the limit."
        ),
        "busy": True,
        "in_flight": _in_flight,
        "waited_s": round(waited, 1),
        "retryable": True,
    }


def deadline_payload(name: str, elapsed: float, held: dict | None) -> dict:
    out: dict = {
        "success": False,
        "error": (
            f"{name} hit the server deadline after {elapsed:.0f}s "
            f"(WEBSEARCH_TOOL_DEADLINE={DEADLINE:.0f}s). The client would have "
            f"timed out anyway; narrow the request (lower depth/count, "
            f"synthesize=false) or raise the deadline."
        ),
        "deadline_exceeded": True,
        "deadline_s": DEADLINE,
        "elapsed_s": round(elapsed, 1),
        "retryable": True,
    }
    if held:
        # Whatever the request already collected is worth more than nothing:
        # the caller can cite these sources instead of retrying blind.
        out["partial"] = held
        out["partial_note"] = (
            "Partial result captured before the deadline: the stages that "
            "finished are present, the rest were abandoned.")
    return out


async def run_tool(name: str, factory) -> dict | None:
    """Run ``factory()`` under the concurrency bound and the call deadline.

    Returns the tool payload dict, or a busy/deadline payload dict describing
    why nothing ran. Returns ``None`` only when ``DEADLINE <= 0`` and the tool
    raised nothing (caller then handles its own result). ``factory`` must be a
    zero-arg callable returning an awaitable, so the coroutine is only created
    once a slot exists (waiting on a semaphore must not start the work).
    """
    global _in_flight
    name = str(name)
    t0 = time.monotonic()
    sem = _semaphore()
    if sem is None:
        return await _call(name, factory, t0)
    waited = 0.0
    try:
        if QUEUE_WAIT > 0:
            await asyncio.wait_for(sem.acquire(), timeout=QUEUE_WAIT)
        else:
            await sem.acquire()
    except asyncio.TimeoutError:
        return busy_payload(name, time.monotonic() - t0)
    waited = time.monotonic() - t0
    _in_flight += 1
    try:
        return await _call(name, factory, t0)
    finally:
        _in_flight -= 1
        sem.release()
        if waited > 5:
            logger.info("%s waited %.1fs for a free slot", name, waited)


async def _call(name: str, factory, t0: float) -> dict | None:
    """Run ``factory()`` with the budget left after admission, not a full one.

    The deadline must cover the whole call as the client sees it, so queue wait
    counts against it. Charging the full deadline after a queue wait let one
    call run QUEUE_WAIT + DEADLINE (20 + 240 = 260s, measured under 12-lane
    load) and approach the 300s client timeout this module exists to avoid.
    ``factory`` is called here rather than by the caller so a call that has no
    budget left never builds a coroutine it would then abandon unawaited.
    """
    if DEADLINE <= 0:
        return await factory()
    budget = DEADLINE - (time.monotonic() - t0)
    if budget <= 0:
        elapsed = time.monotonic() - t0
        payload = deadline_payload(name, elapsed, partial())
        payload["queue_wait_exhausted"] = True
        logger.warning("DEADLINE %s before it started after %.0fs (partial=%s)",
                       name, elapsed, bool(payload.get("partial")))
        return payload
    try:
        return await asyncio.wait_for(factory(), timeout=budget)
    except asyncio.TimeoutError:
        elapsed = time.monotonic() - t0
        payload = deadline_payload(name, elapsed, partial())
        # stdout is the MCP wire; keep the trace on stderr where the host
        # collects it as "MCP [websearch] stderr".
        logger.warning("DEADLINE %s after %.0fs (partial=%s)", name, elapsed,
                       bool(payload.get("partial")))
        return payload
    finally:
        _partial.set(None)


def note_slow(name: str, elapsed: float) -> None:
    """Log a slow but successful call — the tail is what causes timeouts."""
    if SLOW_CALL_S > 0 and elapsed >= SLOW_CALL_S:
        logger.warning("SLOW %s took %.0fs", name, elapsed)


def selfcheck() -> None:
    """Offline checks of the payload shapes and the deadline wrapper."""
    import asyncio as _a

    global DEADLINE, MAX_CONCURRENT, QUEUE_WAIT
    orig_deadline, orig_max, orig_wait = DEADLINE, MAX_CONCURRENT, QUEUE_WAIT

    async def _fast() -> dict:
        return {"success": True}

    async def _slow() -> dict:
        await _a.sleep(5)
        return {"success": True}

    async def _publishing() -> dict:
        d = {"success": True, "results": [1, 2, 3]}
        hold(d)
        await _a.sleep(5)
        return d

    async def _boom() -> dict:
        raise RuntimeError("boom")

    try:
        DEADLINE = 0.2
        # A finished tool passes through untouched.
        assert _a.run(run_tool("quick", _fast)) == {"success": True}

        # A deadline hit returns a structured, retryable error, not silence.
        p = _a.run(run_tool("search", _slow))
        assert p["success"] is False and p["deadline_exceeded"] is True, p
        assert p["retryable"] is True and "deadline" in p["error"], p
        assert "partial" not in p, p

        # Held progress survives the deadline and is returned as partial.
        p = _a.run(run_tool("search", _publishing))
        assert p["partial"] == {"success": True, "results": [1, 2, 3]}, p
        assert p["partial_note"], p

        # Exceptions from the tool propagate (the dispatcher formats them).
        try:
            _a.run(run_tool("boom", _boom))
            raise AssertionError("exception should propagate")
        except RuntimeError:
            pass

        # A partial from one request must not leak into the next one.
        assert partial() is None

        # Admission control: with one slot, a second caller gets "busy", not a
        # second unbounded run, once its queue wait expires.
        DEADLINE = 5.0
        MAX_CONCURRENT = 1
        QUEUE_WAIT = 0.2

        async def _pair():
            hold_task = _a.create_task(run_tool("search", _slow))
            await _a.sleep(0.05)
            second = await run_tool("fetch", _fast)
            hold_task.cancel()
            try:
                await hold_task
            except _a.CancelledError:
                pass
            return second

        second = _a.run(_pair())
        assert second["busy"] is True, second
        assert second["retryable"] is True and second["in_flight"] == 1, second

        # Unlimited mode never reports busy.
        MAX_CONCURRENT = 0
        assert _a.run(run_tool("fetch", _fast)) == {"success": True}
    finally:
        DEADLINE, MAX_CONCURRENT, QUEUE_WAIT = orig_deadline, orig_max, orig_wait
        _partial.set(None)


if __name__ == "__main__":  # pragma: no cover - manual selfcheck entry
    selfcheck()
    print(json.dumps({"selfcheck": "ok", "checks": 9}))
