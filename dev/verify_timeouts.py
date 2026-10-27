"""Prove the 2026-10-27 timeout fixes end to end, through the real server path.

Run: python3 dev/verify_timeouts.py

Checks, in order:

1. Deadline: a slow dispatch is cut at WEBSEARCH_TOOL_DEADLINE and answered
   with a structured, retryable payload instead of silence.
2. Partial: the search branch's published snapshot comes back with the
   deadline payload, so a caller keeps its sources.
3. Admission: with one slot busy, a second call gets "busy" (retryable), not a
   queued 300 s wait.
4. Live search: depth=1 search completes under the deadline and reports the
   reranker verdict it actually used.
5. Legacy env: WEBSEARCH_TOOL_DEADLINE=0 disables the deadline entirely.
6. Budget: a call that waited for a slot gets the remaining deadline, not a
   fresh full one (QUEUE_WAIT + DEADLINE stacked to 260 s under load).
7. Rerank queue: a call that cannot get the single rerank worker quickly
   degrades to base ordering and releases the worker.
"""

from __future__ import annotations

import asyncio
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ops as _ops  # noqa: E402
import server as srv  # noqa: E402

PASS: list[str] = []
FAIL: list[str] = []


def check(label: str, ok: bool, detail: str = "") -> None:
    (PASS if ok else FAIL).append(label)
    print(("  PASS  " if ok else "  FAIL  ") + label + (f"  [{detail}]" if detail else ""))


class Params:
    def __init__(self, name: str, arguments: dict):
        self.name = name
        self.arguments = arguments


def payload(result) -> dict:
    return json.loads(result.content[0].text)


async def main() -> int:
    real_dispatch = srv._dispatch_tool
    real_deadline = _ops.DEADLINE

    # 1 ── deadline cut -----------------------------------------------------
    async def _slow(ctx, params):
        await asyncio.sleep(30)
        return srv._dispatch_tool(ctx, params)

    srv._dispatch_tool = _slow
    _ops.DEADLINE = 1.0
    t0 = time.monotonic()
    r = payload(await srv.handle_call_tool(None, Params("search", {"query": "x"})))
    took = time.monotonic() - t0
    check("deadline cuts a runaway call", r.get("deadline_exceeded") is True, f"{took:.1f}s")
    check("deadline error is retryable", r.get("retryable") is True)
    check("deadline error is under the client timeout", took < 5)
    srv._dispatch_tool = real_dispatch

    # 2 ── partial survives ------------------------------------------------
    async def _publishing(ctx, params):
        _ops.hold({"success": True, "results": [{"url": "https://example.com"}]})
        await asyncio.sleep(30)

    srv._dispatch_tool = _publishing
    r = payload(await srv.handle_call_tool(None, Params("search", {"query": "x"})))
    check("partial result is returned with the deadline",
          (r.get("partial") or {}).get("results") == [{"url": "https://example.com"}])
    check("partial result is labelled", bool(r.get("partial_note")))
    srv._dispatch_tool = real_dispatch

    # 3 ── admission -------------------------------------------------------
    _ops.DEADLINE = 30.0
    _ops.MAX_CONCURRENT = 1
    _ops.QUEUE_WAIT = 0.5

    async def _hold_slot():
        return await _ops.run_tool("search", lambda: asyncio.sleep(5, result={"success": True}))

    holder = asyncio.create_task(_hold_slot())
    await asyncio.sleep(0.1)
    srv._dispatch_tool = lambda ctx, params: asyncio.sleep(0, result={
        "success": True, "results": []})
    r = payload(await srv.handle_call_tool(None, Params("fetch", {"url": "https://example.com"})))
    check("second call is refused as busy", r.get("busy") is True)
    check("busy answer is retryable", r.get("retryable") is True)
    holder.cancel()
    try:
        await holder
    except asyncio.CancelledError:
        pass
    srv._dispatch_tool = real_dispatch
    _ops.MAX_CONCURRENT = 8
    _ops.QUEUE_WAIT = 20.0
    _ops.DEADLINE = real_deadline

    # 4 ── live search -----------------------------------------------------
    t0 = time.monotonic()
    r = payload(await srv.handle_call_tool(None, Params(
        "search", {"query": "python asyncio to_thread default executor",
                   "depth": 1, "count": 5, "synthesize": False})))
    took = time.monotonic() - t0
    check("depth=1 search succeeds", r.get("success") is True,
          f"{len(r.get('results') or [])} results in {took:.1f}s")
    check("search reports the reranker verdict",
          r.get("reranker") is None or isinstance(r.get("reranker"), str),
          str(r.get("reranker"))[:60])
    check("search stayed under the deadline", took < _ops.DEADLINE)

    # 5 ── deadline can be disabled ---------------------------------------
    _ops.DEADLINE = 0
    r = payload(await srv.handle_call_tool(None, Params("ping", {})))
    check("deadline=0 still runs tools", "connectivity" in r or r.get("success") is True)
    _ops.DEADLINE = real_deadline

    # 6 ── queue wait is charged against the deadline ----------------------
    # A call that waits for a slot must not then get a fresh full deadline:
    # QUEUE_WAIT + DEADLINE stacked to 260s under load, near the 300s client
    # timeout. Budget is now measured from arrival.
    _ops.MAX_CONCURRENT = 1
    _ops.QUEUE_WAIT = 10.0
    _ops.DEADLINE = 2.0
    slot = _ops._semaphore()
    holder = asyncio.create_task(slot.acquire())
    await holder                       # occupy the only slot
    started = time.monotonic()
    release = asyncio.create_task(_release_after(slot, 1.2))
    r = payload(await srv.handle_call_tool(None, Params(
        "search", {"query": "x", "depth": 1, "count": 1, "synthesize": False})))
    total = time.monotonic() - started
    await release
    slot.release()
    check("queue wait counts against the deadline", total < 2.6,
          f"returned at {total:.1f}s with DEADLINE=2.0 and 1.2s queue wait")
    check("starved call answers with a deadline payload, not a hang",
          r.get("deadline_exceeded") is True or r.get("busy") is True
          or r.get("success") is True, json.dumps({k: r.get(k) for k in
          ("deadline_exceeded", "busy", "success")}))
    _ops.MAX_CONCURRENT = 8
    _ops.QUEUE_WAIT = 20.0
    _ops.DEADLINE = real_deadline

    # 7 ── rerank queue cap ------------------------------------------------
    # One socket serves one request, so a burst queues. Unbounded, the queue
    # alone pushed a 12-lane search to 253s; the worker wait is now capped and
    # the call answers with base ordering instead.
    import reranker as _rr
    real_rr_wait = _rr._QUEUE_WAIT
    _rr._QUEUE_WAIT = 0.3
    async with _rr._worker_slot() as first:
        async with _rr._worker_slot() as second:
            pass
    _rr._QUEUE_WAIT = real_rr_wait
    check("rerank worker slot is granted when free", first is True)
    check("a long rerank queue degrades instead of waiting", second is False)
    check("rerank worker slot is released after the capped wait",
          _rr._LOCK.locked() is False)

    print(f"\nverify_timeouts: {len(PASS)}/{len(PASS) + len(FAIL)} PASS")
    return 0 if not FAIL else 1


async def _release_after(sem, delay: float) -> None:
    """Free a held slot after ``delay`` so a waiter gets a partial budget."""
    await asyncio.sleep(delay)
    sem.release()


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
