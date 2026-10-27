"""Shared reranker bridge — one worker subprocess via Unix socket, shared by
both MCP servers.

Single-worker rule: a flock-guarded spawn + ownership-scoped cleanup guarantee
no double model-load and no mutual kills — whoever spawns owns _PROC; a
borrower (connected to the sibling's worker) never kills or unlinks."""

import asyncio
import contextlib
import contextvars
import fcntl
import json
import logging
import os
import signal
import subprocess
import time

logger = logging.getLogger(__name__)

_SOCKET_PATH = "/tmp/reranker_worker.sock"
_RERANKER_PYTHON = os.environ.get("RERANKER_PYTHON", "/usr/bin/python3")
_RERANKER_WORKER = os.environ.get(
    "RERANKER_WORKER",
    os.path.expanduser("~/.local/share/reranker-rust/worker.py"),
)
_RERANKER_KILLSWITCH = os.environ.get(
    "RERANKER_KILLSWITCH",
    os.path.expanduser("~/.local/share/reranker-rust/disabled"),
)


def killswitch_active() -> bool:
    """True when the reranker killswitch file exists — reranking is off and the
    worker must not run (frees GPU VRAM). Enable with:
    rm ~/.local/share/reranker-rust/disabled"""
    return os.path.exists(_RERANKER_KILLSWITCH)

_PROC = None
_LOCK = asyncio.Lock()
_READER = None
_WRITER = None
_SPAWN_COOLDOWN = 120  # s: after a failed spawn (e.g. OOM-killed model load),
# don't retry on every search — each attempt loads ~500MB into RAM before dying
_last_spawn_fail = 0.0

# Read timeout and cooldown (2026-10-27 incident). Measured worker latency over
# one day: median 4 s, p90 56 s, max 119 s — under CPU contention (docling OCR
# in-process, swarm agents, 1GB free RAM) a single rerank can eat 120 s and
# blow a search's whole deadline on its own. A timeout now degrades to base
# ordering instead of stalling, and repeated timeouts skip the worker for a
# short cooldown so the next agent in the burst does not pay the same wait.
_READ_TIMEOUT = float(os.environ.get("WEBSEARCH_RERANK_TIMEOUT", "60") or 60)
_TIMEOUT_COOLDOWN = float(os.environ.get("WEBSEARCH_RERANK_COOLDOWN", "20") or 0)
_last_timeout = 0.0

# How long a call may wait for the single rerank worker before giving up and
# answering with base ordering (2026-10-27, 12-lane burst). One socket, one
# request at a time: with N concurrent searches the Nth waits for N-1
# completions, and that queue is what pushed one search to 253 s. A caller
# that cannot get the worker quickly is better off with slightly worse
# ordering than with a 4-minute search.
_QUEUE_WAIT = float(os.environ.get("WEBSEARCH_RERANK_QUEUE_WAIT", "15") or 0)


@contextlib.asynccontextmanager
async def _worker_slot():
    """Hold the single rerank worker, but never wait longer than _QUEUE_WAIT.

    Yields True when the worker is ours, False when the queue was too long.
    """
    acquired = False
    try:
        if _QUEUE_WAIT > 0:
            await asyncio.wait_for(_LOCK.acquire(), timeout=_QUEUE_WAIT)
        else:
            await _LOCK.acquire()
        acquired = True
    except asyncio.TimeoutError:
        pass
    try:
        yield acquired
    finally:
        if acquired:
            _LOCK.release()


# Per-request outcome of the last rerank() call (contextvar: concurrent
# requests must not read each other's verdict).
_STATUS: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "rerank_status", default=None)


def status() -> dict:
    """How this request's most recent rerank() call ended."""
    return dict(_STATUS.get() or {"status": "ok"})


def _mark(state: str, reason: str = "") -> None:
    _STATUS.set({"status": state, "reason": reason,
                 "timeout_s": _READ_TIMEOUT} if reason else
                {"status": state, "timeout_s": _READ_TIMEOUT})


async def _close_connection():
    """Close our connection; kill + unlink ONLY if we spawned the worker
    (_PROC is not None). Borrowers never reach the kill branch."""
    global _READER, _WRITER, _PROC
    if _READER is not None:
        try:
            _READER.feed_eof()
        except Exception:
            pass
        _READER = None
    if _WRITER is not None:
        try:
            _WRITER.close()
            await _WRITER.wait_closed()
        except Exception:
            pass
        _WRITER = None
    # Kill a stale worker that won't accept new connections
    if _PROC is not None:
        try:
            if _PROC.returncode is None:
                _PROC.send_signal(signal.SIGTERM)
                try:
                    await asyncio.wait_for(asyncio.to_thread(_PROC.wait), timeout=3)
                except asyncio.TimeoutError:
                    _PROC.kill()
                    await asyncio.to_thread(_PROC.wait)
        except Exception:
            pass
        _PROC = None
        # Best-effort remove the socket we owned — a SIGKILLed worker leaves a
        # stale file that would keep the stat-based health check green forever.
        try:
            os.unlink(_SOCKET_PATH)
        except OSError:
            pass


async def _ensure_worker():
    """Connect to the shared worker socket, spawning it only when no live
    listener exists. A flock guard prevents concurrent servers from
    double-spawning; whoever wins spawns, the rest just connect.
    """
    global _PROC, _READER, _WRITER, _last_spawn_fail
    if killswitch_active():
        return False
    # A stranger's machine has no worker at this path (it is not part of the pip
    # package). Bail out before touching the socket or taking the spawn lock, so
    # every search does not pay a lock + spawn attempt for a model that will
    # never load. Results fall back to score ordering; doctor() explains it.
    if not os.path.isfile(_RERANKER_WORKER):
        return False
    import time
    if time.monotonic() - _last_spawn_fail < _SPAWN_COOLDOWN:
        return False
    try:
        _READER, _WRITER = await asyncio.open_unix_connection(_SOCKET_PATH, limit=2**20)
        return True
    except (FileNotFoundError, ConnectionRefusedError, OSError):
        pass
    lock_fd = os.open(_SOCKET_PATH + ".lock", os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            _READER, _WRITER = await asyncio.open_unix_connection(_SOCKET_PATH, limit=2**20)
            return True
        except (FileNotFoundError, ConnectionRefusedError, OSError):
            _last_spawn_fail = time.monotonic()
            return False
    try:
        try:
            os.unlink(_SOCKET_PATH)
        except OSError:
            pass
        # reap a previous crashed worker so it doesn't linger as a zombie
        if _PROC is not None and _PROC.returncode is not None:
            try:
                _PROC.wait(timeout=1)
            except Exception:
                pass
        _PROC = subprocess.Popen(
            [_RERANKER_PYTHON, _RERANKER_WORKER],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            start_new_session=True,
            close_fds=True,
        )
        for _ in range(50):
            try:
                _READER, _WRITER = await asyncio.open_unix_connection(_SOCKET_PATH, limit=2**20)
                return True
            except (FileNotFoundError, ConnectionRefusedError, OSError):
                await asyncio.sleep(0.1)
        _last_spawn_fail = time.monotonic()
        logger.warning("reranker: worker did not come up within 5s — cooldown %ds", _SPAWN_COOLDOWN)
        return False
    except Exception as e:
        logger.warning(f"reranker: start failed: {e}")
        _last_spawn_fail = time.monotonic()
        return False
    finally:
        fcntl.flock(lock_fd, fcntl.LOCK_UN)
        os.close(lock_fd)


async def _is_healthy() -> bool:
    """Stat-based health check — no wire ping, so a busy worker (40-120s
    rerank) never stalls other agents' checks or queues stale ping lines."""
    global _PROC, _WRITER, _READER
    return (
        _PROC is not None
        and _PROC.returncode is None
        and _WRITER is not None
        and _READER is not None
        and os.path.exists(_SOCKET_PATH)
    )


def fallback_sort(passages: list[dict], top_k: int) -> list[dict]:
    """Sort passages by best available relevance score when reranker fails."""
    scored = []
    for p in passages:
        score = p.get("_rel") or p.get("_relevance") or p.get("_hybrid") or 0
        scored.append((score, p))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [p for _, p in scored[:top_k]]


async def rerank(query: str, passages: list[dict], top_k: int = 20) -> list[dict]:
    if not passages:
        return []
    global _last_timeout
    if killswitch_active():
        await _close_connection()  # free VRAM if worker still running
        _mark("disabled", "killswitch")
        return fallback_sort(passages, top_k)
    if _TIMEOUT_COOLDOWN > 0 and _last_timeout \
            and (time.monotonic() - _last_timeout) < _TIMEOUT_COOLDOWN:
        # A recent read timed out: the worker is wedged or starved. Do not make
        # every other agent wait the full timeout behind it.
        _mark("degraded", "cooldown after timeout")
        return fallback_sort(passages, top_k)
    async with _worker_slot() as got_worker:
        if not got_worker:
            logger.info("reranker: worker busy for >%.0fs, using base order",
                        _QUEUE_WAIT)
            _mark("degraded", f"queued behind rerank >{_QUEUE_WAIT:.0f}s")
            return fallback_sort(passages, top_k)
        global _READER, _WRITER, _PROC
        if not await _is_healthy() and not await _ensure_worker():
            _mark("degraded", "worker unavailable")
            return fallback_sort(passages, top_k)
        normalized = []
        for p in passages:
            item = dict(p)
            text = item.get("snippet") or item.get("text") or item.get("content") or item.get("full_content") or ""
            item["snippet"] = text[:4000]  # ~1K tokens: enough signal, ~8x less compute than 32K
            normalized.append(item)
        req = json.dumps({"query": query, "passages": normalized, "top_k": top_k})
        try:
            _WRITER.write((req + "\n").encode())
            await asyncio.wait_for(_WRITER.drain(), timeout=5)
        except (BrokenPipeError, OSError, asyncio.TimeoutError) as e:
            logger.warning(f"reranker: write failed: {e}")
            await _close_connection()
            _mark("degraded", "write failed")
            return fallback_sort(passages, top_k)
        try:
            r = await asyncio.wait_for(_READER.readuntil(b"\n"), timeout=_READ_TIMEOUT)
        except (asyncio.IncompleteReadError, ConnectionResetError, asyncio.TimeoutError) as e:
            logger.warning(f"reranker: read failed: {e}")
            if isinstance(e, asyncio.TimeoutError):
                _last_timeout = time.monotonic()
            await _close_connection()
            _mark("degraded", f"read failed: {type(e).__name__}")
            return fallback_sort(passages, top_k)
        try:
            result = json.loads(r)
        except json.JSONDecodeError:
            logger.warning("reranker: invalid JSON response from worker")
            await _close_connection()
            _mark("degraded", "invalid JSON from worker")
            return fallback_sort(passages, top_k)
        if result.get("error"):
            logger.warning(f"reranker error: {result['error']}")
            _mark("degraded", str(result["error"])[:120])
            return fallback_sort(passages, top_k)
        scored = result.get("scores", [])
        for s in scored:
            if "score" in s:
                s["_rerank"] = s.pop("score")
        _mark("ok")
        return scored
