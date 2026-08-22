"""Shared reranker bridge — one subprocess via Unix socket, both MCP servers connect to it."""

import asyncio
import fcntl
import json
import logging
import os
import signal
import subprocess

logger = logging.getLogger(__name__)

_SOCKET_PATH = "/tmp/reranker_worker.sock"
_RERANKER_PYTHON = "/usr/bin/python3"
_RERANKER_WORKER = os.path.expanduser("~/.local/share/reranker-rust/worker.py")
_RERANKER_KILLSWITCH = os.path.expanduser("~/.local/share/reranker-rust/disabled")


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


async def _close_connection():
    """Close the current connection and kill the stale worker process."""
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
    if killswitch_active():
        await _close_connection()  # free VRAM if worker still running
        return fallback_sort(passages, top_k)
    async with _LOCK:
        global _READER, _WRITER, _PROC
        if not await _is_healthy() and not await _ensure_worker():
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
            return fallback_sort(passages, top_k)
        try:
            r = await asyncio.wait_for(_READER.readuntil(b"\n"), timeout=120)
        except (asyncio.IncompleteReadError, ConnectionResetError, asyncio.TimeoutError) as e:
            logger.warning(f"reranker: read failed: {e}")
            await _close_connection()
            return fallback_sort(passages, top_k)
        try:
            result = json.loads(r)
        except json.JSONDecodeError:
            logger.warning("reranker: invalid JSON response from worker")
            await _close_connection()
            return fallback_sort(passages, top_k)
        if result.get("error"):
            logger.warning(f"reranker error: {result['error']}")
            return fallback_sort(passages, top_k)
        scored = result.get("scores", [])
        for s in scored:
            if "score" in s:
                s["_rerank"] = s.pop("score")
        return scored
