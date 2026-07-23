"""Shared reranker bridge — one subprocess via Unix socket, both MCP servers connect to it."""

import asyncio
import json
import logging
import os
import signal

logger = logging.getLogger(__name__)

_SOCKET_PATH = "/tmp/reranker_worker.sock"
_RERANKER_PYTHON = "/usr/bin/python3"
_RERANKER_WORKER = os.path.expanduser("~/.local/share/reranker-rust/worker.py")

_PROC = None
_LOCK = asyncio.Lock()
_READER = None
_WRITER = None


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
                    await asyncio.wait_for(_PROC.wait(), timeout=3)
                except asyncio.TimeoutError:
                    _PROC.kill()
                    await _PROC.wait()
        except Exception:
            pass
        _PROC = None


async def _ensure_worker():
    """Spawn the reranker worker process and connect to its socket.

    Always removes the stale socket and spawns a fresh worker — never tries
    to reuse a dead peer connection. Tests the connection with a ping after
    connecting.
    """
    global _PROC, _READER, _WRITER
    # Kill any stale process and close old connection
    await _close_connection()
    # Remove stale socket so new worker can bind
    try:
        os.unlink(_SOCKET_PATH)
    except OSError:
        pass
    try:
        _PROC = await asyncio.create_subprocess_exec(
            _RERANKER_PYTHON, _RERANKER_WORKER,
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.DEVNULL,
        )
        for _ in range(50):
            try:
                _READER, _WRITER = await asyncio.open_unix_connection(_SOCKET_PATH)
                return True
            except (FileNotFoundError, ConnectionRefusedError, OSError):
                await asyncio.sleep(0.1)
        return False
    except Exception as e:
        logger.warning(f"reranker: start failed: {e}")
        return False


async def _is_healthy() -> bool:
    """Quick ping to check if the worker socket is alive."""
    global _PROC, _WRITER, _READER
    if _PROC is None or _PROC.returncode is not None or _WRITER is None or _READER is None:
        return False
    # Return early if we can't reach the socket
    try:
        req = json.dumps({"query": "ping", "passages": [{"snippet": ""}], "top_k": 1})
        _WRITER.write((req + "\n").encode())
        await asyncio.wait_for(_WRITER.drain(), timeout=2)
        r = await asyncio.wait_for(_READER.readuntil(b"\n"), timeout=5)
        return not json.loads(r).get("error")
    except Exception:
        return False


async def warmup() -> bool:
    async with _LOCK:
        global _READER, _WRITER, _PROC
        if await _is_healthy():
            return True
        if not await _ensure_worker():
            return False
        try:
            req = json.dumps({"query": "warmup", "passages": [{"snippet": "warmup"}], "top_k": 1})
            _WRITER.write((req + "\n").encode())
            await asyncio.wait_for(_WRITER.drain(), timeout=5)
            r = await asyncio.wait_for(_READER.readuntil(b"\n"), timeout=120)
            result = json.loads(r)
            return not result.get("error")
        except Exception:
            await _close_connection()
            return False


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
    async with _LOCK:
        global _READER, _WRITER, _PROC
        if not await _is_healthy() and not await _ensure_worker():
            return fallback_sort(passages, top_k)
        normalized = []
        for p in passages:
            item = dict(p)
            text = item.get("snippet") or item.get("text") or item.get("content") or ""
            item["snippet"] = text[:3072]
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
            r = await asyncio.wait_for(_READER.readuntil(b"\n"), timeout=30)
        except (asyncio.IncompleteReadError, ConnectionResetError, asyncio.TimeoutError) as e:
            logger.warning(f"reranker: read failed: {e}")
            await _close_connection()
            return fallback_sort(passages, top_k)
        try:
            result = json.loads(r)
        except json.JSONDecodeError:
            return fallback_sort(passages, top_k)
    if result.get("error"):
        logger.warning(f"reranker error: {result['error']}")
        return fallback_sort(passages, top_k)
    scored = result.get("scores", [])
    for s in scored:
        if "score" in s:
            s["_rerank"] = s.pop("score")
    return scored
