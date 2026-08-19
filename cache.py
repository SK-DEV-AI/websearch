"""SQLite content cache for fetch results.

WAL mode, keyed by URL+extraction_type+css_selector, TTL eviction, size cap.
Stores full extracted content so paginated re-reads are instant.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import sqlite3
import time
from pathlib import Path
from typing import Any

import aiosqlite

logger = logging.getLogger("cache")

CACHE_DIR = Path.home() / ".cache" / "websearch"
DB_NAME = "fetch_cache.db"
DEFAULT_TTL = 3600  # 1 hour
MAX_ENTRIES = 10000

_db_initialized: dict[Path, bool] = {}
_db_init_lock = asyncio.Lock()


def _cache_key(url: str, extraction_type: str = "markdown",
               css_selector: str | None = None) -> str:
    raw = f"{url}|{extraction_type}|{css_selector or ''}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


async def _ensure_db() -> Path:
    d = CACHE_DIR
    d.mkdir(parents=True, exist_ok=True)
    db_path = d / DB_NAME

    if _db_initialized.get(db_path):
        return db_path

    async with _db_init_lock:
        if _db_initialized.get(db_path):
            return db_path

        async with aiosqlite.connect(db_path) as db:
            await db.execute("PRAGMA journal_mode=WAL")
            await db.execute("PRAGMA busy_timeout=5000")
            await db.execute("""
                CREATE TABLE IF NOT EXISTS fetch_cache (
                    key TEXT PRIMARY KEY,
                    url TEXT NOT NULL,
                    content TEXT NOT NULL,
                    status INTEGER NOT NULL DEFAULT 0,
                    content_type TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    metadata TEXT NOT NULL DEFAULT '{}',
                    total_chars INTEGER NOT NULL DEFAULT 0,
                    fetched_at REAL NOT NULL,
                    ttl INTEGER NOT NULL DEFAULT 3600
                )
            """)
            await db.execute("CREATE INDEX IF NOT EXISTS idx_fc_fetched_at ON fetch_cache(fetched_at)")
            await db.commit()

        _db_initialized[db_path] = True
        return db_path


async def get_cached(url: str, extraction_type: str = "markdown",
                     css_selector: str | None = None,
                     ttl: int = DEFAULT_TTL) -> dict | None:
    """Return cached fetch result if fresh, else None."""
    key = _cache_key(url, extraction_type, css_selector)
    db_path = await _ensure_db()

    async with aiosqlite.connect(db_path) as db:
        db.row_factory = aiosqlite.Row
        cursor = await db.execute(
            "SELECT * FROM fetch_cache WHERE key = ? AND fetched_at + MIN(ttl, ?) > ?",
            (key, ttl, time.time()),
        )
        row = await cursor.fetchone()
        if row is None:
            return None
        return {
            "content": row["content"],
            "status": row["status"],
            "content_type": row["content_type"],
            "title": row["title"],
            "metadata": json.loads(row["metadata"]),
            "total_chars": row["total_chars"],
            "cached": True,
        }


async def set_cached(url: str, content: str,
                     extraction_type: str = "markdown",
                     css_selector: str | None = None,
                     status: int = 200, content_type: str = "",
                     title: str = "", metadata: dict | None = None,
                     ttl: int = DEFAULT_TTL) -> None:
    """Store a fetch result in cache."""
    key = _cache_key(url, extraction_type, css_selector)
    db_path = await _ensure_db()

    async with aiosqlite.connect(db_path) as db:
        await db.execute(
            """INSERT OR REPLACE INTO fetch_cache
               (key, url, content, status, content_type, title, metadata,
                total_chars, fetched_at, ttl)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (key, url, content, status, content_type, title,
             json.dumps(metadata or {}), len(content),
             time.time(), ttl),
        )
        count_cursor = await db.execute("SELECT COUNT(*) FROM fetch_cache")
        (count,) = await count_cursor.fetchone()
        if count > MAX_ENTRIES:
            excess = count - int(MAX_ENTRIES * 0.9)
            await db.execute(
                "DELETE FROM fetch_cache WHERE key IN "
                "(SELECT key FROM fetch_cache ORDER BY fetched_at ASC LIMIT ?)",
                (excess,),
            )
        await db.commit()


async def clear_cache(url: str = "", prefix: str = "") -> dict:
    """E4a: explicit cache invalidation for fast-moving topics.

    - url="" and prefix="" → wipe the whole fetch cache
    - prefix="foo" → delete every entry whose URL starts with foo
    - url="https://x/y" → delete that exact URL (all extraction variants)
    Returns {"deleted": N, "prefix": ...}.
    """
    db_path = await _ensure_db()
    async with aiosqlite.connect(db_path) as db:
        if url:
            cur = await db.execute(
                "SELECT key FROM fetch_cache WHERE url = ?", (url,))
        elif prefix:
            cur = await db.execute(
                "SELECT key FROM fetch_cache WHERE url LIKE ?",
                (prefix.replace("%", "%%") + "%",))
        else:
            cur = await db.execute("SELECT COUNT(*) FROM fetch_cache")
            row = await cur.fetchone()
            total = row[0] if row else 0
            await db.execute("DELETE FROM fetch_cache")
            await db.commit()
            return {"deleted": total, "scope": "all"}
        rows = await cur.fetchall()
        keys = [r[0] for r in rows]
        if keys:
            await db.executemany("DELETE FROM fetch_cache WHERE key = ?",
                                 [(k,) for k in keys])
        await db.commit()
        return {"deleted": len(keys), "scope": "prefix" if prefix else "url"}
