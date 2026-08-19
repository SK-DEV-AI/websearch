"""Per-tier fetch success statistics (ghost-tier-learning, theme A).

SQLite mirror of GhostState's in-profile tier_stats: durable across restarts,
queryable by a reporter. The in-profile dict stays the fast sync routing
source; this table is best-effort durability (never awaited on the sync path).
"""

from __future__ import annotations

import time

import aiosqlite

from cache import CACHE_DIR, DB_NAME, _ensure_db, _db_initialized, _db_init_lock

_TABLE = """
    CREATE TABLE IF NOT EXISTS tier_stats (
        host TEXT, tier TEXT, attempts INTEGER, ok INTEGER, fail INTEGER,
        updated_at REAL, PRIMARY KEY(host, tier)
    )
"""


async def _ensure_tier_db():
    db_path = await _ensure_db()
    if not _db_initialized.get(db_path):
        async with _db_init_lock:
            if not _db_initialized.get(db_path):
                db_path = await _ensure_db()
    async with aiosqlite.connect(db_path) as db:
        await db.execute(_TABLE)
        await db.commit()
    return db_path


async def record_tier_result(host: str, tier: str, ok: bool):
    try:
        db_path = await _ensure_tier_db()
        async with aiosqlite.connect(db_path) as db:
            await db.execute("""
                INSERT INTO tier_stats (host, tier, attempts, ok, fail, updated_at)
                VALUES (?, ?, 1, ?, ?, ?)
                ON CONFLICT(host, tier) DO UPDATE SET
                    attempts = attempts + 1,
                    ok = ok + excluded.ok,
                    fail = fail + excluded.fail,
                    updated_at = excluded.updated_at
            """, (host, tier, 1 if ok else 0, 0 if ok else 1, time.time()))
            await db.commit()
    except Exception:
        pass


async def tier_success_rate(host: str, tier: str) -> float:
    try:
        db_path = await _ensure_tier_db()
        async with aiosqlite.connect(db_path) as db:
            cur = await db.execute(
                "SELECT attempts, ok FROM tier_stats WHERE host=? AND tier=?",
                (host, tier))
            row = await cur.fetchone()
            if not row or row[0] == 0:
                return 1.0
            return row[1] / row[0]
    except Exception:
        return 1.0
