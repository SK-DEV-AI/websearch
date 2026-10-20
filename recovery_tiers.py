"""Jina Reader pre-CDP tier + Common Crawl supplement + CDX history (proposals 4/6).

- jina_fetch: keyless https://r.jina.ai/<url> after direct fetch fails, BEFORE
  the 12s CDP escalation. Cuts browser load on soft walls (20 RPM keyless).
- cc_lookup: index.commoncrawl.org domain-survey assist for map_site +
  dead-link content recovery (WARC range fetch, first record only).
- cdx_history: web.archive.org CDX multi-timestamp recovery — the existing
  _try_wayback/_try_archive_today cover closest-only; this returns alternate
  snapshots (timestamp, url, digest) for dead entries before the 0.5x demote.

All advisory: failures return None/[], never raise. SSRF validation mirrors
fast_paths (derived snapshot URLs re-validated). robots.txt is NOT consulted
here — the caller (fetch_url/enrich) owns the politeness gate.
"""

from __future__ import annotations

import asyncio
import gzip
import json
import logging
import time
import urllib.parse

from config import get_http_client
from security import safe_fetch_capped, validate_url as _validate_url

logger = logging.getLogger("recovery_tiers")

_TIMEOUT = 15.0
_JINA_RPM_GUARD_S = 3.2  # keyless 20 RPM: local spacing, best-effort only
_last_jina_ts = 0.0

# Byte caps for the capped fetches below. HTML/markdown text gets the
# 2 MiB default; small JSON APIs (CDX index, CC index list) get a tighter
# 512 KiB because their payloads are bounded by the server already.
_HTML_MAX_BYTES = 2_000_000
_JSON_MAX_BYTES = 512 * 1024


async def _spacing_guard() -> None:
    global _last_jina_ts
    try:
        now = time.monotonic()
        wait = _JINA_RPM_GUARD_S - (now - _last_jina_ts)
        if wait > 0:
            await asyncio.sleep(wait)
        _last_jina_ts = time.monotonic()
    except Exception:
        pass


async def jina_fetch(url: str, timeout: float = _TIMEOUT) -> dict | None:
    """Fetch url via keyless Jina Reader. Returns fetch-like dict or None."""
    try:
        url = await _validate_url(url)
    except Exception:
        return None
    await _spacing_guard()
    try:
        c = get_http_client()
        meta: dict = {}
        body_bytes, _trunc = await safe_fetch_capped(
            c, "https://r.jina.ai/" + url, timeout=timeout,
            max_bytes=_HTML_MAX_BYTES,
            headers={"Accept": "text/plain",
                     "User-Agent": "websearch-mcp/1.0"},
            meta_out=meta)
        if meta.get("status") != 200:
            return None
        final = meta.get("url")
        if final and final != "https://r.jina.ai/" + url:
            try:
                await _validate_url(str(final))
            except Exception:
                return None
        body = body_bytes.decode("utf-8", errors="replace")
        if not body or len(body.strip()) < 50:
            return None
        title = ""
        for line in body.splitlines()[:8]:
            s = line.strip().lstrip("# ").strip()
            if len(s) >= 8:
                title = s[:200]
                break
        return {"success": True, "url": url, "content": body.strip(),
                "status": 200, "title": title, "metadata": {},
                "content_type": "text/plain", "method": "jina-reader"}
    except Exception as e:
        logger.debug("jina fetch failed for %s: %s", url, e)
        return None


async def cdx_history(url: str, limit: int = 5,
                      timeout: float = _TIMEOUT) -> list[dict]:
    """Alternate Wayback snapshots for url (timestamp/url/digest/status).

    collapse=digest drops same-content re-crawls; filter=statuscode:200
    keeps only good captures. Returns [] when none. The url= param is
    normalized to bare-host form: CDX intermittently 503s on other forms
    (verified 2026-10-11: full-URL form 503, bare domain 200).
    """
    try:
        q = (url or "").strip()
        try:
            parts = urllib.parse.urlsplit(q if "://" in q else "http://" + q)
            host = (parts.hostname or q).lower()
            if host.startswith("www."):
                host = host[4:]
            path = parts.path or ""
            if path in ("", "/"):
                q = host
        except Exception:
            pass
        from config import get_http_client
        c = get_http_client()
        meta: dict = {}
        body_bytes, _trunc = await safe_fetch_capped(
            c,
            "https://web.archive.org/cdx/search/cdx",
            params={"url": q, "output": "json", "limit": limit,
                    "collapse": "digest", "filter": "statuscode:200"},
            timeout=timeout,
            max_bytes=_JSON_MAX_BYTES,
            headers={"User-Agent": "websearch-mcp/1.0"},
            meta_out=meta,
        )
        if meta.get("status") != 200:
            return []
        try:
            rows = json.loads(body_bytes.decode("utf-8", errors="replace"))
        except ValueError:
            return []
        out: list[dict] = []
        if not rows:
            return []
        header = rows[0] if isinstance(rows[0], list) else []
        try:
            ti = header.index("timestamp")
            oi = header.index("original")
            di = header.index("digest") if "digest" in header else -1
            si = header.index("statuscode") if "statuscode" in header else -1
        except ValueError:
            ti, oi, di, si = 1, 0, -1, -1
        for row in rows[1:]:
            if not isinstance(row, list) or len(row) <= max(ti, oi):
                continue
            ts = str(row[ti] or "")
            orig = str(row[oi] or "") or url
            if not ts:
                continue
            out.append({
                "timestamp": ts,
                "url": f"https://web.archive.org/web/{ts}id_/{orig}",
                "digest": row[di] if di >= 0 and len(row) > di else "",
                "status": row[si] if si >= 0 and len(row) > si else "200",
            })
            if len(out) >= limit:
                break
        return out
    except Exception as e:
        logger.debug("cdx history failed for %s: %s", url, e)
        return []


async def cdx_fetch_snapshot(snapshot_url: str,
                             timeout: float = _TIMEOUT) -> dict | None:
    """Fetch one CDX snapshot URL through trafilatura (shared tail shape)."""
    try:
        c = get_http_client()
        meta: dict = {}
        body_bytes, _trunc = await safe_fetch_capped(
            c, snapshot_url, timeout=timeout,
            max_bytes=_HTML_MAX_BYTES,
            headers={"User-Agent": "websearch-mcp/1.0"},
            meta_out=meta)
        if meta.get("status") != 200:
            return None
        body = body_bytes.decode("utf-8", errors="replace")
        if not body or len(body.strip()) < 50:
            return None
        try:
            from fetch import _archive_extract
            content, title = _archive_extract(body, snapshot_url)
        except Exception:
            return None
        if not content or len(content) < 50:
            return None
        return {"success": True, "content": content, "title": title,
                "snapshot_url": snapshot_url, "method": "wayback-cdx"}
    except Exception as e:
        logger.debug("cdx snapshot fetch failed: %s", e)
        return None


_CC_INDEX = "https://index.commoncrawl.org/collinfo.json"
_CC_TIMEOUT = 12.0


async def cc_latest_index() -> str | None:
    """Newest Common Crawl index id (e.g. CC-MAIN-2026-38). Cached 24h in-process."""
    global _cc_idx_cache
    try:
        if _cc_idx_cache[1] and time.monotonic() - _cc_idx_cache[0] < 86400:
            return _cc_idx_cache[1]
    except Exception:
        pass
    try:
        c = get_http_client()
        meta: dict = {}
        body_bytes, _trunc = await safe_fetch_capped(
            c, _CC_INDEX, timeout=_CC_TIMEOUT,
            max_bytes=_JSON_MAX_BYTES,
            headers={"User-Agent": "websearch-mcp/1.0"},
            meta_out=meta)
        if meta.get("status") != 200:
            return None
        try:
            idxs = json.loads(body_bytes.decode("utf-8", errors="replace"))
        except ValueError:
            return None
        name = (idxs[0].get("id") or "") if idxs else ""
        if name:
            _cc_idx_cache = (time.monotonic(), name)
        return name or None
    except Exception as e:
        logger.debug("cc collinfo failed: %s", e)
        return None


_cc_idx_cache: tuple[float, str | None] = (0.0, None)


async def cc_lookup(url_or_domain: str, limit: int = 20,
                    index: str | None = None) -> list[dict]:
    """Common Crawl URL-index lookup: records {url, timestamp, offset, length,
    filename} for domain surveys (map_site assist) or dead-link recovery."""
    idx = index or await cc_latest_index()
    if not idx:
        return []
    try:
        c = get_http_client()
        meta: dict = {}
        body_bytes, _trunc = await safe_fetch_capped(
            c, f"https://index.commoncrawl.org/{idx}-index",
            params={"url": url_or_domain, "output": "json",
                    "limit": min(max(limit, 1), 200)},
            timeout=_CC_TIMEOUT,
            max_bytes=_JSON_MAX_BYTES,
            headers={"User-Agent": "websearch-mcp/1.0"},
            meta_out=meta,
        )
        if meta.get("status") != 200:
            return []
        out: list[dict] = []
        for line in body_bytes.decode("utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            if not rec.get("url"):
                continue
            out.append({"url": rec.get("url", ""),
                        "timestamp": rec.get("timestamp", ""),
                        "offset": rec.get("offset", ""),
                        "length": rec.get("length", ""),
                        "filename": rec.get("filename", ""),
                        "status": rec.get("status", "")})
            if len(out) >= limit:
                break
        return out
    except Exception as e:
        logger.debug("cc lookup failed for %s: %s", url_or_domain, e)
        return []


async def cc_fetch_record(filename: str, offset: str, length: str,
                          url: str = "") -> dict | None:
    """Fetch one WARC record by byte range; return extracted text or None."""
    try:
        off, ln = int(offset), int(length)
        if off < 0 or ln <= 0 or ln > 100_000_000:
            return None
    except (TypeError, ValueError):
        return None
    try:
        data_url = f"https://data.commoncrawl.org/{filename}"
        c = get_http_client()
        meta: dict = {}
        raw, _trunc = await safe_fetch_capped(
            c, data_url, timeout=20,
            max_bytes=ln,
            headers={"Range": f"bytes={off}-{off + ln - 1}",
                     "User-Agent": "websearch-mcp/1.0"},
            meta_out=meta)
        if meta.get("status") not in (200, 206):
            return None
        if not raw:
            return None
        try:
            try:
                raw = gzip.decompress(raw)
            except Exception:
                pass
            text = raw.decode("utf-8", errors="replace")
            sep = "\r\n\r\n"
            body = text.split(sep, 2)[-1] if sep in text else text
            if len(body.strip()) < 50:
                return None
            return {"success": True, "content": body.strip()[:20000],
                    "url": url, "method": "commoncrawl",
                    "cached_from": f"commoncrawl ({filename})"}
        except Exception:
            return None
    except Exception as e:
        logger.debug("cc record fetch failed: %s", e)
        return None


if __name__ == "__main__":
    async def check():
        snaps = await cdx_history("https://example.com", limit=3)
        print(f"cdx example.com: {len(snaps)} snapshots")
        idx = await cc_latest_index()
        print(f"cc latest index: {idx}")
        recs = await cc_lookup("example.com", limit=2)
        print(f"cc example.com: {len(recs)} records")

    asyncio.run(check())
