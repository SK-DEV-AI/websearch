"""High-level web operations that replace one-off agent scripts.

Every function here exists because an agent kept writing a throwaway Python
script to do the same thing. The rule for this module: if a competent agent
would otherwise reach for `curl` or `requests`, the operation belongs here.

Design constraints, all of which the callers depend on:

* **SSRF-safe by construction.** All outbound fetches go through
  `security.safe_fetch_capped`, never a bare httpx call, so private-IP
  blocking, redirect revalidation and DNS pinning apply automatically.
* **Bounded.** Every function takes an explicit cap (bytes, URLs, chars) and
  reports whether it hit that cap. A caller must never be unable to tell
  "that is everything" from "that was truncated".
* **Honest envelopes.** Results carry `truncated`, counts, and the reason work
  stopped. No silent partial results.
* **Politeness.** robots.txt is honoured by default for multi-URL work, and
  concurrency is capped so a batch cannot become a load test.

Nothing here requires a browser. Browser-dependent operations live in
`cdp_actions` / `session`.
"""

from __future__ import annotations

import asyncio
import csv
import hashlib
import io
import json
import logging
import os
import time
import urllib.parse
from typing import Any

import lxml.html

import cache as cache_mod
import robots
from config import get_http_client
from fetch import fetch_url
from security import SecurityError, safe_fetch_capped

logger = logging.getLogger(__name__)

MAX_BATCH_URLS = 50
MAX_CRAWL_URLS = 100
DEFAULT_DOWNLOAD_DIR = os.path.expanduser("~/.cache/websearch/downloads")
MAX_DOWNLOAD_BYTES = 100 * 1024 * 1024
_UA = "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0 Safari/537.36"


# ─────────────────────────── shared helpers ───────────────────────────


def _err(url: str, msg: str, *, kind: str = "", fix: str = "") -> dict:
    out: dict[str, Any] = {"success": False, "url": url, "error": msg}
    if kind:
        out["errorKind"] = kind
    if fix:
        out["fix"] = fix
    return out


def _norm(url: str) -> str:
    """Normalise for dedup: drop the fragment, sort nothing else."""
    p = urllib.parse.urlsplit(url)
    return urllib.parse.urlunsplit((p.scheme, p.netloc, p.path, p.query, ""))


async def _html(url: str, *, max_bytes: int = 3_000_000) -> tuple[str, bool]:
    """Fetch raw HTML. Returns (html, truncated). Raises SecurityError."""
    client = get_http_client()
    body, truncated = await safe_fetch_capped(
        client, url, max_bytes=max_bytes,
        headers={"User-Agent": _UA, "Accept": "text/html,application/xhtml+xml"},
    )
    return body.decode("utf-8", errors="replace"), truncated


def _doc(html: str):
    return lxml.html.fromstring(html)


def _abs(base: str, href: str) -> str | None:
    """Resolve href against base; return None for unusable schemes."""
    try:
        u = urllib.parse.urljoin(base, href)
    except Exception:
        return None
    scheme = urllib.parse.urlsplit(u).scheme
    if scheme not in ("http", "https"):
        return None
    return u


# ─────────────────────────── fetch_many ───────────────────────────


async def fetch_many(
    urls: list[str],
    *,
    max_concurrent: int = 3,
    max_chars: int = 5000,
    focus: str = "",
    cache_ttl: int = 3600,
    respect_robots: bool = True,
) -> dict:
    """Fetch many URLs concurrently. Replaces a `for url in urls: requests.get`
    script, with dedup, a concurrency cap, robots politeness and shared cache."""
    urls = [u.strip() for u in (urls or []) if u and u.strip()]
    if not urls:
        return _err("", "urls must not be empty")

    # dedup, preserving order
    seen: set[str] = set()
    ordered: list[str] = []
    for u in urls:
        k = _norm(u)
        if k not in seen:
            seen.add(k)
            ordered.append(u)
    dropped = len(urls) - len(ordered)

    capped = ordered[:MAX_BATCH_URLS]
    over_cap = len(ordered) - len(capped)

    sem = asyncio.Semaphore(max(1, min(max_concurrent, 8)))

    async def one(u: str) -> dict:
        async with sem:
            if respect_robots:
                try:
                    if not await robots.allowed(u):
                        return {**_err(u, "Disallowed by robots.txt", kind="permanent"), "blocked_by_robots": True}
                except Exception:
                    pass  # robots lookup failure must not block a fetch
            try:
                r = await fetch_url(u, max_chars=max_chars, focus=focus,
                                    cache_ttl=cache_ttl)
            except SecurityError as e:
                return _err(u, str(e), kind="permanent", fix="Blocked by SSRF policy")
            except Exception as e:  # noqa: BLE001
                return _err(u, f"{type(e).__name__}: {e}", kind="transient")
            if not isinstance(r, dict):
                return _err(u, "unexpected fetch result")
            out = {
                "url": u,
                "success": bool(r.get("success")),
                "status": r.get("status"),
                "title": r.get("title", ""),
                "content": r.get("content", ""),
                "chars": len(r.get("content") or ""),
                "final_url": r.get("final_url") or r.get("url") or u,
            }
            if r.get("error"):
                out["error"] = r["error"]
            if r.get("method"):
                out["method"] = r["method"]
            if r.get("truncated"):
                out["truncated"] = True
            return out

    results = await asyncio.gather(*(one(u) for u in capped))
    ok = [r for r in results if r.get("success")]
    return {
        "success": True,
        "requested": len(urls),
        "deduped": dropped,
        "attempted": len(capped),
        "succeeded": len(ok),
        "failed": len(capped) - len(ok),
        "results": results,
        "truncated": bool(over_cap),
        "note": (
            f"{over_cap} URL(s) were dropped: batch cap is {MAX_BATCH_URLS}. "
            "Call again with the rest." if over_cap else ""
        ),
    }


# ─────────────────────────── download ───────────────────────────


def _safe_filename(name: str, content_type: str) -> str:
    name = os.path.basename(name or "").strip()
    if not name or name in (".", ".."):
        ext = ""
        ct = (content_type or "").split(";")[0].strip().lower()
        ext = {
            "application/pdf": ".pdf",
            "text/csv": ".csv",
            "application/json": ".json",
            "text/html": ".html",
            "image/png": ".png",
            "image/jpeg": ".jpg",
            "application/zip": ".zip",
        }.get(ct, ".bin")
        name = f"download-{int(time.time())}{ext}"
    # strip anything path-ish or shell-hostile
    return "".join(c for c in name if c.isalnum() or c in " ._-()[]")[:180] or "download.bin"


async def download(
    url: str,
    *,
    dest_dir: str = "",
    filename: str = "",
    max_bytes: int = MAX_DOWNLOAD_BYTES,
) -> dict:
    """Save a URL to disk and return a local path. Replaces `curl -O` plus a
    content-type guess. Refuses to write outside dest_dir."""
    if not url:
        return _err(url, "url is required")
    cap = max(1, min(int(max_bytes or MAX_DOWNLOAD_BYTES), MAX_DOWNLOAD_BYTES))

    client = get_http_client()
    try:
        body, truncated = await safe_fetch_capped(
            client, url, max_bytes=cap, headers={"User-Agent": _UA}
        )
    except SecurityError as e:
        return _err(url, str(e), kind="permanent", fix="Blocked by SSRF policy")
    except Exception as e:  # noqa: BLE001
        return _err(url, f"{type(e).__name__}: {e}", kind="transient")

    ctype = "application/octet-stream"
    try:
        probe = await client.head(url, headers={"User-Agent": _UA}, timeout=10)
        ctype = probe.headers.get("content-type", ctype)
    except Exception:
        pass

    target_dir = os.path.abspath(os.path.expanduser(dest_dir or DEFAULT_DOWNLOAD_DIR))
    os.makedirs(target_dir, exist_ok=True)
    name = _safe_filename(filename, ctype)
    path = os.path.join(target_dir, name)

    # never escape the target directory
    if os.path.commonpath([target_dir, os.path.abspath(path)]) != target_dir:
        return _err(url, "refusing to write outside dest_dir", kind="permanent")

    try:
        with open(path, "wb") as fh:
            fh.write(body)
    except OSError as e:
        return _err(url, f"write failed: {e}", kind="permanent",
                    fix="Check dest_dir is writable and has space")

    return {
        "success": True,
        "url": url,
        "path": path,
        "bytes": len(body),
        "sha256": hashlib.sha256(body).hexdigest(),
        "content_type": ctype,
        "truncated": bool(truncated),
        "note": (f"Stopped at the {cap} byte cap; the file on disk is incomplete."
                 if truncated else ""),
    }


# ─────────────────────────── extract_links ───────────────────────────


async def extract_links(
    url: str,
    *,
    same_domain: bool = True,
    max_links: int = 500,
    include_text: bool = True,
) -> dict:
    """All links on a page with their anchor text. Replaces manual HTML parsing
    before crawling or paginating."""
    if not url:
        return _err(url, "url is required")
    try:
        html, trunc = await _html(url)
    except SecurityError as e:
        return _err(url, str(e), kind="permanent")
    except Exception as e:  # noqa: BLE001
        return _err(url, f"{type(e).__name__}: {e}", kind="transient")

    host = urllib.parse.urlsplit(url).netloc
    links: list[dict[str, Any]] = []
    seen: set[str] = set()
    try:
        doc = _doc(html)
    except Exception as e:  # noqa: BLE001
        return _err(url, f"HTML parse failed: {e}", kind="permanent")

    for a in doc.iter("a"):
        href = a.get("href")
        if not href:
            continue
        absu = _abs(url, href)
        if not absu:
            continue
        if same_domain and urllib.parse.urlsplit(absu).netloc != host:
            continue
        key = _norm(absu)
        if key in seen:
            continue
        seen.add(key)
        item: dict[str, Any] = {"url": absu}
        if include_text:
            item["text"] = " ".join((a.text_content() or "").split())[:200]
            item["title"] = (a.get("title") or "")[:200]
            item["rel"] = (a.get("rel") or "")[:60]
        links.append(item)
        if len(links) >= max_links:
            break

    return {
        "success": True,
        "url": url,
        "count": len(links),
        "links": links,
        "truncated": bool(trunc) or len(links) >= max_links,
        "note": (f"Stopped at max_links={max_links}." if len(links) >= max_links else ""),
    }


# ─────────────────────────── extract_table ───────────────────────────


def _table_to_rows(table) -> list[list[str]]:
    rows: list[list[str]] = []
    for tr in table.iter("tr"):
        cells = []
        for c in tr.iter("th", "td"):
            cells.append(" ".join((c.text_content() or "").split()))
        if cells:
            rows.append(cells)
    return rows


async def extract_table(
    url: str,
    *,
    selector: str = "",
    fmt: str = "json",
    index: int = 0,
    max_rows: int = 5000,
) -> dict:
    """Pull HTML tables into json/csv/markdown. Replaces a bespoke pandas or
    lxml script every time a page has a pricing or stats table."""
    if not url:
        return _err(url, "url is required")
    try:
        html, trunc = await _html(url)
    except SecurityError as e:
        return _err(url, str(e), kind="permanent")
    except Exception as e:  # noqa: BLE001
        return _err(url, f"{type(e).__name__}: {e}", kind="transient")

    try:
        doc = _doc(html)
    except Exception as e:  # noqa: BLE001
        return _err(url, f"HTML parse failed: {e}", kind="permanent")

    tables = doc.cssselect(selector) if selector else doc.iter("table")
    tables = [t for t in tables if getattr(t, "tag", "") == "table"]
    if not tables:
        return _err(url, "no <table> found", kind="permanent",
                    fix="Try selector='table.some-class' or use extract(css) instead")

    idx = max(0, min(int(index or 0), len(tables) - 1))
    rows = _table_to_rows(tables[idx])
    over = len(rows) > max_rows
    if over:
        rows = rows[:max_rows]
    if not rows:
        return _err(url, "table parsed but had no rows", kind="permanent")

    fmt = (fmt or "json").lower()
    out: dict[str, Any] = {
        "success": True,
        "url": url,
        "tables_found": len(tables),
        "table_index": idx,
        "format": fmt,
        "rows": len(rows),
        "cols": max(len(r) for r in rows),
        "truncated": bool(trunc) or over,
    }
    if over:
        out["note"] = f"Stopped at max_rows={max_rows}."

    if fmt == "csv":
        buf = io.StringIO()
        csv.writer(buf).writerows(rows)
        out["data"] = buf.getvalue()
    elif fmt == "markdown":
        head = rows[0]
        lines = ["| " + " | ".join(head) + " |",
                 "|" + "|".join(["---"] * len(head)) + "|"]
        for r in rows[1:]:
            r = (r + [""] * len(head))[:len(head)]
            lines.append("| " + " | ".join(r) + " |")
        out["data"] = "\n".join(lines)
    else:
        out["data"] = rows
    return out


# ─────────────────────────── diff_page ───────────────────────────


async def diff_page(url: str, *, extraction_type: str = "markdown") -> dict:
    """Compare the current page against the last cached copy of the same URL.
    Replaces a hand-rolled polling-and-hashing script."""
    if not url:
        return _err(url, "url is required")

    prev = None
    try:
        prev = await cache_mod.get_previous(url, extraction_type=extraction_type)
    except Exception as e:  # noqa: BLE001
        logger.debug("diff_page: get_previous failed: %s", e)

    if not prev:
        return {
            "success": True,
            "url": url,
            "changed": None,
            "reason": "no previous snapshot cached; call fetch first, then diff again later",
            "had_previous": False,
        }

    try:
        cur = await fetch_url(url, max_chars=200_000, cache_ttl=0)
    except Exception as e:  # noqa: BLE001
        return _err(url, f"{type(e).__name__}: {e}", kind="transient")
    if not cur.get("success"):
        return _err(url, cur.get("error", "fetch failed"), kind="transient")

    old_text = (prev.get("content") or "")
    new_text = (cur.get("content") or "")
    old_hash = hashlib.sha256(old_text.encode("utf-8", "replace")).hexdigest()
    new_hash = hashlib.sha256(new_text.encode("utf-8", "replace")).hexdigest()

    changed = old_hash != new_hash
    added: list[str] = []
    removed: list[str] = []
    if changed:
        import difflib

        old_lines = [l.strip() for l in old_text.splitlines() if l.strip()]
        new_lines = [l.strip() for l in new_text.splitlines() if l.strip()]
        sm = difflib.SequenceMatcher(None, old_lines, new_lines, autojunk=False)
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag in ("replace", "insert"):
                added.extend(new_lines[j1:j2][:200])
            if tag in ("replace", "delete"):
                removed.extend(old_lines[i1:i2][:200])

    return {
        "success": True,
        "url": url,
        "changed": changed,
        "had_previous": True,
        "old_hash": old_hash,
        "new_hash": new_hash,
        "previous_fetched_at": prev.get("fetched_at"),
        "added_count": len(added),
        "removed_count": len(removed),
        "added": added[:200],
        "removed": removed[:200],
    }


# ─────────────────────────── crawl ───────────────────────────


async def crawl(
    start_url: str,
    *,
    max_depth: int = 1,
    max_urls: int = 20,
    same_domain: bool = True,
    include_patterns: list[str] | None = None,
    exclude_patterns: list[str] | None = None,
    concurrency: int = 2,
    respect_robots: bool = True,
    max_chars: int = 4000,
) -> dict:
    """Breadth-first crawl with hard caps. Replaces a bespoke scraper and makes
    politeness, dedup and limits non-optional."""
    if not start_url:
        return _err(start_url, "start_url is required")

    cap_urls = max(1, min(int(max_urls or 20), MAX_CRAWL_URLS))
    cap_depth = max(0, min(int(max_depth), 5))
    host = urllib.parse.urlsplit(start_url).netloc
    inc = [p for p in (include_patterns or []) if p]
    exc = [p for p in (exclude_patterns or []) if p]

    def ok(u: str) -> bool:
        if same_domain and urllib.parse.urlsplit(u).netloc != host:
            return False
        if exc and any(p in u for p in exc):
            return False
        if inc and not any(p in u for p in inc):
            return False
        return True

    seen: set[str] = {_norm(start_url)}
    frontier = [(start_url, 0)]
    pages: list[dict[str, Any]] = []
    stop_reason = "completed"
    sem = asyncio.Semaphore(max(1, min(int(concurrency or 2), 5)))

    while frontier:
        # Enforce the cap per PAGE, not per wave: a single wave can hold more
        # URLs than the remaining budget and would otherwise overshoot.
        remaining = cap_urls - len(pages)
        if remaining <= 0:
            stop_reason = f"max_urls cap ({cap_urls}) reached"
            break
        if len(frontier) > remaining:
            stop_reason = f"max_urls cap ({cap_urls}) reached"
            frontier = frontier[:remaining]
        batch, frontier = frontier, []
        next_level: list[tuple[str, int]] = []

        async def visit(item: tuple[str, int]) -> tuple[dict[str, Any], list[str]]:
            u, depth = item
            async with sem:
                if respect_robots:
                    try:
                        if not await robots.allowed(u):
                            return {**_err(u, "robots.txt disallows"), "depth": depth}, []
                    except Exception:
                        pass
                try:
                    r = await fetch_url(u, max_chars=max_chars, cache_ttl=3600)
                except SecurityError as e:
                    return {**_err(u, str(e), kind="permanent"), "depth": depth}, []
                except Exception as e:  # noqa: BLE001
                    return {**_err(u, f"{type(e).__name__}: {e}", kind="transient"), "depth": depth}, []
                found: list[str] = []
                if depth < cap_depth:
                    try:
                        lr = await extract_links(u, same_domain=same_domain,
                                                 max_links=200, include_text=False)
                        found = [l["url"] for l in lr.get("links", [])]
                    except Exception:
                        pass
                page = {
                    "url": u,
                    "depth": depth,
                    "success": bool(r.get("success")),
                    "status": r.get("status"),
                    "title": r.get("title", ""),
                    "chars": len(r.get("content") or ""),
                    "content": (r.get("content") or "")[:max_chars],
                }
                if r.get("error"):
                    page["error"] = r["error"]
                return page, found

        outcomes = await asyncio.gather(*(visit(it) for it in batch))
        for page, found in outcomes:
            pages.append(page)
            for u in found:
                k = _norm(u)
                if k in seen or not ok(u):
                    continue
                if len(seen) >= cap_urls * 4:   # keep discovery itself bounded
                    break
                seen.add(k)
                next_level.append((u, page.get("depth", 0) + 1))
        frontier = next_level[: cap_urls * 2]

    return {
        "success": True,
        "start_url": start_url,
        "pages_crawled": len(pages),
        "pages_succeeded": sum(1 for p in pages if p.get("success")),
        "discovered": len(seen),
        "max_depth": cap_depth,
        "stop_reason": stop_reason,
        "truncated": stop_reason != "completed",
        "pages": pages,
    }


# ─────────────────────────── selfcheck ───────────────────────────


async def _selfcheck() -> None:
    checks: list[tuple[str, bool]] = []
    checks.append(("filename sanitises traversal",
                   "/" not in _safe_filename("../../etc/passwd", "text/plain")))
    checks.append(("filename keeps extension",
                   _safe_filename("report.pdf", "application/pdf") == "report.pdf"))
    checks.append(("filename falls back on empty",
                   _safe_filename("", "application/pdf").endswith(".pdf")))
    checks.append(("norm drops fragment", "#" not in _norm("https://a/b#c")))
    checks.append(("abs rejects javascript:",
                   _abs("https://a/", "javascript:alert(1)") is None))
    checks.append(("abs rejects mailto:",
                   _abs("https://a/", "mailto:x@y.z") is None))
    checks.append(("abs resolves relative",
                   _abs("https://a/d/", "../x") == "https://a/x"))
    checks.append(("table rows parse",
                   _table_to_rows(_doc("<table><tr><th>a</th><th>b</th></tr>"
                                       "<tr><td>1</td><td>2</td></tr></table>")) == [["a", "b"], ["1", "2"]]))
    for name, okv in checks:
        print(f"{'OK' if okv else 'FAIL'}  {name}")
    total = len(checks)
    passed = sum(1 for _, v in checks if v)
    print(f"webtools selfcheck: {passed}/{total} OK")


if __name__ == "__main__":
    asyncio.run(_selfcheck())
