"""Conditional revalidation cache (ETag / Last-Modified / Cache-Control).

Port of donsetch src/fetch/revalidate.rs. Browser-true cache
behavior: honor fresh windows without a request, otherwise send
conditional headers and accept 304. Scrapers never do this;
browsers always do.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field
from typing import Any

MAX_ENTRIES = 512
MAX_BODY = 8 << 20       # 8 MiB
MAX_FRESH_SECS = 3600    # max-age cap (revalidate at least hourly)


@dataclass
class CacheEntry:
    body: bytes
    status: int
    headers: list[tuple[str, str]]
    etag: str | None
    last_modified: str | None
    fresh_until: float | None  # unix seconds


@dataclass
class RevalidationCache:
    map: dict[str, CacheEntry] = field(default_factory=dict)
    _lock: threading.Lock = field(default_factory=threading.Lock)

    # Returns ("fresh", body, status, headers) to serve without a
    # request, ("revalidate", [(name, value), ...]) conditional headers
    # to send (a 304 means serve the stored body), or None.
    def check(self, url: str) -> tuple | None:
        with self._lock:
            entry = self.map.get(url)
            if entry is None:
                return None
            if entry.fresh_until is not None and time.time() < entry.fresh_until:
                return ("fresh", entry.body, entry.status, entry.headers)
            cond: list[tuple[str, str]] = []
            if entry.etag is not None:
                cond.append(("if-none-match", entry.etag))
            if entry.last_modified is not None:
                cond.append(("if-modified-since", entry.last_modified))
            if not cond:
                return None
            return ("revalidate", cond)

    # Stored body for a 304 merge.
    def stored(self, url: str) -> tuple | None:
        with self._lock:
            e = self.map.get(url)
            if e is None:
                return None
            return (e.body, e.status, e.headers)

    def store(self, url: str, status: int, headers: Any, body: bytes) -> None:
        if status != 200 or len(body) > MAX_BODY:
            return

        def get(name: str) -> str | None:
            if isinstance(headers, dict):
                for k, v in headers.items():
                    if k.lower() == name and isinstance(v, str):
                        return v
                return None
            for n, v in headers:
                if n.lower() == name:
                    return v
            return None

        cache_control = (get("cache-control") or "").lower()
        if "no-store" in cache_control or "private" in cache_control:
            return
        etag = get("etag")
        last_modified = get("last-modified")
        fresh_until = _parse_max_age(cache_control)
        if fresh_until is not None:
            fresh_until = min(fresh_until, MAX_FRESH_SECS)
            fresh_until = time.time() + fresh_until
        # Cache only when there's a reason: a validator or a fresh window.
        if etag is None and last_modified is None and fresh_until is None:
            return
        with self._lock:
            if len(self.map) >= MAX_ENTRIES and url not in self.map:
                self.map.pop(next(iter(self.map)))
            self.map[url] = CacheEntry(
                bytes(body), status, list(headers.items()) if isinstance(headers, dict)
                else [(str(n), str(v)) for n, v in headers],
                etag, last_modified, fresh_until,
            )


def _parse_max_age(cache_control: str) -> float | None:
    for part in cache_control.split(","):
        part = part.strip()
        if part.startswith("max-age="):
            try:
                return float(part[len("max-age="):].strip().strip('"'))
            except ValueError:
                return None
    return None


def _selfcheck() -> None:
    c = RevalidationCache()
    hdr = [("etag", '"abc"'), ("last-modified", "Wed, 01 Jan 2025 00:00:00 GMT")]

    # no validator / fresh window -> not cached
    c.store("http://x.no-vals", 200, [("content-type", "text/html")], b"<html>hi</html>")
    assert c.check("http://x.no-vals") is None, "no validator must not cache"

    # etag-only entry -> revalidate with if-none-match
    c.store("http://x/e", 200, hdr, b"body")
    kind, cond = c.check("http://x/e")
    assert kind == "revalidate" and ("if-none-match", '"abc"') in cond, cond

    # fresh window -> serve without request
    c.store("http://x/f", 200, [("cache-control", "max-age=300")], b"fresh")
    kind = c.check("http://x/f")[0]
    assert kind == "fresh", kind

    # max-age cap: 3600 max, not the requested 7200
    c.store("http://x/cap", 200, [("cache-control", "max-age=7200")], b"cap")
    until = c.map["http://x/cap"].fresh_until
    assert until - time.time() <= 3601, until

    # no-store / private rejected
    c.store("http://x/ns", 200, [("cache-control", "no-store"), ("etag", "x")], b"ns")
    assert c.check("http://x/ns") is None
    c.store("http://x/pr", 200, [("cache-control", "private"), ("etag", "x")], b"pr")
    assert c.check("http://x/pr") is None

    # non-200 and oversized rejected
    c.store("http://x/err", 404, hdr, b"nope")
    assert c.check("http://x/err") is None
    c.store("http://x/big", 200, hdr, b"0" * (MAX_BODY + 1))
    assert c.check("http://x/big") is None

    # 304 merge path
    c.store("http://x/m", 200, hdr, b"stored-body")
    merged = c.stored("http://x/m")
    assert merged == (b"stored-body", 200, hdr), merged

    # max-age parsing
    assert _parse_max_age("public, max-age=60") == 60.0
    assert _parse_max_age("max-age=\"30\"") == 30.0
    assert _parse_max_age("no-cache") is None

    print("REVALIDATE-OK")


if __name__ == "__main__":
    _selfcheck()
