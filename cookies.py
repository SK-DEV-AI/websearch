"""Minimal RFC 6265 cookie jar, scoped per domain/path.

Port of donsetch src/fetch/cookies.rs. Tracks real expiry
(Max-Age -> expires_at) so the self-improving fetch loop can
write fresh cookies back into the ghost-state vault.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

MAX_AGE_HEADERS = ("max-age",)


@dataclass
class Cookie:
    name: str
    value: str
    domain: str
    path: str
    host_only: bool
    expires_at: float | None = None  # unix-seconds; None = session cookie


@dataclass
class CookieJar:
    cookies: list[Cookie] = field(default_factory=list)

    # ── RFC 6265 §5.3 ingestion ───────────────────────────────────
    def store_from_headers(self, host: str, headers: Any) -> None:
        """Store all Set-Cookie headers from a response for `host`."""
        for name, value in self._set_cookie_pairs(headers):
            self._store_one(host, name, value)

    def _set_cookie_pairs(self, headers: Any) -> list[tuple[str, str]]:
        pairs: list[tuple[str, str]] = []
        if headers is None:
            return pairs
        # httpx.Headers exposes get_list for multi-valued headers; get_all does not exist
        get_list = getattr(headers, "get_list", None)
        if callable(get_list):
            try:
                for v in get_list("set-cookie"):
                    pairs.append(("set-cookie", v))
                if pairs:
                    return pairs
            except Exception:
                pass
        if isinstance(headers, dict):
            for k, v in headers.items():
                if k.lower() == "set-cookie" and isinstance(v, str):
                    pairs.append(("set-cookie", v))
            return pairs
        for item in headers:
            k = item[0]
            if k.lower() == "set-cookie" and len(item) > 1:
                pairs.append(("set-cookie", item[1]))
        return pairs

    def _store_one(self, host: str, _name: str, value: str) -> None:
        parts = value.split(";")
        if not parts:
            return
        pair = parts[0]
        if "=" not in pair:
            return
        name, val = pair.split("=", 1)
        name = name.strip()
        val = val.strip()
        if not name:
            return
        # Control chars in name/value can split the Cookie request
        # header later (request splitting). Reject the cookie outright.
        if any(c in name + val for c in "\r\n\0"):
            return
        domain = host
        host_only = True
        path = "/"
        expired = False
        expires_at: float | None = None
        for attr in parts[1:]:
            attr = attr.strip()
            if "=" not in attr:
                continue
            k, v = attr.split("=", 1)
            k = k.strip().lower()
            v = v.strip()
            if k == "domain":
                d = v.lstrip(".").lower()
                # RFC 6265 §5.3 step 6: reject Domain attributes that
                # are not the request host or a parent of it — else any
                # origin can pin cookies on any victim domain
                # (cookie tossing).
                if d == host or host.endswith("." + d):
                    domain = d
                    host_only = False
            elif k == "path":
                path = v
            elif k == "max-age":
                try:
                    secs = int(v)
                except ValueError:
                    continue
                if secs <= 0:
                    expired = True
                else:
                    expires_at = time.time() + secs
        # Replace any existing cookie with same (name, domain, path).
        self.cookies = [c for c in self.cookies
                        if not (c.name == name and c.domain == domain and c.path == path)]
        if not expired:
            self.cookies.append(Cookie(name, val, domain, path, host_only, expires_at))
        self.purge_expired()

    # ── out-of-band handoff ───────────────────────────────────────
    def store_raw(self, name: str, value: str, domain: str,
                  expires_at: float | None = None) -> None:
        """Inject a cookie harvested out-of-band (CDP clearance
        handoff). Leading-dot domains are subdomain cookies; bare
        domains are host-only. `expires_at` carries the real CDP expiry."""
        if any(c in name + value for c in "\r\n\0"):
            return
        dom, host_only = (domain.lstrip("."), False) if domain.startswith(".") else (domain, True)
        self.cookies = [c for c in self.cookies
                        if not (c.name == name and c.domain == dom and c.path == "/")]
        self.cookies.append(Cookie(name, value, dom, "/", host_only, expires_at))

    # ── export ────────────────────────────────────────────────────
    def snapshot_for(self, host: str) -> list[dict[str, Any]]:
        """Export all cookies matching `host` for write-back to the
        persistent domain profile."""
        now = time.time()
        out = []
        for c in self.cookies:
            if c.expires_at is not None and c.expires_at <= now:
                continue
            if not self._domain_ok(c, host):
                continue
            out.append({"name": c.name, "value": c.value,
                        "domain": c.domain, "expires_at": c.expires_at})
        return out

    def dict_for(self, host: str, path: str) -> dict[str, str]:
        """Cookie dict for a request to `host` + `path`, if any match."""
        pairs = []
        now = time.time()
        for c in self.cookies:
            # Session cookies (no expiry) always match; expired
            # cookies must never be replayed.
            if c.expires_at is not None and c.expires_at <= now:
                continue
            if not self._domain_ok(c, host):
                continue
            # RFC 6265 §5.1.4 path-match: exact, or prefix followed
            # by '/' (a /foo cookie must not match /foobar).
            if not self._path_ok(c, path):
                continue
            pairs.append(c)
        if not pairs:
            return {}
        # Longest path first, per RFC 6265 §5.4.
        pairs.sort(key=lambda c: len(c.path), reverse=True)
        return {c.name: c.value for c in pairs}

    # ── helpers ───────────────────────────────────────────────────
    @staticmethod
    def _domain_ok(c: Cookie, host: str) -> bool:
        if c.host_only:
            return host == c.domain
        return host == c.domain or host.endswith("." + c.domain)

    @staticmethod
    def _path_ok(c: Cookie, path: str) -> bool:
        if path == c.path:
            return True
        if not path.startswith(c.path):
            return False
        return c.path.endswith("/") or path[len(c.path):len(c.path) + 1] == "/"

    def purge_expired(self) -> None:
        now = time.time()
        self.cookies = [c for c in self.cookies
                        if c.expires_at is None or c.expires_at > now]


def _selfcheck() -> None:
    j = CookieJar()
    now = time.time()

    # domain suffix validation (cookie tossing rejected)
    j.store_from_headers("example.com", [("set-cookie", "sid=1; Domain=evil.com")])
    assert j.dict_for("evil.com", "/") == {}, "tossed cookie must be rejected"
    # donsetch contract: a rejected Domain attr falls back to host-only
    # storage for the request host (RFC says drop; donsetch scopes safe)
    assert j.dict_for("example.com", "/")["sid"] == "1", "rejected-domain cookie stays host-only"
    j.store_from_headers("example.com", [("set-cookie", "sid=1; Domain=example.com; Path=/x")])
    assert "sid" in j.dict_for("example.com", "/x/a"), "subdomain cookie must match"
    assert "sid" in j.dict_for("sub.example.com", "/x"), "domain cookie must match subdomains"

    # host-only cookies
    j.store_from_headers("example.com", [("set-cookie", "h=1")])
    assert j.dict_for("example.com", "/")["h"] == "1"
    assert j.dict_for("www.example.com", "/") == {}, "host-only must not leak to subdomain"

    # path-match prefix rule: /foo must not match /foobar
    j.store_from_headers("path.test", [("set-cookie", "p=1; Path=/foo")])
    assert j.dict_for("path.test", "/foobar") == {}, "/foo cookie must not match /foobar"
    assert "p" in j.dict_for("path.test", "/foo/bar"), "prefix path must match"

    # longest-path-first ordering
    j.store_from_headers("path.test", [("set-cookie", "a=1; Path=/")])
    j.store_from_headers("path.test", [("set-cookie", "b=2; Path=/deep")])
    d = j.dict_for("path.test", "/deep/path")
    assert list(d.items())[0] == ("b", "2"), "longest path must sort first"

    # max-age expiry + replacement
    j.store_from_headers("example.com", [("set-cookie", f"e=1; Max-Age={int(-1)}")])
    assert "e" not in j.dict_for("example.com", "/"), "negative max-age must delete"
    j.store_from_headers("example.com", [("set-cookie", f"r=1; Max-Age={int(100)}")])
    j.store_from_headers("example.com", [("set-cookie", "r=2")])
    assert j.dict_for("example.com", "/")["r"] == "2", "same (name,domain,path) must replace"

    # control chars rejected
    j.store_from_headers("example.com", [("set-cookie", "x=1\r\nInjected: 1")])
    assert "x" not in j.dict_for("example.com", "/"), "control chars must reject cookie"

    # store_raw handoff (leading-dot subdomain)
    j.store_raw("cf_clearance", "tok", ".example.com", now + 60)
    assert j.dict_for("www.example.com", "/")["cf_clearance"] == "tok"
    assert j.dict_for("example.com", "/")["cf_clearance"] == "tok"
    j.store_raw("dead", "v", "example.com", now - 1)
    assert "dead" not in j.dict_for("example.com", "/"), "expired raw cookie must not replay"

    # snapshot filter: expiry applied, domain cookies included,
    # tracking cookies dropped by the caller's clearance filter
    snap = {c["name"] for c in j.snapshot_for("www.example.com")}
    assert "cf_clearance" in snap, f"snapshot missing clearance cookie: {snap}"
    assert "dead" not in snap, f"snapshot must skip expired cookies: {snap}"

    print("COOKIES-OK")


if __name__ == "__main__":
    _selfcheck()
