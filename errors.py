"""Structured error contract — donsetch mcp/server.rs:1691-1800 port.

Every tool failure carries errorKind ∈ {permanent, transient, walled}
plus a one-line next_action hint derived from verdict + status, so the
agent can decide its fallback without parsing prose.
"""

from __future__ import annotations

import socket

PERMANENT = "permanent"
TRANSIENT = "transient"
WALLED = "walled"

VERDICT_TO_KIND = {
    "challenge": WALLED,        # browser tier already tried server-side → walled
    "auth_wall": WALLED,
    "paywall": WALLED,
    "soft_not_found": PERMANENT,
    "blocked": PERMANENT,
    "rate_limited": TRANSIENT,
    "server_error": TRANSIENT,
    "content_ok": PERMANENT,    # unreachable in an error path; belt-and-braces
}

TRANSIENT_STATUSES = {429, 503, 502, 504, 408}
TRANSIENT_ERR_SUBSTR = (
    "timed out", "timeout", "connection", "dns", "resolve",
    "refused", "unreachable", "reset", "broken pipe", "eof",
)


def classify(verdict: str | None = None, status: int = 0,
             error: str = "") -> str:
    """Map verdict/status/error text → errorKind (donsetch verdict_kind +
    fetch_error_kind)."""
    if verdict and verdict in VERDICT_TO_KIND:
        kind = VERDICT_TO_KIND[verdict]
    elif status in TRANSIENT_STATUSES:
        kind = TRANSIENT
    else:
        kind = PERMANENT
    if kind == PERMANENT:
        el = (error or "").lower()
        if any(s in el for s in TRANSIENT_ERR_SUBSTR):
            kind = TRANSIENT
    return kind


def next_action(verdict: str | None = None, status: int = 0,
                kind: str = PERMANENT) -> str:
    """One-line actionable guidance (donsetch next_action_for)."""
    if verdict == "auth_wall":
        return ("requires login credentials — no keyless automated path; "
                "use an interactive browser with your session")
    if verdict == "paywall":
        return ("paid content — no automated path; look for an open "
                "preprint/copy via web_search")
    if verdict == "soft_not_found":
        return ("verify the URL (typo? deleted page?) — or web_search the "
                "page title to find the moved copy")
    if verdict == "challenge":
        if kind == WALLED:
            return ("browser solve failed — interactive verification needed; "
                    "no automated path")
        return ("retry with the browser path — it solves most JS/cookie "
                "challenges")
    if verdict == "blocked":
        if status == 429:
            return "rate limited — wait 30-60s and retry"
        if status == 403:
            return ("access denied — retry later or from a different "
                    "network; this server refuses bots")
        return "server rejected the request — retrying later sometimes works"
    if verdict == "rate_limited":
        return "rate limited — wait 30-60s and retry"
    if verdict == "server_error":
        return "server error — safe to retry after a short wait"
    if kind == TRANSIENT:
        return "transient network failure — safe to retry immediately"
    if kind == WALLED:
        return ("no extractable content behind the wall — use an "
                "interactive agent browser for this site")
    return "check the URL and retry; if repeated, the site may be down or blocking"


def annotate(data: dict) -> dict:
    """Stamp errorKind + next_action onto a failure dict in place.
    Derives from the dict's own verdict/status/error fields when present."""
    verdict = data.get("verdict")
    status = int(data.get("status") or 0)
    error = data.get("error") or data.get("message") or ""
    kind = classify(verdict, status, error)
    data.setdefault("errorKind", kind)
    data.setdefault("next_action", next_action(verdict, status, kind))
    return data


def classify_exception(e: Exception) -> str:
    """Exception → errorKind (donsetch fetch_error_kind)."""
    if isinstance(e, (TimeoutError, socket.timeout, ConnectionError,
                      OSError)):
        return TRANSIENT
    msg = str(e).lower()
    if any(s in msg for s in TRANSIENT_ERR_SUBSTR):
        return TRANSIENT
    return PERMANENT


if __name__ == "__main__":
    assert classify(verdict="auth_wall") == WALLED
    assert classify(verdict="paywall") == WALLED
    assert classify(verdict="challenge") == WALLED
    assert classify(verdict="soft_not_found") == PERMANENT
    assert classify(verdict="rate_limited") == TRANSIENT
    assert classify(status=429) == TRANSIENT
    assert classify(status=403) == PERMANENT
    assert classify(error="Connection timed out") == TRANSIENT
    assert classify(error="dns resolution failed") == TRANSIENT
    assert classify() == PERMANENT
    assert classify_exception(TimeoutError()) == TRANSIENT
    assert classify_exception(ConnectionResetError()) == TRANSIENT
    assert classify_exception(ValueError("boom")) == PERMANENT
    assert "browser" in next_action(verdict="auth_wall")
    assert "30-60s" in next_action(verdict="blocked", status=429)
    assert "preprint" in next_action(verdict="paywall")
    assert "moved copy" in next_action(verdict="soft_not_found")
    d = annotate({"success": False, "error": "boom"})
    assert d["errorKind"] == PERMANENT and "check the URL" in d["next_action"]
    d2 = annotate({"success": False, "verdict": "rate_limited", "status": 429,
                   "error": "slow down"})
    assert d2["errorKind"] == TRANSIENT and "30-60s" in d2["next_action"]
    print("ERRORS-OK")