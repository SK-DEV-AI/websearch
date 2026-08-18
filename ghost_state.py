"""Ghost state: domain-profile router + clearance-cookie vault.

Port of donsetch's self-improving fetch loop (cache.rs:108-587,
detect/walls.rs, server.rs verdict gate) onto our fetch stack.
Turns one successful browser solve into N cheap cookie-backed
tier-1 fetches, and stops CDP-rendered error shells from being
served as content.

State persists as JSON at ~/.cache/websearch/ghost_state.json
(atomic 0600 writes). See
session-root/work/donsetch/IMPLEMENTATION-PLAN.md items #3/#4/#5.
"""

import json
import logging
import os
import tempfile
import time
from pathlib import Path
from urllib.parse import urlsplit

logger = logging.getLogger("ghost_state")

STATE_DIR = Path.home() / ".cache" / "websearch"
STATE_FILE = STATE_DIR / "ghost_state.json"

# ── verdicts (donsetch detect/walls.rs:21-49) ─────────────────────
CONTENT_OK = "content_ok"
CHALLENGE = "challenge"
AUTH_WALL = "auth_wall"
PAYWALL = "paywall"
BLOCKED = "blocked"
SOFT_NOT_FOUND = "soft_not_found"
RATE_LIMITED = "rate_limited"
SERVER_ERROR = "server_error"

# terminal verdicts never escalate to a browser solve and never
# count against the warm-cookie path (cache.rs:418-424)
TERMINAL = {AUTH_WALL, PAYWALL, BLOCKED, SOFT_NOT_FOUND,
            RATE_LIMITED, SERVER_ERROR}

# ── clearance cookie filter (donsetch cookies.rs) ─────────────────
CLEARANCE_PREFIXES = (
    "cf_", "__cf", "datadome", "_dd_s", "ak_bmsc", "bm_sz", "bm_mi",
    "bm_sv", "perma-cookie", "incap_ses", "visid_incap", "nlbi_",
    "_tracker", "xp_", "pxhd", "svvid", "wrhst", "mc_", "_pc",
    "apt.uid", "bmsid", "authid", "akavpau", "sails-", "_mcid",
)
CLEARANCE_EXACT = {
    "__cf_bm": 7200, "cf_clearance": 1800, "cf_chl_rc_m": 600,
    "cf_chl_opt_m": 600,
}

# ── router constants (cache.rs) ───────────────────────────────────
COLD_CHECK_TTL = 3600.0   # a recent failed cold check → skip tier-1
COOKIE_FLOOR = 120.0      # min observed cookie lifetime
COOKIE_DEFAULT_TTL = 1800.0  # 30 min when lifetime is unlearned

# ── classifier markers ────────────────────────────────────────────
CHALLENGE_MARKERS = (
    "just a moment", "checking your browser", "cf-challenge",
    "__cf_chl_", "cf-turnstile", "verify you are human",
    "attention required", "checking if the site connection is secure",
    "datadome", "anubis", "captcha", "are you a robot",
)
AUTH_MARKERS = (
    "log in", "sign in", "login", "membership required",
    "you need to sign in", "please log in",
)
PAYWALL_MARKERS = (
    "subscribe to continue", "this article is for subscribers",
    "subscribe now", "you have reached your free", "free article limit",
    "sign in to continue reading", "this is a subscriber-only",
)
NOT_FOUND_MARKERS = ("page not found", "does not exist", "not found (404)",
                     "404 error", "404 not found", "no longer available")
BLOCKED_MARKERS = ("access denied", "network security", "request forbidden",
                   "your request has been blocked", "access blocked")

AUTH_STATUSES = {401, 402, 403}
NOT_FOUND_STATUSES = {404, 410}
RATE_LIMIT_STATUSES = {429}
SHORT_SHELL_MAX = 5000  # shells are short; big content is never a shell


def classify(status: int, content: str = "", title: str = "") -> str:
    """Classify a fetch outcome into a verdict.

    Status wins when it is decisive; otherwise DOM markers are
    matched against title (high weight) and the first 2000 chars of
    content. Short pages need marker evidence in the first 800 chars,
    long pages are trusted as real content.
    """
    text = ((content or "")[:2000]).lower()
    head = text[:800]
    title_l = (title or "").lower()

    if status in AUTH_STATUSES:
        return AUTH_WALL
    if status in NOT_FOUND_STATUSES:
        return SOFT_NOT_FOUND
    if status in RATE_LIMIT_STATUSES:
        return RATE_LIMITED
    if status >= 500:
        return SERVER_ERROR
    if any(m in text for m in CHALLENGE_MARKERS) or any(m in title_l for m in CHALLENGE_MARKERS):
        return CHALLENGE
    if any(m in title_l for m in AUTH_MARKERS) or (
            len(text) < SHORT_SHELL_MAX and any(m in head for m in AUTH_MARKERS)):
        return AUTH_WALL
    if any(m in title_l for m in PAYWALL_MARKERS) or (
            len(text) < SHORT_SHELL_MAX and any(m in head for m in PAYWALL_MARKERS)):
        return PAYWALL
    if any(m in title_l for m in NOT_FOUND_MARKERS) or (
            len(text) < SHORT_SHELL_MAX and any(m in head for m in NOT_FOUND_MARKERS)):
        return SOFT_NOT_FOUND
    if len(text) < SHORT_SHELL_MAX and any(m in head for m in BLOCKED_MARKERS):
        return BLOCKED
    return CONTENT_OK


def _filter_cookies(cookies: list[dict]) -> list[dict]:
    """Keep only clearance/anti-bot cookies worth replaying."""
    keep = []
    for c in cookies or []:
        name = c.get("name", "")
        if name in CLEARANCE_EXACT:
            keep.append(c)
        elif any(name.startswith(p) for p in CLEARANCE_PREFIXES):
            keep.append(c)
    return keep


class GhostState:
    """Per-host fetch profiles with dampened cookie-lifetime learning."""

    def __init__(self, path: Path = STATE_FILE):
        self._path = path
        self._profiles: dict[str, dict] = {}

    # ── persistence ────────────────────────────────────────────
    def load(self) -> None:
        try:
            data = json.loads(self._path.read_text())
            self._profiles = {k: v for k, v in data.items() if isinstance(v, dict)}
        except FileNotFoundError:
            self._profiles = {}
        except Exception as e:
            logger.warning("ghost_state: failed to load %s: %s", self._path, e)
            self._profiles = {}

    def save(self) -> None:
        try:
            STATE_DIR.mkdir(parents=True, exist_ok=True)
            fd, tmp = tempfile.mkstemp(dir=STATE_DIR, prefix="ghost_state.")
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(self._profiles, f, indent=1)
                    f.flush()
                    os.fsync(f.fileno())
                os.chmod(tmp, 0o600)
                os.replace(tmp, self._path)
            except BaseException:
                try:
                    os.unlink(tmp)
                except OSError:
                    pass
                raise
        except Exception as e:
            logger.warning("ghost_state: failed to save %s: %s", self._path, e)

    # ── helpers ────────────────────────────────────────────────
    @staticmethod
    def host_of(url: str) -> str:
        try:
            return (urlsplit(url).hostname or "").lower()
        except ValueError:
            return ""

    def _profile(self, host: str) -> dict:
        p = self._profiles.get(host)
        if p is None:
            p = self._profiles[host] = {
                "needs_tier2": False,
                "last_solved": 0.0,
                "last_cold_check": 0.0,
                "observed_lifetime": COOKIE_DEFAULT_TTL,
                "replay_ok": False,
                "warm_fails": 0,
                "solved_count": 0,
                "ok_count": 0,
                "fail_count": 0,
                "verdicts": {},
                "cookies": [],
            }
        return p

    def _vault_fresh_at(self, profile: dict) -> float:
        """Earliest expiry across vault cookies, trusting the learned
        observed_lifetime over server-expired values (cache.rs:205-232)."""
        if not profile.get("cookies"):
            return 0.0
        lifetime = float(profile.get("observed_lifetime", COOKIE_DEFAULT_TTL))
        learned_deadline = profile.get("last_solved", 0.0) + lifetime
        server_deadlines = [
            float(c.get("expires_at", 0.0)) for c in profile["cookies"]
            if c.get("expires_at")
        ]
        if server_deadlines:
            server_deadline = min(server_deadlines)
        else:
            server_deadline = 0.0
        deadline = min(learned_deadline, server_deadline) if server_deadline else learned_deadline
        return deadline if deadline > time.time() else 0.0

    # ── routing (cache.rs:386 route_for) ───────────────────────
    def route_for(self, host: str) -> str:
        """Cold | Warm | SkipToSolve | RecheckCold."""
        p = self._profile(host)
        if not p.get("needs_tier2"):
            return "cold"
        if self._vault_fresh_at(p) and p.get("replay_ok"):
            return "warm"
        if time.time() - p.get("last_cold_check", 0.0) < COLD_CHECK_TTL:
            return "skip_to_solve"
        return "recheck_cold"

    # ── observation (cache.rs:418-554) ─────────────────────────
    def record_fetch(self, host: str, verdict: str) -> None:
        """Move counters only. Non-challenge verdicts never set
        needs_tier2 (cache.rs:418-424); a content_ok means the wall
        went away."""
        p = self._profile(host)
        p["verdicts"][verdict] = p["verdicts"].get(verdict, 0) + 1
        if verdict == CONTENT_OK:
            p["ok_count"] += 1
            p["needs_tier2"] = False
            p["replay_ok"] = False
            p["warm_fails"] = 0
        else:
            p["fail_count"] += 1
            if verdict == CHALLENGE:
                p["needs_tier2"] = True
            # stamp last_cold_check so a wall we could not beat routes
            # to skip_to_solve (straight to browser) on the next fetch
            if verdict in TERMINAL or verdict == CHALLENGE:
                p["last_cold_check"] = time.time()
        self.save()

    def record_solved(self, host: str, cookies: list[dict],
                      replay_ok: bool = False) -> None:
        """Browser solved the wall: store clearance cookies, mark the
        domain tier-2 and arm the replay gate."""
        p = self._profile(host)
        now = time.time()
        p["needs_tier2"] = True
        p["last_solved"] = now
        p["last_cold_check"] = now
        p["solved_count"] += 1
        p["replay_ok"] = replay_ok
        p["warm_fails"] = 0
        p["cookies"] = _filter_cookies(cookies)
        if not p["cookies"]:
            # no usable clearance → lifetime learning is moot
            p["replay_ok"] = False
        self.save()

    def warm_ok(self, host: str) -> None:
        """A tier-1 fetch with vault cookies returned real content."""
        p = self._profile(host)
        p["warm_fails"] = 0
        p["replay_ok"] = True

    def record_warm_stale(self, host: str) -> None:
        """A warm tier-1 fetch failed. First failure is tolerated as
        transient; the second consecutive one learns the observed
        cookie lifetime (floored at 120s) and clears the vault
        (cache.rs:501-519)."""
        p = self._profile(host)
        p["warm_fails"] = p.get("warm_fails", 0) + 1
        if p["warm_fails"] >= 2 and p.get("last_solved"):
            learned = max(time.time() - p["last_solved"], COOKIE_FLOOR)
            p["observed_lifetime"] = learned
            p["cookies"] = []
            p["replay_ok"] = False
            logger.info("ghost_state: %s cookies stale after %.0fs, vault cleared",
                        host, learned)
        self.save()

    def set_replay_ok(self, host: str, ok: bool) -> None:
        p = self._profile(host)
        p["replay_ok"] = ok
        if ok:
            p["warm_fails"] = 0
        self.save()

    # ── handoff ────────────────────────────────────────────────
    def vault(self, host: str) -> dict[str, str]:
        """Fresh clearance cookies as {name: value} for tier-1 retry."""
        p = self._profile(host)
        if not self._vault_fresh_at(p):
            return {}
        return {c["name"]: c["value"] for c in p.get("cookies", [])}


ghost = GhostState()
ghost.load()
