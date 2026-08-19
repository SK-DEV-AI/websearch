"""Per-session coherent fingerprint bundle (Camoufox methodology steal).

One matched UA/platform/locale/timezone set per session seed, so stealth
knobs never contradict each other (over-spoofing with mismatched pieces
burns sessions). Bundles are real observed browser profiles, not invented
combos. No browser binary, no per-call jitter — coherence over variety.

See session-root/work/mcp-opt/steal-antibots.md #3.
"""

import hashlib

_BUNDLES = [
    {
        "name": "win-chrome",
        "useragent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/126.0.0.0 Safari/537.36"),
        "platform": "Win32",
        "locale": "en-US",
        "timezone_id": "America/New_York",
    },
    {
        "name": "win-edge",
        "useragent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/126.0.0.0 Safari/537.36 "
                      "Edg/126.0.0.0"),
        "platform": "Win32",
        "locale": "en-US",
        "timezone_id": "Europe/London",
    },
    {
        "name": "mac-chrome",
        "useragent": ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/126.0.0.0 Safari/537.36"),
        "platform": "MacIntel",
        "locale": "en-US",
        "timezone_id": "America/Los_Angeles",
    },
    {
        "name": "linux-chrome",
        "useragent": ("Mozilla/5.0 (X11; Linux x86_64) "
                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                      "Chrome/126.0.0.0 Safari/537.36"),
        "platform": "Linux x86_64",
        "locale": "en-US",
        "timezone_id": "Europe/Berlin",
    },
]


def bundle_for(seed: str) -> dict:
    """Stable bundle for *seed* — same seed always gets the same set."""
    digest = hashlib.md5(seed.encode()).digest()
    return _BUNDLES[int.from_bytes(digest[:4], "big") % len(_BUNDLES)]


def bundle_fills(seed: str, locale: str = "", timezone_id: str = "",
                 useragent: str = "") -> dict:
    """Coherent fill values for empty knobs; explicit per-call values win."""
    b = bundle_for(seed)
    return {
        "locale": locale or b["locale"],
        "timezone_id": timezone_id or b["timezone_id"],
        "useragent": useragent or b["useragent"],
    }