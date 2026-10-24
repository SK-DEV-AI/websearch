"""Lane preflight — report which capabilities are live and how to enable the rest.

Why this exists: the expensive lanes (Google AI Mode, browser_action, CDP
bot-wall bypass, GPU reranking) all depend on things a fresh install does not
have. Without a preflight, a new user tries the flagship feature, gets nothing
useful back, and concludes the package is broken. Every degraded lane must say
so, name the exact thing that is missing, and give the command that fixes it.

Report shape (stable — the doctor tool returns this verbatim):

    {
      "summary": {"live": n, "needs_setup": n, "off": n},
      "lanes": [ {id, name, status, requires, how_to_enable, note} ],
      "env":   {cdp_url, reranker_path, keys_present: [...]}   # names only
    }

`status` is one of:
    "live"        usable right now
    "needs_setup"  the dependency is absent but the user can fix it
    "off"         deliberately disabled (killswitch / not applicable)

Never reports key VALUES — only which env var names are set.
"""

from __future__ import annotations

import os
import shutil
from typing import Any

import httpx

from config import HELIUM_CDP

# Env vars that upgrade existing lanes. Names only, never values.
_OPTIONAL_KEYS = {
    "GROQ_API_KEYS": "LLM synthesis + query rewriting (answers with citations)",
    "TAVILY_KEYS": "Tavily search lane (a real API-backed engine)",
    "NV_KEY": "NVIDIA embeddings used for semantic ranking",
    "SE_KEY": "Stack Exchange: 10k/day instead of 300",
    "S2_KEY": "Semantic Scholar: 1 RPS instead of shared pool",
    "GITHUB_TOKEN": "GitHub code search (codesearch server)",
}

# Chromium-family binaries we can point at CDP. Order = preference.
_CHROMIUM_CANDIDATES = (
    "helium-browser", "helium", "chromium", "chromium-browser", "google-chrome",
    "google-chrome-stable", "brave", "brave-browser", "microsoft-edge",
    "vivaldi", "thorium",
)

_LAUNCH_HINT = (
    'chromium --remote-debugging-port=9222 --user-data-dir="$HOME/.mcp-browser"'
)


def _which_chromium() -> str | None:
    for name in _CHROMIUM_CANDIDATES:
        p = shutil.which(name)
        if p:
            return p
    return None


async def _cdp_probe(cdp_url: str, timeout: float = 3.0) -> dict[str, Any]:
    """Hit /json/version. Returns {ok, browser, error}."""
    try:
        async with httpx.AsyncClient(timeout=timeout) as c:
            r = await c.get(cdp_url.rstrip("/") + "/json/version")
            if r.status_code != 200:
                return {"ok": False, "error": f"HTTP {r.status_code}"}
            d = r.json()
            return {"ok": True, "browser": d.get("Browser", "unknown")}
    except Exception as e:  # noqa: BLE001 - any failure means "no browser"
        return {"ok": False, "error": f"{type(e).__name__}: {e}"}


def _reranker_worker() -> str:
    """Resolve the reranker worker path the same way reranker.py does."""
    return os.environ.get(
        "RERANKER_WORKER",
        os.path.expanduser("~/.local/share/reranker-rust/worker.py"),
    )


def _killswitch() -> str:
    return os.environ.get(
        "RERANKER_KILLSWITCH",
        os.path.expanduser("~/.local/share/reranker-rust/disabled"),
    )


async def preflight(probe_network: bool = True) -> dict[str, Any]:
    """Build the lane report. `probe_network=False` skips outbound HTTP checks."""
    cdp = await _cdp_probe(HELIUM_CDP)
    chromium = _which_chromium()
    worker = _reranker_worker()
    killswitch = _killswitch()
    reranker_off = os.path.exists(killswitch)
    worker_present = os.path.isfile(worker)
    keys_present = [k for k in _OPTIONAL_KEYS if os.environ.get(k, "").strip()]

    lanes: list[dict[str, Any]] = []

    # --- keyless HTTP lanes: the part that works anywhere, including servers ---
    net_ok = True
    net_note = "reachable"
    if probe_network:
        net_ok = await _net_probe()
        if not net_ok:
            net_note = "outbound HTTPS failed — all HTTP lanes will fail"
    lanes.append({
        "id": "keyless_http",
        "name": "Keyless search + fetch (search, fetch, wikipedia, arxiv, map_site, pdf_extract, codesearch x17)",
        "status": "live" if net_ok else "off",
        "requires": "outbound HTTPS only. No API keys, no browser.",
        "how_to_enable": "Check network/proxy/DNS if off.",
        "note": net_note,
    })

    # --- CDP browser lanes ---
    browser_lanes = (
        "Google AI Mode (search(google_ai_only=True))",
        "browser_action (click/type/evaluate/screenshot flows)",
        "CDP fallback for bot-walled pages",
    )
    if cdp.get("ok"):
        lanes.append({
            "id": "cdp_browser",
            "name": " + ".join(browser_lanes),
            "status": "live",
            "requires": "a Chromium-based browser with remote debugging, same host.",
            "how_to_enable": "Already connected.",
            "note": f"connected: {cdp.get('browser')} at {HELIUM_CDP}",
        })
        gai_status = "live"
        gai_note = (
            "Reachable. A Google account must be signed in in that browser profile "
            "for Google AI Mode to answer; if it is not, expect a sign-in page."
        )
    else:
        launch = chromium or _LAUNCH_HINT
        if chromium:
            how = (
                f'Start a Chromium with remote debugging, then sign in to Google once:\n'
                f'  {launch} --remote-debugging-port=9222 --user-data-dir="$HOME/.mcp-browser"\n'
                f'Then set WEBSEARCH_CDP_URL if you used a different port.'
            )
        else:
            how = (
                "No Chromium-family browser found on PATH. Install one "
                "(chromium, google-chrome, brave, or Helium), then:\n"
                f'  {_LAUNCH_HINT}'
            )
        lanes.append({
            "id": "cdp_browser",
            "name": " + ".join(browser_lanes),
            "status": "needs_setup",
            "requires": "a Chromium-based browser with remote debugging, same host.",
            "how_to_enable": how,
            "note": (
                f"not connected at {HELIUM_CDP} ({cdp.get('error')}). These lanes are "
                "LOCAL-ONLY by design: a headless server has no signed-in browser, so "
                "they stay unavailable there. Everything else still works."
            ),
        })
        gai_status = "needs_setup"
        gai_note = "Depends on cdp_browser above."

    lanes.append({
        "id": "google_ai_mode",
        "name": "Google AI Mode answer + citations",
        "status": gai_status,
        "requires": "cdp_browser + a Google account signed in in that profile.",
        "how_to_enable": "Enable cdp_browser, then open google.com in it and sign in.",
        "note": gai_note,
    })

    # --- reranker ---
    if reranker_off:
        lanes.append({
            "id": "reranker",
            "name": "Cross-encoder reranking (better result order)",
            "status": "off",
            "requires": "the reranker worker + its model; optional.",
            "how_to_enable": f"Remove the killswitch: rm {killswitch}",
            "note": "Deliberately disabled. Results fall back to score ordering.",
        })
    elif worker_present:
        lanes.append({
            "id": "reranker",
            "name": "Cross-encoder reranking (better result order)",
            "status": "live",
            "requires": "the reranker worker + its model; optional.",
            "how_to_enable": "Already installed.",
            "note": f"worker: {worker}",
        })
    else:
        lanes.append({
            "id": "reranker",
            "name": "Cross-encoder reranking (better result order)",
            "status": "needs_setup",
            "requires": (
                "an optional local cross-encoder worker (separate download, ~500MB "
                "model, GPU recommended). Not bundled — it is not part of the pip "
                "package."
            ),
            "how_to_enable": (
                "Optional. Set RERANKER_WORKER=/path/to/worker.py if you have one. "
                "Without it results are ordered by blended score instead — still "
                "usable, just less precise."
            ),
            "note": (
                f"no worker at {worker}. Degrades gracefully: ranking is score-based, "
                "not model-based. Search still works."
            ),
        })

    # --- optional keys ---
    for var, why in _OPTIONAL_KEYS.items():
        if var == "GITHUB_TOKEN":
            continue  # belongs to the codesearch server
        have = var in keys_present
        lanes.append({
            "id": f"key_{var.lower()}",
            "name": why,
            "status": "live" if have else "off",
            "requires": f"optional env var {var}",
            "how_to_enable": f"export {var}=<your key> (free tier is usually enough)",
            "note": "set" if have else "not set — lane skipped, everything else unaffected",
        })

    summary = {
        "live": sum(1 for l in lanes if l["status"] == "live"),
        "needs_setup": sum(1 for l in lanes if l["status"] == "needs_setup"),
        "off": sum(1 for l in lanes if l["status"] == "off"),
    }
    return {
        "summary": summary,
        "lanes": lanes,
        "env": {
            "cdp_url": HELIUM_CDP,
            "chromium_on_path": chromium,
            "reranker_worker": worker,
            "keys_present": keys_present,
        },
        "where_am_i": (
            "local" if cdp.get("ok") else
            "server/headless or no browser — browser lanes unavailable, HTTP lanes fine"
        ),
    }


async def _net_probe() -> bool:
    """Cheap connectivity check used by preflight."""
    try:
        async with httpx.AsyncClient(timeout=4.0) as c:
            r = await c.get("https://1.1.1.1", follow_redirects=False)
            return r.status_code < 500
    except Exception:  # noqa: BLE001
        return False
