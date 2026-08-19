"""Stealth init-script injection for the shared CDP browser (theme A2).

Wraps tf-playwright-stealth 1.2.0's pure-JS script bundle so it can be injected
per-tab via Page.addScriptToEvaluateOnNewDocument — no new windows, no browser
restart, CDP-agnostic (the bundle is a plain JS string, no Playwright objects).

Two paths: primary combine_scripts() call, fallback that reads the 17 js/*.js
files directly so a future removal of the playwright python dep does not break
injection. The user-agent is kept real (Helium Chrome) — only client-hint /
accept-* headers are spoofed.
"""

from __future__ import annotations


def build_stealth_init_script() -> str:
    try:
        from playwright_stealth.stealth import combine_scripts
        from playwright_stealth.core import StealthConfig
        from playwright_stealth.properties import BrowserType, Properties

        return combine_scripts(Properties(browser_type=BrowserType.CHROME), StealthConfig())
    except Exception:
        import os

        import playwright_stealth
        from playwright_stealth.core._stealth_config import StealthConfig as C
        from playwright_stealth.properties import BrowserType, Properties

        jsdir = os.path.join(os.path.dirname(playwright_stealth.__file__), "js")
        names = C().enabled_scripts(Properties(browser_type=BrowserType.CHROME))
        parts = [open(os.path.join(jsdir, n), encoding="utf-8").read()
                 for n in names if isinstance(n, str)]
        return "\n".join(parts)


def stealth_headers() -> dict:
    """Spoofed client-hint/accept headers; real Helium UA preserved."""
    try:
        from playwright_stealth.properties import BrowserType, Properties

        h = Properties(browser_type=BrowserType.CHROME).as_dict().get("header", {})
    except Exception:
        h = {}
    h.pop("user-agent", None)
    return h
