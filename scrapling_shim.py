"""Lazy scrapling access.

scrapling pulls a browser stack and is a large optional dependency, so it is
NOT imported at module import time. This shim keeps the import inside the
first call that actually needs it, and raises a clear, actionable error if the
extras were not installed.

Install with:  pip install websearch-mcp[scraping]
"""

from __future__ import annotations

from typing import Any

_SCRAPLING_ERR = (
    "scrapling is not installed. It is an optional dependency of this MCP "
    "server. Install it with:  pip install websearch-mcp[scraping]  "
    "(or websearch-mcp[all] for every extra)."
)


def _fetchers() -> Any:
    """Return the scrapling fetchers module, importing it on first use."""
    try:
        from scrapling.fetchers import AsyncFetcher, AsyncStealthySession
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError(_SCRAPLING_ERR) from e
    return AsyncFetcher, AsyncStealthySession


class _LazyFetcher:
    """Proxy that resolves AsyncFetcher on first attribute access."""

    def __getattr__(self, name: str) -> Any:
        AsyncFetcher, _ = _fetchers()
        return getattr(AsyncFetcher, name)


class _LazyStealthySession:
    """Proxy that resolves AsyncStealthySession on first use."""

    def __getattr__(self, name: str) -> Any:
        _, AsyncStealthySession = _fetchers()
        return getattr(AsyncStealthySession, name)


AsyncFetcher = _LazyFetcher()
AsyncStealthySession = _LazyStealthySession()
