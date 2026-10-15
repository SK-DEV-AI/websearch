"""Grep MCP client (https://mcp.grep.app) — keyless code search over ~1M repos.

Why MCP, not REST: grep.app/api/search 429-challenges this egress (Vercel
bot-wall), but the StreamableHTTP MCP endpoint answers in <1s with no auth
and no observed rate limit (10-call burst clean 2026-10-15). Same index,
different door.

Protocol: MCP StreamableHTTP (SDK 2.x: streamable_http_client 2-tuple +
ClientSession). One session per search call — handshake adds ~1s, keeps the
code stateless and avoids cross-request session lifecycle bugs. Single tool:
searchGitHub(query, matchCase, matchWholeWords, useRegexp, repo, path,
language[]). Only query is required.

Response text format (parsed by _parse_hits):
  Repository: <owner/repo>
  Path: <file path>
  URL: <github blob url>
  License: <spdx|Unknown>
  Snippets:
  --- Snippet N (Line L) ---
  <code lines>
"""

from __future__ import annotations

import asyncio
import logging
import re
from typing import Any

logger = logging.getLogger("grep_mcp")

_MCP_URL = "https://mcp.grep.app"
_TIMEOUT_S = 30.0

_BLOCK_RE = re.compile(
    r"^Repository:\s*(?P<repo>.+?)\n"
    r"^Path:\s*(?P<path>.+?)\n"
    r"^URL:\s*(?P<url>\S+)",
    re.MULTILINE,
)
_SNIPPET_RE = re.compile(
    r"^--- Snippet \d+ \(Line (?P<line>\d+)\) ---\n(?P<body>.*?)(?=^--- Snippet |\Z)",
    re.MULTILINE | re.DOTALL,
)


def _parse_hits(text: str) -> list[dict]:
    """Parse searchGitHub text into [{repo, file, url, matches[{preview,line}]}].

    Blocks are split on 'Repository:' lines; each block's Snippets section
    yields match previews (first 3 lines, 200 chars each).
    """
    hits: list[dict] = []
    starts = [m.start() for m in re.finditer(r"^Repository:", text, re.MULTILINE)]
    starts.append(len(text))
    for i in range(len(starts) - 1):
        block = text[starts[i]:starts[i + 1]]
        m = _BLOCK_RE.search(block)
        if not m:
            continue
        matches = []
        for sm in _SNIPPET_RE.finditer(block):
            try:
                line = int(sm.group("line"))
            except ValueError:
                line = 0
            preview = " ".join(
                ln.strip() for ln in sm.group("body").strip().splitlines()
                if ln.strip())[:200]
            if preview:
                matches.append({"preview": preview, "line": line})
        hits.append({
            "repo": m.group("repo").strip(),
            "file": m.group("path").strip(),
            "url": m.group("url").strip(),
            "matches": matches,
        })
    return hits


async def grep_mcp_search(query: str, *, language: list[str] | None = None,
                          repo: str = "", path: str = "",
                          use_regexp: bool = False,
                          timeout: float = _TIMEOUT_S) -> dict[str, Any]:
    """Call searchGitHub on mcp.grep.app. Returns {success, hits|error}.

    Never raises — every failure is a dict (lane-safe for asyncio.gather).
    """
    if not (query or "").strip():
        return {"success": False, "error": "query required"}
    try:
        from mcp.client.streamable_http import streamable_http_client
        from mcp.client.session import ClientSession
    except ImportError as e:
        return {"success": False, "error": f"mcp SDK client missing: {e}"}
    args: dict[str, Any] = {"query": query.strip()}
    if language:
        args["language"] = language
    if repo:
        args["repo"] = repo
    if path:
        args["path"] = path
    if use_regexp:
        args["useRegexp"] = True
    try:
        async with asyncio.timeout(timeout):
            async with streamable_http_client(_MCP_URL) as (rs, ws):
                async with ClientSession(rs, ws) as s:
                    await s.initialize()
                    result = await s.call_tool("searchGitHub", args)
    except (asyncio.TimeoutError, TimeoutError):
        return {"success": False, "error": "grep MCP timeout"}
    except Exception as e:
        logger.debug("grep MCP call failed: %s", e)
        return {"success": False, "error": f"grep MCP: {type(e).__name__}: {e}"}
    try:
        texts = [c.text for c in (result.content or [])
                 if getattr(c, "type", "") == "text" and getattr(c, "text", "")]
    except Exception as e:
        return {"success": False, "error": f"grep MCP shape: {e}"}
    if getattr(result, "isError", False):
        return {"success": False,
                "error": "; ".join(texts)[:300] or "grep MCP tool error"}
    return {"success": True, "hits": _parse_hits("\n".join(texts))}
