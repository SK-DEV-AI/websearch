"""grep.app + Sourcegraph code-content lane (proposal 7).

Fills the true code-content gap: the GitHub vertical only does repos/issues.
grep.app is the primary (keyless GET, regex/RE2, ~1M repos); Sourcegraph
public GraphQL (no-auth public search) is the backup. Bot-walled from some
egress, so backup-only and advisory: every failure returns [], never raises.
"""

from __future__ import annotations

import asyncio
import logging

logger = logging.getLogger("code_vertical")

_GREP_API = "https://grep.app/api/search"
_SG_GRAPHQL = "https://sourcegraph.com/.api/graphql"
_TIMEOUT = 12.0
_SG_TIMEOUT = 30.0
_BACKOFF_S = (1.0, 3.0)


def _shape_grep(rank: int, hit: dict) -> dict | None:
    repo = (hit.get("repo") or {}).get("raw", "") or ""
    path = (hit.get("path") or {}).get("raw", "") or ""
    if not repo or not path:
        return None
    branch = "HEAD"
    url = f"https://github.com/{repo}/blob/{branch}/{path.lstrip('/')}"
    snippet = ((hit.get("content") or {}).get("snippet") or "").strip()
    return {"title": f"{repo}: {path}", "url": url,
            "snippet": (snippet or "code match")[:400],
            "published": None, "rank": rank}


def _shape_sg(rank: int, m: dict) -> dict | None:
    repo = (m.get("repository") or {}).get("name", "") or ""
    f = m.get("file") or {}
    path = f.get("path", "") or ""
    if not repo or not path:
        return None
    line = (f.get("content") or "").strip()[:400]
    if not line:
        for sym in (f.get("symbols") or [])[:1]:
            nm = (sym.get("name") or "").strip() if isinstance(sym, dict) else ""
            if nm:
                line = f"symbol: {nm}"[:400]
                break
    url = (f.get("url", "") or "").strip()
    if url.startswith("/"):
        url = "https://sourcegraph.com" + url
    elif not url.startswith("http"):
        url = f"https://github.com/{repo}/blob/HEAD/{path.lstrip('/')}"
    return {"title": f"{repo}: {path}", "url": url,
            "snippet": line or "code match", "published": None, "rank": rank}


async def _grep_search(query: str, count: int) -> list[dict]:
    """grep.app via MCP endpoint (mcp.grep.app searchGitHub).

    The REST API (/api/search) 429-challenges this egress; the MCP
    endpoint answers in <1s with no auth. Same index, different door.
    """
    try:
        from grep_mcp import grep_mcp_search
        r = await grep_mcp_search(query, timeout=30)
        if not r.get("success"):
            return []
        out: list[dict] = []
        for h in (r.get("hits") or [])[:count]:
            shaped = _shape_grep_mcp(len(out), h)
            if shaped:
                out.append(shaped)
        return out
    except Exception as e:
        logger.debug("grep MCP search failed: %s", e)
        return []


def _shape_grep_mcp(rank: int, h: dict) -> dict | None:
    repo = h.get("repo", "") or ""
    path = h.get("file", "") or ""
    if not repo or not path:
        return None
    url = h.get("url", "") or f"https://github.com/{repo}/blob/HEAD/{path.lstrip('/')}"
    snippet = "; ".join(m.get("preview", "") for m in h.get("matches", [])[:3])
    return {"title": f"{repo}: {path}", "url": url,
            "snippet": (snippet or "code match")[:400],
            "published": None, "rank": rank}


async def _grep_search_rest(query: str, count: int) -> list[dict]:
    """Legacy REST path — kept as last resort (bot-walled, usually 429)."""
    try:
        from config import get_http_client
        c = get_http_client()
        out: list[dict] = []
        for attempt in range(2):
            try:
                r = await c.get(
                    _GREP_API,
                    params={"q": query},
                    timeout=_TIMEOUT,
                    headers={"User-Agent": "websearch-mcp/1.0",
                             "Accept": "application/json"},
                )
                if r.status_code == 429 and attempt == 0:
                    await asyncio.sleep(_BACKOFF_S[attempt])
                    continue
                if r.status_code != 200:
                    return []
                v = r.json()
                break
            except Exception as e:
                logger.debug("grep.app attempt %d failed: %s", attempt, e)
                if attempt == 0:
                    await asyncio.sleep(_BACKOFF_S[attempt])
                    continue
                return []
        hits = v.get("hits", {}).get("hits", []) if isinstance(v, dict) else []
        for h in hits[:count]:
            shaped = _shape_grep(len(out), h.get("_source", h))
            if shaped:
                out.append(shaped)
        return out
    except Exception as e:
        logger.debug("grep.app search failed: %s", e)
        return []


_SG_Q = """query($q: String!) {
  search(query: $q, version: V3, patternType: keyword) {
    results { results { __typename
      ... on FileMatch { repository { name } file { path url content } }
    } }
  }
}"""


async def _sg_search(query: str, count: int) -> list[dict]:
    try:
        from config import get_http_client
        c = get_http_client()
        q = f"context:global count:{min(max(count, 1), 50)} {query} select:file"
        r = await c.post(
            _SG_GRAPHQL,
            json={"query": _SG_Q, "variables": {"q": q}},
            timeout=_SG_TIMEOUT,
            headers={"User-Agent": "websearch-mcp/1.0",
                     "Content-Type": "application/json",
                     "Accept": "application/json"},
        )
        if r.status_code != 200:
            return []
        v = r.json()
        res = (((v.get("data") or {}).get("search") or {}).get("results")
               or {}).get("results", [])
        out: list[dict] = []
        for m in res:
            if not isinstance(m, dict) or m.get("__typename") != "FileMatch":
                continue
            shaped = _shape_sg(len(out), m)
            if shaped:
                out.append(shaped)
            if len(out) >= count:
                break
        return out
    except Exception as e:
        logger.debug("sourcegraph search failed: %s", e)
        return []


async def search_code(query: str, count: int = 5) -> list[dict]:
    """grep.app MCP primary, Sourcegraph backup on short primary. Dedupe by URL."""
    primary = await _grep_search(query, count)
    out = list(primary)
    if len(out) < count:
        seen = {h["url"] for h in out}
        for h in await _sg_search(query, count):
            if h["url"] not in seen:
                seen.add(h["url"])
                h["rank"] = len(out)
                out.append(h)
            if len(out) >= count:
                break
    return out[:count]


if __name__ == "__main__":
    async def check():
        hits = await search_code("rate limit retry-after backoff")
        print(f"code vertical: {len(hits)} hits")
        for h in hits[:2]:
            print(" -", h["title"][:90], "|", h["url"][:110])

    asyncio.run(check())
