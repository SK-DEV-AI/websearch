"""grep.app + Sourcegraph code-content lane (proposal 7).

Fills the true code-content gap: the GitHub vertical only does repos/issues.
grep.app is the primary (keyless GET, regex/RE2, ~1M repos); Sourcegraph
public GraphQL (no-auth public search) is the backup. Bot-walled from some
egress, so backup-only and advisory: every failure returns [], never raises.
"""

from __future__ import annotations

import asyncio
import logging
import re

from config import get_http_client
from grep_mcp import grep_mcp_search

logger = logging.getLogger("code_vertical")

_GREP_API = "https://grep.app/api/search"
_SG_GRAPHQL = "https://sourcegraph.com/.api/graphql"
_TIMEOUT = 12.0
_SG_TIMEOUT = 30.0
_BACKOFF_S = (1.0, 3.0)

# Sourcegraph server-side time budget: count:50 blows our 30s timeout and
# returns nothing (measured 2026-10-20). count:5..count:10 answer in ~2.4s.
SG_COUNT_MAX = 10
SG_RETRY_DELAY_S = 2.0
# research.py gives the code vertical a 35s wall-clock budget. A Sourcegraph
# attempt can take up to _SG_TIMEOUT (30s), so the backup lane must be
# bounded well below that budget — otherwise the Sourcegraph retry cancels
# the whole vertical and the grep.app hits are lost with it.
SG_BUDGET_S = 22.0


# ── code-shaped query construction ──────────────────────────────────
# grep.app's MCP applies AND semantics across all words: a 4+ word
# natural-language query matches NOTHING (measured live 2026-10-20:
# "python httpx async client timeout retry" -> 2 hits, "httpx timeout" -> 2
# hits but the NL form explodes). So we shorten the query before handing it
# to the index: keep the most code-significant 2-3 terms, drop NL filler.

# Natural-language filler: articles, prepositions, question scaffolding,
# generic verbs and "content" nouns. Dropped outright.
_CODE_STOPWORDS = {
    "a", "an", "the", "this", "that", "these", "those", "it", "its",
    "in", "on", "of", "for", "to", "from", "into", "with", "without", "by",
    "and", "or", "but", "if", "then", "than", "as", "at", "is", "are", "was",
    "were", "be", "been", "being", "do", "does", "did", "done",
    "how", "what", "why", "who", "whom", "whose", "when", "where", "which",
    "i", "we", "you", "your", "my", "our", "me", "us", "can", "could",
    "should", "would", "will", "shall", "may", "might", "must", "want",
    "need", "please", "help", "use", "using", "used", "make", "making",
    "create", "creating", "build", "building", "add", "adding", "get",
    "setting", "set", "configure", "configuring", "configuration", "implement",
    "implementation", "implementing", "explain", "explained", "understand",
    "understanding", "best", "good", "new", "old", "latest", "top", "most",
    "tutorial", "guide", "guides", "example", "examples", "docs",
    "documentation", "learn", "learning", "way", "ways", "thing", "things",
    "code", "coding", "program", "programming", "problem", "solution",
    "issue", "issues", "error", "errors", "question", "answer", "blog",
    "article", "post", "page", "site", "website", "link", "links",
}

# Common technical / English words that carry little index signal on their
# own: they rank BELOW a distinctive identifier ("useeffect", "httpx").
_CODE_GENERIC = {
    "python", "java", "javascript", "typescript", "rust", "golang", "ruby",
    "php", "kotlin", "swift", "scala", "elixir", "haskell", "clojure",
    "react", "vue", "angular", "svelte", "node", "nodejs", "nextjs",
    "nuxt", "django", "flask", "fastapi", "rails", "spring", "laravel",
    "async", "await", "sync", "client", "server", "request", "response",
    "timeout", "retry", "retries", "cleanup", "settings", "config", "value",
    "values", "key", "keys", "data", "file", "files", "path", "name",
    "type", "types", "function", "method", "class", "object", "string",
    "number", "list", "array", "map", "set", "get", "post", "put", "delete",
    "http", "https", "api", "url", "json", "yaml", "xml", "sql", "db",
    "database", "cache", "queue", "log", "logs", "error", "exception",
    "test", "tests", "mock", "stub", "unit", "integration", "end", "start",
    "run", "call", "calls", "input", "output", "result", "results", "state",
    "status", "user", "users", "auth", "token", "tokens", "password",
    "secret", "header", "headers", "body", "content", "page", "size",
    "rate", "limit", "backoff", "sleep", "loop", "handle", "handler",
    "middleware", "hook", "hooks", "context", "manager", "service",
    "module", "package", "library", "framework", "version", "update",
    "install", "setup", "environment", "env", "dev", "prod", "deploy",
}


def _is_code_shaped(token: str) -> bool:
    """True for identifier-ish tokens: dotted, underscored, camelCase,
    bracketed, namespaced or otherwise not plain prose."""
    if not token:
        return False
    if any(ch in token for ch in "._()[]{}<>/\\:@#$&*!?"):
        return True
    if re.search(r"[a-z][A-Z]", token):
        return True
    if re.search(r"\d", token) and re.search(r"[A-Za-z]", token):
        return True
    if token.isupper() and len(token) > 1:
        return True
    return False


def _code_query(query: str) -> str:
    """Shorten a natural-language query into a code-shaped grep.app query.

    grep.app's MCP search applies AND semantics across every word, so a
    natural-language query ("python httpx async client timeout retry")
    matches almost nothing. This picks the most code-significant 2-3
    terms: identifiers and distinctive unknown tokens first, then generic
    technical words, dropping natural-language filler entirely.

    A query that is ALREADY a clean symbol/identifier ("handleAuth",
    "foo.bar()", "get_http_client") is returned unchanged.

    Deterministic, no LLM, no network.
    """
    q = (query or "").strip()
    if not q:
        return q
    raw = q.split()
    # Already a clean identifier / symbol / call -> pass through untouched.
    if len(raw) == 1 and _is_code_shaped(raw[0]):
        return q
    scored: list[tuple[int, int, int, str]] = []
    for i, tok in enumerate(raw):
        core = tok.strip(".,;:!?\"'()[]{}")
        if not core:
            continue
        low = core.lower()
        if low in _CODE_STOPWORDS:
            continue
        if _is_code_shaped(core):
            score = 4
        elif low in _CODE_GENERIC:
            score = 1
        elif len(core) >= 3 and core.isalnum():
            score = 3  # distinctive unknown token: likely an identifier
        else:
            score = 1
        scored.append((score, len(core), -i, tok))
    if not scored:
        return q
    scored.sort(key=lambda t: (-t[0], -t[1], -t[2]))
    if scored[0][0] >= 3:
        # A distinctive identifier narrows the index enough on its own:
        # keep it plus the single most signal-bearing companion term.
        chosen = scored[:2]
    else:
        # Only generic technical words: keep the FIRST ones in query order
        # (natural phrasing usually leads with the most index-productive
        # pair, e.g. "rate limit retry backoff python" -> "rate limit retry").
        chosen = sorted(scored[:3], key=lambda t: -t[2])
    out = " ".join(t[3] for t in chosen)
    return out or q


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

    The query is shortened to a code-shaped form first (see _code_query):
    grep.app applies AND semantics across all words, so a raw
    natural-language query matches nothing. If the shortened query yields
    no hits, it is retried once (the index is flaky: the same query
    measured 0/10/0/10/10 hits over 5 identical calls) and the raw query
    is tried last as a fallback.
    """
    try:
        cq = _code_query(query)
        raw = (query or "").strip()
        out: list[dict] = []
        seen: set[str] = set()

        async def _run(aq: str) -> int:
            r = await grep_mcp_search(aq, timeout=30)
            if not r.get("success"):
                return 0
            added = 0
            for h in (r.get("hits") or []):
                shaped = _shape_grep_mcp(len(out), h)
                if shaped and shaped["url"] not in seen:
                    seen.add(shaped["url"])
                    out.append(shaped)
                    added += 1
                if len(out) >= count:
                    break
            return added

        added = await _run(cq)
        if not added and len(out) < count:
            # One retry: the index answers 0 hits intermittently.
            added = await _run(cq)
        if not added and len(out) < count and cq.strip().lower() != raw.lower():
            await _run(raw)
        return out[:count]
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
    """Sourcegraph public GraphQL backup lane.

    Two changes vs. the original: the count: prefix is capped at
    SG_COUNT_MAX (count:50 blows our 30s server-side budget and returns
    nothing), and the call is retried ONCE with a short backoff when the
    response is failed or empty. The retry uses the raw query when the
    shortened one produced nothing, so the shortened form never loses
    recall.
    """
    try:
        c = get_http_client()
        out: list[dict] = []
        cq = _code_query(query)
        attempts = [cq]
        if cq.strip().lower() != (query or "").strip().lower():
            attempts.append(query)
        for attempt in range(2):
            aq = attempts[attempt] if attempt < len(attempts) else query
            q = f"context:global count:{min(max(count, 1), SG_COUNT_MAX)} {aq} select:file"
            try:
                r = await c.post(
                    _SG_GRAPHQL,
                    json={"query": _SG_Q, "variables": {"q": q}},
                    timeout=_SG_TIMEOUT,
                    headers={"User-Agent": "websearch-mcp/1.0",
                             "Content-Type": "application/json",
                             "Accept": "application/json"},
                )
            except Exception as e:
                logger.debug("sourcegraph attempt %d failed: %s", attempt, e)
                if attempt == 0:
                    await asyncio.sleep(SG_RETRY_DELAY_S)
                    continue
                return []
            if r.status_code != 200:
                logger.debug("sourcegraph attempt %d HTTP %s", attempt, r.status_code)
                if attempt == 0:
                    await asyncio.sleep(SG_RETRY_DELAY_S)
                    continue
                return []
            try:
                v = r.json()
            except ValueError as e:
                logger.debug("sourcegraph attempt %d bad JSON: %s", attempt, e)
                if attempt == 0:
                    await asyncio.sleep(SG_RETRY_DELAY_S)
                    continue
                return []
            res = (((v.get("data") or {}).get("search") or {}).get("results")
                   or {}).get("results", [])
            for m in res:
                if not isinstance(m, dict) or m.get("__typename") != "FileMatch":
                    continue
                shaped = _shape_sg(len(out), m)
                if shaped:
                    out.append(shaped)
                if len(out) >= count:
                    break
            if out:
                return out
            # Empty result set: retry once (upstream flakiness), then give up.
            if attempt == 0:
                await asyncio.sleep(SG_RETRY_DELAY_S)
                continue
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
        try:
            # Bounded: the backup must never eat the vertical's wall-clock
            # budget and take the grep.app hits down with it.
            backup = await asyncio.wait_for(_sg_search(query, count),
                                            timeout=SG_BUDGET_S)
        except Exception as e:
            logger.debug("sourcegraph backup skipped: %s", e)
            backup = []
        for h in backup:
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
