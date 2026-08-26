"""Keyless JSON verticals (port of donsetch src/search/verticals.rs).

Intent-gated bonus lanes that feed the same merge as the web engines:
a GitHub repo that also ranks in Brave gets consensus. Advisory only —
a vertical failure returns [] and never fails the search.
"""
from __future__ import annotations

import json
import re
from urllib.parse import quote

from config import get_http_client
from search_academic import search_academic

FETCH_TIMEOUT = 12.0
MAX_PER_VERTICAL = 5

_ACADEMIC = {"openalex", "crossref", "pubmed", "europepmc"}

_RSS_ITEM = re.compile(r"<item>(.*?)</item>", re.S)
_ATOM_ENTRY = re.compile(r"<entry>(.*?)</entry>", re.S)
_TAG = re.compile(r"<(/?)[a-zA-Z][^>]*>")
_CDATA = re.compile(r"<!\[CDATA\[(.*?)\]\]>", re.S)

_GITHUB_ERRORISH = ["error", "exception", "failed", "bug", "crash", "panic",
                    "cannot", "undefined"]
_GITHUB_REPOISH = ["library", "crate", "framework", "plugin", "sdk", "cli",
                   "tool", "vs code", "extension"]


def endpoint(vertical: str, query: str) -> str | None:
    q = quote(query, safe="")
    if vertical == "wikipedia":
        return (f"https://en.wikipedia.org/w/api.php?action=query&list=search"
                f"&srsearch={q}&format=json&srlimit={MAX_PER_VERTICAL}")
    if vertical == "hn":
        return (f"https://hn.algolia.com/api/v1/search?query={q}"
                f"&hitsPerPage={MAX_PER_VERTICAL}")
    if vertical == "stackexchange":
        return (f"https://api.stackexchange.com/2.3/search/advanced?"
                f"order=desc&sort=relevance&q={q}&site=stackoverflow"
                f"&pagesize={MAX_PER_VERTICAL}")
    if vertical == "mdn":
        return f"https://developer.mozilla.org/api/v1/search?q={q}"
    if vertical == "github":
        lower = query.lower().replace("+", " ")
        words = lower.split()
        errorish = any(s in lower for s in _GITHUB_ERRORISH)
        repoish = (any(s in lower for s in _GITHUB_REPOISH) or len(words) <= 4)
        # Natural-language how-tos return spam repos from GitHub's loose
        # matcher — skip the vertical.
        if not errorish and not repoish:
            return None
        if errorish:
            return f"https://api.github.com/search/issues?q={q}&per_page={MAX_PER_VERTICAL}"
        return f"https://api.github.com/search/repositories?q={q}&per_page={MAX_PER_VERTICAL}"
    if vertical == "scholar":
        return (f"https://api.semanticscholar.org/graph/v1/paper/search?"
                f"query={q}&limit={MAX_PER_VERTICAL}&fields=title,url,abstract,year")
    if vertical == "news":
        return (f"https://news.google.com/rss/search?q={q}&hl=en-US&gl=US"
                f"&ceid=US:en")
    if vertical == "arxiv":
        return (f"https://export.arxiv.org/api/query?search_query=all:{q}"
                f"&max_results={MAX_PER_VERTICAL}&sortBy=relevance")
    return None


async def run(vertical: str, query: str) -> list[dict]:
    """Fetch one vertical, return hits in the standard {title,url,snippet} shape."""
    if vertical in _ACADEMIC:
        return await search_academic(query, MAX_PER_VERTICAL, source=vertical)
    url = endpoint(vertical, query)
    if not url:
        return []
    c = get_http_client()
    try:
        r = await c.get(url, timeout=FETCH_TIMEOUT)
        if r.status_code != 200:
            return []
        body = r.text
    except Exception:
        return []
    if vertical == "news":
        return _parse_rss(body)
    if vertical == "arxiv":
        return _parse_atom(body)
    return _parse_json(vertical, body)


def _strip_tags(s: str) -> str:
    return " ".join(_TAG.sub("", s).split())


def _entry(rank: int, title: str, url: str, snippet: str,
           published: str | None = None) -> dict:
    return {"title": title, "url": url, "snippet": snippet,
            "published": published, "rank": rank}


def _parse_json(vertical: str, body: str) -> list[dict]:
    try:
        v = json.loads(body)
    except Exception:
        return []
    if vertical == "wikipedia":
        out = []
        for rank, it in enumerate(v.get("query", {}).get("search", [])):
            title = it.get("title", "")
            out.append(_entry(rank, title,
                              f"https://en.wikipedia.org/wiki/{title.replace(' ', '_')}",
                              _strip_tags(it.get("snippet", ""))))
        return out
    if vertical == "hn":
        out = []
        for rank, it in enumerate(v.get("hits", [])):
            url = it.get("url") or it.get("story_url")
            if not url:
                continue
            points = it.get("points", 0) or 0
            comments = it.get("num_comments", 0) or 0
            out.append(_entry(rank, it.get("title", ""), url,
                              f"Hacker News: {points} points, {comments} comments"))
        return out
    if vertical == "github":
        out = []
        for rank, it in enumerate(v.get("items", [])):
            if "stargazers_count" in it:
                stars = it.get("stargazers_count", 0)
                out.append(_entry(rank, it.get("full_name", ""),
                                  it.get("html_url", ""),
                                  f"{it.get('description', '') or ''} (★ {stars})"))
            else:
                repo = (it.get("repository_url", "")
                        .replace("https://api.github.com/repos/", ""))
                out.append(_entry(rank, f"{repo}: {it.get('title', '')}",
                                  it.get("html_url", ""),
                                  f"GitHub issue ({it.get('state', '')})"))
        return out
    if vertical == "stackexchange":
        out = []
        for rank, it in enumerate(v.get("items", [])):
            score = it.get("score", 0)
            answers = it.get("answer_count", 0)
            answered = it.get("is_answered", False)
            tail = ", accepted answer" if answered else ""
            out.append(_entry(rank, it.get("title", ""), it.get("link", ""),
                              f"Stack Overflow: score {score}, {answers} answers{tail}"))
        return out
    if vertical == "mdn":
        out = []
        for rank, it in enumerate(v.get("documents", [])[:MAX_PER_VERTICAL]):
            mdn_url = it.get("mdn_url", "")
            if not mdn_url:
                continue
            out.append(_entry(rank, it.get("title", ""),
                              f"https://developer.mozilla.org{mdn_url}",
                              it.get("summary", "")))
        return out
    if vertical == "scholar":
        out = []
        for rank, it in enumerate(v.get("data", [])):
            year = it.get("year", "") or ""
            abstract = (it.get("abstract", "") or "")[:220]
            out.append(_entry(rank, it.get("title", ""), it.get("url", ""),
                              f"{abstract} ({year})" if year else abstract))
        return out
    return []


def _rss_date_to_iso(date: str) -> str | None:
    parts = date.split()
    if len(parts) < 4:
        return None
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
              "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
    try:
        m = next(i for i, name in enumerate(months) if parts[2].startswith(name))
    except StopIteration:
        return None
    return f"{parts[3]}-{m + 1:02d}-{parts[1]}"


def _grab(item: str, tag: str) -> str:
    m = re.search(rf"<{tag}>(.*?)</{tag}>", item, re.S)
    if not m:
        return ""
    return _CDATA.sub(lambda x: x.group(1), m.group(1))


def _parse_rss(body: str) -> list[dict]:
    out = []
    for rank, item in enumerate(_RSS_ITEM.findall(body)):
        if rank >= 8:
            break
        title = _grab(item, "title").strip()
        url = _grab(item, "link").strip()
        date = _grab(item, "pubDate").strip()
        if title and url.startswith("http"):
            out.append(_entry(rank, title, url, date,
                              _rss_date_to_iso(date)))
    return out


def _parse_atom(body: str) -> list[dict]:
    out = []
    for rank, entry in enumerate(_ATOM_ENTRY.findall(body)):
        if rank >= MAX_PER_VERTICAL:
            break
        title = " ".join(_grab(entry, "title").split())
        url = _grab(entry, "id").strip()
        summary = " ".join(_grab(entry, "summary").split())[:220]
        published = _grab(entry, "published").strip()[:10]
        if title and url.startswith("http"):
            out.append(_entry(rank, title, url, summary,
                              published or None))
    return out


if __name__ == "__main__":
    import asyncio

    async def check():
        cases = [
            ("github", "rust async runtime"),
            ("github", "how to fix a leaking kitchen faucet"),
            ("stackexchange", "python asyncio gather vs wait"),
            ("mdn", "array map javascript"),
            ("hn", "rust async"),
            ("scholar", "retrieval augmented generation"),
            ("news", "ukraine war"),
            ("arxiv", "attention is all you need"),
        ]
        for vertical, q in cases:
            hits = await run(vertical, q)
            print(f"{vertical:14s} {q!r}: {len(hits)} hits", flush=True)
            assert len(hits) <= 8
            for h in hits[:1]:
                assert h["title"] and h["url"].startswith("http")

    asyncio.run(check())
    print("VERTICALS-OK")
