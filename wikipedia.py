from __future__ import annotations

import html
import urllib.parse
from datetime import datetime, timedelta
from typing import Any

from config import get_http_client


_WIKI_BASE = "https://{lang}.wikipedia.org/w/api.php"
_REST_BASE = "https://{lang}.wikipedia.org/api/rest_v1/"
_UA = "mcp-codesearch/1.0"
_ALLOWED_LANGS = {"en","de","fr","es","it","pt","ru","ja","zh","pl","nl","ar","ko","tr","sv","no","da","fi","hu","cs","ro","uk","th","vi","id","he","el","fa","hi","bn","ta","te","ur","ms","bg","hr","sr","sk","sl","lt","lv","et","is","mk"}
def _lang(lang: str) -> str:
    return lang if lang in _ALLOWED_LANGS else "en"


async def _get(params: dict, language: str = "en", timeout: int = 10) -> dict | None:
    c = get_http_client()
    r = await c.get(_WIKI_BASE.format(lang=_lang(language)), params=params, headers={"User-Agent": _UA}, timeout=timeout)
    return r.json() if r.status_code == 200 else None


def _page_url(title: str, language: str = "en") -> str:
    return f"https://{_lang(language)}.wikipedia.org/wiki/{urllib.parse.quote(title.replace(' ', '_'))}"


def _clean_snippet(s: str) -> str:
    return html.unescape(s).replace('<span class="searchmatch">', "**").replace("</span>", "**")[:500]


async def search_wikipedia(query: str, count: int = 3, language: str = "en",
                           namespace: int = 0) -> list[dict]:
    try:
        params: dict[str, Any] = {"action": "query", "list": "search", "srsearch": query, "format": "json",
                           "srlimit": min(count, 10), "srprop": "snippet|titlesnippet|timestamp|sectiontitle|wordcount"}
        if namespace:
            params["srnamespace"] = namespace
        data = await _get(params, language)
        if not data:
            return []
        return {"results": [{"title": h.get("title", ""),
                 "url": _page_url(h.get("title", ""), language),
                 "snippet": _clean_snippet(h.get("snippet", "")),
                 "source": "wikipedia", "timestamp": h.get("timestamp", ""),
                 "wordcount": h.get("wordcount", 0)}
                for h in data.get("query", {}).get("search", [])],
                "total_available": data.get("query", {}).get("searchinfo", {}).get("totalhits", 0)}
    except Exception:
        return {"results": [], "total_available": 0}


async def fetch_wikipedia_summary(query: str, language: str = "en",
                                   include_images: bool = False) -> dict | None:
    try:
        params = {"action": "query", "prop": "extracts|info|sections|pageimages",
                   "exintro": True, "explaintext": True, "titles": query, "format": "json",
                   "redirects": 1, "inprop": "url"}
        if include_images:
            params["piprop"] = "thumbnail"
            params["pithumbsize"] = 800
        data = await _get(params, language)
        if not data:
            return None
        pages = data.get("query", {}).get("pages", {})
        for pid, page in pages.items():
            if pid == "-1":
                continue
            sections = page.get("sections", [])
            result: dict = {
                "title": page.get("title", ""),
                "url": page.get("fullurl", ""),
                "extract": (page.get("extract", "") or "")[:5000],
                "page_id": pid,
                "sections": [{"index": s.get("index", ""), "level": s.get("level", ""),
                              "title": s.get("toclevel", s.get("line", ""))}
                             for s in sections[:20]],
            }
            if include_images and page.get("thumbnail"):
                result["thumbnail"] = page["thumbnail"].get("source", "")
            return result
        return None
    except Exception:
        return None


async def fetch_wikipedia_summary_rest(title: str, language: str = "en") -> dict | None:
    """Fast page summary via REST API v1 — lighter than Action API."""
    try:
        c = get_http_client()
        r = await c.get(
            _REST_BASE.format(lang=_lang(language)) + f"page/summary/{urllib.parse.quote(title.replace(' ', '_'))}",
            headers={"User-Agent": _UA}, timeout=10)
        if r.status_code != 200:
            return None
        d = r.json()
        return {
            "title": d.get("title", ""), "url": d.get("content_urls", {}).get("desktop", {}).get("page", ""),
            "extract": (d.get("extract", "") or "")[:5000],
            "extract_html": d.get("extract_html", ""), "page_id": d.get("pageid"),
            "thumbnail": (d.get("thumbnail", {}) or {}).get("source", ""),
            "description": d.get("description", ""), "source": "wikipedia-rest",
        }
    except Exception:
        return None


async def fetch_wikipedia_categories(title: str, language: str = "en") -> list[dict]:
    """Get categories a page belongs to."""
    try:
        data = await _get({"action": "query", "prop": "categories", "titles": title,
                           "format": "json", "cllimit": 50, "redirects": 1}, language)
        if not data:
            return []
        for pid, page in data.get("query", {}).get("pages", {}).items():
            if pid == "-1":
                continue
            cats = page.get("categories", [])
            return [{"title": c.get("title", "").replace("Category:", ""),
                     "url": _page_url(c.get("title", ""), language),
                     "sortkey": c.get("sortkey", "")} for c in cats]
        return []
    except Exception:
        return []


async def fetch_wikipedia_links(title: str, language: str = "en", namespace: int = 0,
                                 count: int = 50) -> list[dict]:
    """Get internal links from a page."""
    try:
        data = await _get({"action": "query", "prop": "links", "titles": title,
                           "format": "json", "plnamespace": namespace, "pllimit": min(count, 500),
                           "redirects": 1}, language)
        if not data:
            return []
        for pid, page in data.get("query", {}).get("pages", {}).items():
            if pid == "-1":
                continue
            links = page.get("links", [])
            return [{"title": l.get("title", ""), "pageid": l.get("pageid", 0),
                     "url": _page_url(l.get("title", ""), language)} for l in links[:count]]
        return []
    except Exception:
        return []


async def fetch_wikipedia_extlinks(title: str, language: str = "en", count: int = 50) -> list[dict]:
    """Get external URLs from a page."""
    try:
        data = await _get({"action": "query", "prop": "extlinks", "titles": title,
                           "format": "json", "ellimit": min(count, 500), "redirects": 1}, language)
        if not data:
            return []
        for pid, page in data.get("query", {}).get("pages", {}).items():
            if pid == "-1":
                continue
            return [{"url": l.get("*", "")} for l in page.get("extlinks", [])[:count]]
        return []
    except Exception:
        return []


async def fetch_wikipedia_pageviews(title: str, language: str = "en",
                                      days: int = 30) -> list[dict]:
    """Get daily pageview stats for the last N days."""
    try:
        encoded = urllib.parse.quote(title.replace(" ", "_"))
        c = get_http_client()
        end = datetime.now()
        start = end - timedelta(days=days)
        r = await c.get(
            _REST_BASE.format(lang=_lang(language)) + f"page/per-day/{encoded}/daily/{start.strftime('%Y%m%d')}/{end.strftime('%Y%m%d')}",
            headers={"User-Agent": _UA}, timeout=10)
        if r.status_code != 200:
            return []
        d = r.json()
        items = d.get("items", [])
        return [{"date": i.get("timestamp", "")[:8] if i.get("timestamp") else "",
                 "views": i.get("views", 0), "rank": i.get("rank")} for i in items]
    except Exception:
        return []


async def fetch_wikipedia_revisions(title: str, language: str = "en",
                                     count: int = 10) -> list[dict]:
    """Get recent revision history for a page."""
    try:
        data = await _get({"action": "query", "prop": "revisions", "titles": title,
                           "format": "json", "rvlimit": min(count, 50), "redirects": 1,
                           "rvprop": "ids|timestamp|user|size|comment|flags"}, language)
        if not data:
            return []
        for pid, page in data.get("query", {}).get("pages", {}).items():
            if pid == "-1":
                continue
            revs = page.get("revisions", [])
            return [{"revid": r.get("revid", 0), "parentid": r.get("parentid", 0),
                     "timestamp": r.get("timestamp", ""), "user": r.get("user", ""),
                     "size": r.get("size", 0), "comment": r.get("comment", ""),
                     "minor": r.get("minor", False)} for r in revs]
        return []
    except Exception:
        return []


async def search_wikipedia_category(category: str, language: str = "en",
                                     count: int = 20) -> list[dict]:
    """Enumerate pages in a category."""
    try:
        data = await _get({"action": "query", "list": "categorymembers",
                           "cmtitle": f"Category:{category}", "format": "json",
                           "cmlimit": min(count, 50), "cmtype": "page|subcat"},
                          language)
        if not data:
            return []
        members = data.get("query", {}).get("categorymembers", [])
        return [{"title": m.get("title", ""), "pageid": m.get("pageid", 0),
                 "type": m.get("ns", 0) == 14 and "subcategory" or "page",
                 "url": _page_url(m.get("title", ""), language)}
                for m in members]
    except Exception:
        return []


async def search_wikipedia_backlinks(title: str, language: str = "en",
                                      count: int = 50) -> list[dict]:
    """Find all pages linking to a given page."""
    try:
        data = await _get({"action": "query", "list": "backlinks", "bltitle": title,
                           "format": "json", "bllimit": min(count, 500)}, language)
        if not data:
            return []
        return [{"title": b.get("title", ""), "pageid": b.get("pageid", 0),
                 "url": _page_url(b.get("title", ""), language)}
                for b in data.get("query", {}).get("backlinks", [])]
    except Exception:
        return []


async def search_wikipedia_geosearch(lat: float, lon: float, distance: int = 1000,
                                      count: int = 10, language: str = "en") -> list[dict]:
    try:
        data = await _get({"action": "query", "list": "geosearch", "gscoord": f"{lat}|{lon}",
                           "gsradius": distance, "gslimit": min(count, 50), "format": "json"}, language)
        if not data:
            return []
        return [{"title": h.get("title", ""),
                 "url": _page_url(h.get("title", ""), language),
                 "source": "wikipedia-geosearch",
                 "distance": h.get("dist", 0), "pageid": h.get("pageid", 0)}
                for h in data.get("query", {}).get("geosearch", [])]
    except Exception:
        return []


async def search_wikipedia_random(count: int = 5, language: str = "en",
                                   namespace: int = 0) -> list[dict]:
    try:
        data = await _get({"action": "query", "list": "random", "rnnamespace": namespace,
                           "rnlimit": min(count, 50), "format": "json"}, language)
        if not data:
            return []
        return [{"title": h.get("title", ""),
                 "url": _page_url(h.get("title", ""), language),
                 "source": "wikipedia-random", "pageid": h.get("id", 0)}
                for h in data.get("query", {}).get("random", [])]
    except Exception:
        return []


async def search_wikipedia_allpages(namespace: int = 0, limit: int = 50,
                                     language: str = "en",
                                     apfrom: str = "") -> list[dict]:
    """Enumerate pages in a namespace via action=query&list=allpages."""
    try:
        params: dict[str, Any] = {"action": "query", "list": "allpages",
                                   "apnamespace": namespace,
                                   "aplimit": min(limit, 500), "format": "json"}
        if apfrom:
            params["apfrom"] = apfrom
        data = await _get(params, language)
        if not data:
            return []
        return [{"title": p.get("title", ""),
                 "url": _page_url(p.get("title", ""), language),
                 "pageid": p.get("pageid", 0), "ns": p.get("ns", 0)}
                for p in data.get("query", {}).get("allpages", [])]
    except Exception:
        return []


async def search_wikipedia_recentchanges(language: str = "en", count: int = 20,
                                           type_filter: str = "") -> list[dict]:
    """Get recent changes feed."""
    try:
        params = {"action": "query", "list": "recentchanges", "format": "json",
                  "rclimit": min(count, 50), "rcprop": "title|timestamp|user|comment|flags|sizes|ids"}
        if type_filter:
            params["rctype"] = type_filter
        data = await _get(params, language)
        if not data:
            return []
        return [{"title": c.get("title", ""), "timestamp": c.get("timestamp", ""),
                 "user": c.get("user", ""), "comment": c.get("comment", ""),
                 "new": c.get("new", False), "minor": c.get("minor", False),
                 "oldlen": c.get("oldlen", 0), "newlen": c.get("newlen", 0),
                 "revid": c.get("revid", 0), "pageid": c.get("pageid", 0)}
                for c in data.get("query", {}).get("recentchanges", [])]
    except Exception:
        return []


async def fetch_wikipedia_langlinks(title: str, language: str = "en",
                                     count: int = 50) -> list[dict]:
    """Get language links (interwiki links) from a page."""
    try:
        data = await _get({"action": "query", "prop": "langlinks", "titles": title,
                           "format": "json", "lllimit": min(count, 500), "redirects": 1,
                           "llprop": "url|langname|autonym"}, language)
        if not data:
            return []
        for pid, page in data.get("query", {}).get("pages", {}).items():
            if pid == "-1":
                continue
            langlinks = page.get("langlinks", [])
            return [{"lang": ll.get("lang", ""),
                     "langname": ll.get("langname", ""),
                     "autonym": ll.get("autonym", ""),
                     "title": ll.get("*", ""),
                     "url": ll.get("url", "")} for ll in langlinks[:count]]
        return []
    except Exception:
        return []
