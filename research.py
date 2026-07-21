from __future__ import annotations

import asyncio
import re
import time
import urllib.parse
from collections import Counter

from config import cached
from search_ddg import search_ddg, search_google_rss
from search_tavily import search_tavily
from search_anysearch import search_anysearch
from search_tinyfish import tinyfish_search
from search_gai import get_gai_client
from fetch import fetch_url
from embed import _embed, _dedup_rank, _cosine_sim
from wikipedia import search_wikipedia, fetch_wikipedia_summary_rest
from arxiv import search_arxiv
from reddit import search_reddit
from query_expand import expand_query
from reranker import rerank as _rerank
from resilience import CircuitBreaker


# Circuit breakers for external APIs (shared across calls)
_ddg_breaker = CircuitBreaker(failure_threshold=5, cooldown_seconds=60)


# TinyFish intent detection
_TINYFISH_NEWS = re.compile(r"(?i)\b(news|headlines?|breaking|latest|today|update|coverage|event)\b")
_TINYFISH_ACADEMIC = re.compile(r"(?i)\b(paper|research|study|arxiv|doi|survey|review|implementation|method|experiment)\b")


_STOPWORDS = {
    "the","and","for","with","from","that","this","are","was","were",
    "has","have","had","you","your","its","our","not","but","can",
    "will","into","via","using","use","how","what","when","why","who",
    "which","about","also","more","most","than","then","them","they",
    "their","there","here","such","each","other","some","any","all",
    "one","two","new","get","got","may","might","could","should",
    "would","does","did","done","been","being","very","just","like",
}
_WORD_RE = re.compile(r"[A-Za-z][A-Za-z0-9'+-]{2,}")

# Engines expected in a non-GAI-only search (for engine_blocked reporting)
_ALL_ENGINES = {"google-news-rss", "tavily", "reddit", "wikipedia", "arxiv",
                "anysearch", "tinyfish", "duckduckgo"}


def _query_tokens(query: str) -> set[str]:
    return {w.lower() for w in _WORD_RE.findall(query or "") if w.lower() not in _STOPWORDS}


def _related_queries(query: str, results: list[dict], *, n: int = 6) -> list[str]:
    """Mine follow-up queries from result titles+snippets via bigram doc-frequency.
    Zero API cost — pure text extraction. Returns up to n phrases."""
    if not results:
        return []
    q_tokens = _query_tokens(query)
    docs: list[list[str]] = []
    for r in results:
        text = (f"{r.get('title', '')} {r.get('snippet', '')}").lower()
        words = [w for w in _WORD_RE.findall(text) if w not in _STOPWORDS]
        docs.append(words)
    if not docs:
        return []

    bigram_docfreq: Counter[str] = Counter()
    for words in docs:
        uniq_bi = set()
        for i in range(len(words) - 1):
            a, b = words[i], words[i + 1]
            if len(a) < 3 or len(b) < 3:
                continue
            uniq_bi.add(f"{a} {b}")
        for bi in uniq_bi:
            bigram_docfreq[bi] += 1

    def _overlaps_query(phrase: str) -> bool:
        toks = phrase.split()
        if not toks:
            return True
        if not [t for t in toks if t not in q_tokens]:
            return True
        return query.lower() in phrase or phrase in query.lower()

    scored = [(df, bi) for bi, df in bigram_docfreq.items() if df >= 2 and not _overlaps_query(bi)]
    scored.sort(key=lambda x: (-x[0], x[1]))

    out: list[str] = []
    seen_words: set[str] = set()
    for _df, bi in scored:
        if len(set(bi.split()) & seen_words) >= 2:
            continue
        out.append(bi)
        seen_words.update(bi.split())
        if len(out) >= n:
            break
    return out[:n]


def _normalize_scores(results: list[dict]) -> list[dict]:
    """Normalize _rerank scores to 0-1 relevance_score range."""
    if not results:
        return results
    scores = [r.get("_rerank", 0) or 0 for r in results]
    if not scores:
        return results
    lo, hi = min(scores), max(scores)
    span = hi - lo if hi > lo else 1.0
    for r, s in zip(results, scores):
        normalized = round((s - lo) / span, 4)
        r["relevance_score"] = normalized
        if normalized >= 0.7:
            r["fetch_relevance"] = "high"
        elif normalized >= 0.4:
            r["fetch_relevance"] = "med"
        else:
            r["fetch_relevance"] = "low"
        # Clean up internal keys
        r.pop("_rerank", None)
        r.pop("_embedding", None)
        r.pop("_rel", None)
    return results


def _detect_tinyfish_type(query: str) -> str:
    if _TINYFISH_ACADEMIC.search(query):
        return "research_paper"
    if _TINYFISH_NEWS.search(query):
        return "news"
    return "web"


@cached(ttl=90)
async def search_multi(query: str, count: int = 10, cdp_url: str | None = None,
                       google_ai_only: bool = False, search_type: str = "auto",
                        search_prompt: str = "", gl: str = "",
                       hl: str = "en", tbs: str = "", pws: str = "", backend: str = "auto",
                       timelimit: str = "", page: int = 1, region: str = "wt-wt",
                       safesearch: str = "moderate",
                        language: str = "en", country: str = "",
                        reddit_subreddit: str = "", reddit_comments: bool = True,
                       upload_urls: list[str] | None = None,
                       query_expand: bool = True,
                        tavily_topic: str = "general", tavily_depth: str = "advanced",
                       domain: str = "",
                       include_domains: list[str] | None = None,
                       exclude_domains: list[str] | None = None,
                       size: str = "", color: str = "", type_image: str = "",
                       layout: str = "", license_image: str = "",
                       resolution: str = "", duration: str = "",
                       license_videos: str = "",
                       start_date: str = "", end_date: str = "",
                       exact_phrase: bool = False,
                       anysearch_tag: str = "", anysearch_zone: str = "",
                       anysearch_language: str = "",
                        anysearch_params: dict | None = None) -> dict:
    _start = time.monotonic()
    engines_used: list[str] = []
    results: list[dict] = []
    ai_answer = ""
    follow_up = ""

    async def _gai_search():
        gai = await get_gai_client(cdp_url)
        if not gai:
            return None
        try:
            r = await asyncio.wait_for(gai.search(query, search_prompt=search_prompt,
                gl=gl, hl=hl, tbs=tbs, pws=pws, upload_urls=upload_urls),
                timeout=300)
            if r.get("success"):
                return r
        except (asyncio.TimeoutError, Exception):
            pass
        return None

    if google_ai_only:
        r = await _gai_search()
        if r:
            engines_used.append("google-ai-mode")
            rd = r["result"]
            ai_answer = rd.get("answer", "")
            follow_up = rd.get("followUp", "")
            for s in rd.get("sources", []):
                results.append({"title": s["title"], "url": s["url"],
                    "snippet": s.get("snippet", ""),
                    "source": urllib.parse.urlparse(s["url"]).netloc if s.get("url") else "",
                    "engine": "google-ai-mode"})
    else:
        gai_future = asyncio.create_task(_gai_search())
        queries = [query]
        if query_expand:
            expanded = await expand_query(query)
            queries = expanded[:4]
        ddg_count = max(count * 2 // len(queries), 5)

        async def _ddg_with_breaker(q, n, **kw):
            if not _ddg_breaker.allow():
                return []
            try:
                r = await search_ddg(q, n, **kw)
                _ddg_breaker.record_success()
                return r
            except Exception:
                _ddg_breaker.record_failure()
                return []

        async def _reddit_search(q, n, **kw):
            try:
                return await search_reddit(q, n, **kw)
            except Exception:
                return []

        tr_map = {"d": "day", "w": "week", "m": "month", "y": "year"}
        tavily_tr = tr_map.get(timelimit, "")

        ddg_tasks = {}
        for i, q in enumerate(queries):
            k = f"ddg_{i}"
            ddg_tasks[k] = asyncio.create_task(_ddg_with_breaker(
                q, ddg_count, search_type=search_type, backend=backend,
                timelimit=timelimit, page=page, region=region, safesearch=safesearch,
                size=size, color=color, type_image=type_image, layout=layout,
                license_image=license_image, resolution=resolution, duration=duration,
                license_videos=license_videos))

        tasks = {
            "rss": asyncio.create_task(search_google_rss(query, count, region=region, timelimit=timelimit)),
            "tavily": asyncio.create_task(search_tavily(query, n=count, topic=tavily_topic,
                time_range=tavily_tr or tbs, search_depth=tavily_depth, include_raw_content=True,
                start_date=start_date, end_date=end_date, exact_phrase=exact_phrase,
                country=country, include_domains=include_domains, exclude_domains=exclude_domains)),
            "wiki": asyncio.create_task(search_wikipedia(query, count=min(count, 8), language=language)),
            "reddit": asyncio.create_task(_reddit_search(query, count,
                subreddit=reddit_subreddit or None, include_comments=reddit_comments,
                include_ai_summary=True)),
            "arxiv": asyncio.create_task(search_arxiv(query, count=min(count, 10))),
            "anysearch": asyncio.create_task(search_anysearch(query, count=min(count, 20), domain=domain,
                tag=anysearch_tag, zone=anysearch_zone, language=anysearch_language,
                params=anysearch_params)),
            "tinyfish": asyncio.create_task(tinyfish_search(query, count=min(count, 50),
                domain_type=_detect_tinyfish_type(query), goal=query)),
            **ddg_tasks,
        }
        done = await asyncio.gather(*tasks.values(), return_exceptions=True)
        done_map = dict(zip(tasks.keys(), done))
        for key in ("rss", "tavily", "reddit", "wiki", "arxiv", "anysearch", "tinyfish") + tuple(ddg_tasks.keys()):
            val = done_map[key]
            if isinstance(val, BaseException) or not isinstance(val, list):
                continue
            for r in val:
                if isinstance(r, dict) and "error" not in r and r.get("url"):
                    if not any(e.get("url") == r["url"] for e in results):
                        if key != "reddit" or r.get("engine") in ("reddit", "reddit-comment", "reddit-ai-summary"):
                            new_eng = "duckduckgo" if key.startswith("ddg") else (
                                "google-news-rss" if key == "rss" else key)
                            # Preserve sub-engine (e.g., reddit-ai-summary, reddit-comment)
                            cur = r.get("engine")
                            if not cur or cur == key or cur == new_eng:
                                r["engine"] = new_eng
                        results.append(r)
        eng = {"rss": "google-news-rss", "tavily": "tavily", "reddit": "reddit", "wiki": "wikipedia", "arxiv": "arxiv", "anysearch": "anysearch", "tinyfish": "tinyfish"}
        for key, name in eng.items():
            val = done_map.get(key)
            if isinstance(val, list) and any(isinstance(r, dict) and "error" not in r for r in val):
                engines_used.append(name)
        if any(isinstance(done_map[k], list) for k in ddg_tasks):
            engines_used.append("duckduckgo")

        try:
            completed, _ = await asyncio.wait([gai_future], timeout=300)
        except BaseException:
            completed = set()
        if gai_future in completed:
            try:
                r = gai_future.result()
            except BaseException:
                r = None
            if r and r.get("success"):
                engines_used.append("google-ai-mode")
                rd = r["result"]
                ai_answer = rd.get("answer", "")
                follow_up = rd.get("followUp", "")
                for s in rd.get("sources", []):
                    if not any(e.get("url") == s["url"] for e in results):
                        results.append({"title": s["title"], "url": s["url"],
                            "snippet": s.get("snippet", ""),
                            "source": urllib.parse.urlparse(s["url"]).netloc if s.get("url") else "",
                            "engine": "google-ai-mode"})
        else:
            gai_future.cancel()
    # Track which engines were expected but didn't contribute
    engine_blocked = sorted(_ALL_ENGINES - set(engines_used))

    if results:
        texts = [(r.get("snippet", "") or "")[:300] + " " +
                 (r.get("title", "") or "")[:100] for r in results]
        q_emb, item_emb = await asyncio.gather(
            _embed([query], "query"),
            _embed(texts, "passage"),
        )
        if q_emb and item_emb:
            for r, emb in zip(results, item_emb):
                r["_embedding"] = emb
            deduped = _dedup_rank(results, q_emb[0])
        else:
            deduped = results
        deduped = await _rerank(query, deduped, top_k=min(count, len(deduped) + 1))
        deduped = _normalize_scores(deduped)
        related = _related_queries(query, deduped)
    else:
        deduped = results
        related = []
    out: dict = {"success": True, "engines_used": engines_used,
                 "engine_blocked": engine_blocked,
                 "ai_answer": ai_answer, "follow_up": follow_up,
                 "related_queries": related,
                 "results": deduped[:count], "total": len(deduped[:count]),
                 "duration_ms": round((time.monotonic() - _start) * 1000)}
    return out


async def enrich(results: list[dict], query: str, depth: int = 3,
                 cdp_url: str | None = None,
                 count: int = 10, language: str = "en") -> dict:
    """Fetch full page content from top results, dedup, rerank.

    Takes snippet results (from search_multi), fetches their full content,
    deduplicates by embedding cosine similarity, and reranks by query relevance.
    Also enriches Wikipedia results with full summary extracts.
    """
    urls = [r["url"] for r in results if r.get("url")]
    if not urls:
        return {"fetched_content": []}
    fetched = await asyncio.gather(
        *[fetch_url(url, max_chars=3000, fast=True) for url in urls], return_exceptions=True)
    fetched = [f for f in fetched if isinstance(f, dict) and f.get("success")]
    for i, r in enumerate(results[:5]):
        url = r.get("url", "")
        if not url or "wikipedia.org" not in url:
            continue
        page_path = urllib.parse.urlparse(url).path.strip("/")
        if page_path.startswith("wiki/"):
            page_path = page_path[5:]
        page_title = urllib.parse.unquote(page_path.replace("_", " "))
        if not page_title:
            continue
        wiki_summary = await fetch_wikipedia_summary_rest(page_title, language=language)
        if wiki_summary:
            for f in fetched:
                if f.get("url") == url:
                    f.setdefault("content", "")
                    content = f["content"]
                    summary_text = wiki_summary.get("extract", "")
                    if summary_text and len(summary_text) > len(content):
                        f["content"] = summary_text + "\n\n" + content
                    break
    if len(fetched) > 1:
        texts = [(f.get("content", "") or "")[:300] + " " +
                 (f.get("title", "") or "")[:100] for f in fetched]
        item_emb = await _embed(texts, "passage")
        if item_emb:
            deduped = []
            seen_emb: list[list[float]] = []
            for f, emb in zip(fetched, item_emb):
                is_dup = any(_cosine_sim(emb, se) > 0.90 for se in seen_emb)
                if not is_dup:
                    deduped.append(f)
                    seen_emb.append(emb)
            fetched = deduped
        fetched = await _rerank(query, fetched, top_k=depth * 2)
        fetched = _normalize_scores(fetched)
    return {"fetched_content": fetched}
