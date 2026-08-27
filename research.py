from __future__ import annotations

import asyncio
import json
import logging
import re
import time
import urllib.parse
from collections import Counter

from config import cached
from search_ddg import search_ddg, search_google_rss
from search_brave import search_brave
from search_tavily import search_tavily
from search_anysearch import search_anysearch
from search_tinyfish import tinyfish_search
from search_gai import get_gai_client
from fetch import fetch_url
from embed import _embed, _cosine_sim
from wikipedia import search_wikipedia
from arxiv import search_arxiv
from reddit import search_reddit
from query_expand import expand_query
from reranker import rerank as _rerank, killswitch_active as _reranker_disabled
from merge import (merge_base, blend, apply_coverage, apply_authority, finalize,
                   detect_intent, is_weak, merged_total, norm_key, verticals_for,
                   apply_six_signal)
from verticals import run as vertical_run



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
                "anysearch", "tinyfish", "duckduckgo", "brave"}


def _host_of(url: str) -> str:
    try:
        return (urllib.parse.urlsplit(url).hostname or "").lower()
    except ValueError:
        return ""


def _netloc_of(url: str) -> str:
    try:
        return urllib.parse.urlparse(url).netloc
    except ValueError:
        return ""


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
    """Normalize blended scores to 0-1 relevance_score range."""
    if not results:
        return results
    scores = [
        r.get("score") or r.get("_rerank") or r.get("_rel") or r.get("_relevance")
        or r.get("_hybrid") or 0
        for r in results
    ]
    lo, hi = min(scores), max(scores)
    if hi == 0:
        # Unranked (killswitch/reranker failure): keep engine order with a
        # monotonic descending score so the client still gets a ranking signal.
        n = len(results)
        for i, r in enumerate(results):
            r["relevance_score"] = round(1 - i / n, 4)
            r["fetch_relevance"] = "high" if i < n / 3 else ("med" if i < 2 * n / 3 else "low")
    elif hi == lo:
        # Tied real scores: every result is equally top-ranked.
        for r in results:
            r["relevance_score"] = 1.0
            r["fetch_relevance"] = "high"
    else:
        span = hi - lo
        for r, s in zip(results, scores):
            normalized = round((s - lo) / span, 4)
            r["relevance_score"] = normalized
            if normalized >= 0.7:
                r["fetch_relevance"] = "high"
            elif normalized >= 0.4:
                r["fetch_relevance"] = "med"
            else:
                r["fetch_relevance"] = "low"
    # Clean up internal keys (reranker + merge staging keys)
    for r in results:
        for k in [k for k in r if k.startswith("_")]:
            r.pop(k)
        if "sources" in r:
            # Best-ranked (lowest rank value) engine as the entry's engine,
            # mirroring the pre-merge contract.
            r["engine"] = min(r["sources"], key=lambda se: se[1])[0]
            r.pop("sources")
        r.pop("score", None)
    return results


def _detect_tinyfish_type(query: str) -> str:
    if _TINYFISH_ACADEMIC.search(query):
        return "research_paper"
    if _TINYFISH_NEWS.search(query):
        return "news"
    return "web"


async def _groq_text(sys_prompt: str, user_prompt: str,
                     temperature: float = 0.2, max_tokens: int = 300) -> str | None:
    """One Groq call returning trimmed text; None on any failure (caller falls back)."""
    from query_expand import _next_key
    key = await _next_key()
    if not key:
        return None
    try:
        from config import get_http_client
        c = get_http_client()
        for attempt in range(3):
            r = await c.post(
                "https://api.groq.com/openai/v1/chat/completions",
                headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
                json={"model": "openai/gpt-oss-120b",
                      "messages": [{"role": "system", "content": sys_prompt},
                                   {"role": "user", "content": user_prompt}],
                      "temperature": temperature, "max_tokens": max_tokens},
                timeout=15,
            )
            if r.status_code == 200:
                content = (r.json().get("choices") or [{}])[0].get("message", {}).get("content", "").strip()
                if content:
                    return content
    except Exception as e:
        logging.getLogger("research").warning(f"_groq_text failed: {type(e).__name__}: {e}")
    return None


async def decompose_query(query: str, num_queries: int = 4,
                          date: str | None = None) -> list[dict]:
    """Return [{query, researchGoal}, ...] for parallel fan-out.

    Falls back to [{"query": query, "researchGoal": ""}] on any LLM/parse error.
    """
    today = date or time.strftime("%Y-%m-%d")
    sys_p = (
        "You are an expert research assistant decomposing a question into distinct web search queries.\n"
        "Rules (strict):\n"
        "- Do NOT use search operators: no site:, filetype:, inurl:, intitle:, OR, AND, NOT, and no quote-wrapped phrases.\n"
        "- Write each query as a human would type it into a search engine.\n"
        f"- Assume the current date is {today} if the task is time-sensitive.\n"
        f"- Generate {num_queries} unique, non-overlapping queries covering different angles (background, current status, critiques, data/examples, outlook).\n"
        "Return ONLY a JSON array of objects: [{\"query\": \"...\", \"researchGoal\": \"...\"}].\n"
        "researchGoal = what this query should establish and how it advances the overall answer. No prose, no markdown fences."
    )
    text = await _groq_text(sys_p, f"Task: {query}", temperature=0.3, max_tokens=512)
    if not text:
        return [{"query": query, "researchGoal": ""}]
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    try:
        parsed = json.loads(text)
        if isinstance(parsed, list):
            out = [d for d in parsed if isinstance(d, dict) and d.get("query")][:num_queries]
            if out:
                return out
    except Exception:
        pass
    return [{"query": query, "researchGoal": ""}]


async def rewrite_query(query: str, history: str = "") -> str:
    """Return a self-contained, context-independent rephrase of query.
    Falls back to the original query on any error."""
    sys_p = (
        "Rephrase the user's query into a single self-contained, context-independent search query.\n"
        "Expand pronouns and references (\"it\", \"the second one\", \"they\") using any supplied context.\n"
        "Output ONLY the rewritten query string, no labels, no quotes, no explanation."
    )
    text = await _groq_text(sys_p, f"Context: {history}\nQuery: {query}", temperature=0.2, max_tokens=200)
    if text and len(text) <= 400 and not text.lower().startswith(("query:", "here")):
        return text
    return query


async def classify_need(query: str) -> str:
    """Return one of: 'academic' | 'discussion' | 'general'.
    NEVER returns a skip-search signal. Falls back to 'general' on error."""
    sys_p = (
        "Classify the search intent of the query into exactly one label:\n"
        '- "academic": scholarly/paper/research/technical-documentation intent\n'
        '- "discussion": opinions, forums, social, community, comparisons-by-users intent\n'
        '- "general": everything else\n'
        "Output ONLY the label word. Web search is ALWAYS required; do not suggest skipping it."
    )
    text = await _groq_text(sys_p, f"Query: {query}", temperature=0.1, max_tokens=64)
    if text and text.lower() in ("academic", "discussion", "general"):
        return text.lower()
    return "general"


@cached(ttl=90)
async def search_multi(query: str, count: int = 10, cdp_url: str | None = None,
                       google_ai_only: bool = False, search_type: str = "auto",
                        search_prompt: str = "", gl: str = "",
                       hl: str = "en", tbs: str = "", pws: str = "", backend: str = "auto",
                       timelimit: str = "", page: int = 1, region: str = "wt-wt",
                       safesearch: str = "moderate",
                        language: str = "en",
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
                       anysearch_tag: str = "", anysearch_zone: str = "",
                       anysearch_language: str = "",
                        anysearch_params: dict | None = None,
                        depth: int = 1,
                       history: str = "") -> dict:
    _start = time.monotonic()
    rewritten_query = ""
    search_need = "general"
    out_subqueries: list[dict] = []
    if not google_ai_only:
        # D2: one combined pass → standalone rewrite + search-need label.
        # Never suppresses search — the label only biases engine boosts below.
        rewritten_query, search_need = await asyncio.gather(
            rewrite_query(query, history), classify_need(query),
        )
    effective_query = rewritten_query or query
    engines_used: list[str] = []
    per_engine: dict[str, list[dict]] = {}
    _seen_urls: set[str] = set()
    engine_totals: dict[str, int] = {}
    ai_answer = ""
    follow_up = ""

    async def _gai_search():
        gai = await get_gai_client(cdp_url)
        if not gai:
            return None
        try:
            r = await asyncio.wait_for(gai.search(effective_query, search_prompt=search_prompt,
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
                _h = _host_of(s["url"])
                if _h and ((include_domains and not any(
                        d.lower() in _h for d in include_domains)) or
                        (exclude_domains and any(
                            d.lower() in _h for d in exclude_domains))):
                    continue
                per_engine.setdefault("google-ai-mode", []).append({
                    "title": s["title"], "url": s["url"],
                    "snippet": s.get("snippet", ""),
                    "source": _netloc_of(s["url"]) if s.get("url") else "",
                    "engine": "google-ai-mode",
                    "rank": len(per_engine.get("google-ai-mode", [])),
                })
    else:
        gai_future = asyncio.create_task(_gai_search())
        queries = [effective_query]
        if query_expand:
            if depth >= 2:
                # D1: LLM subquery decomposition for wider coverage; falls back
                # to the heuristic expansion when the LLM pass fails.
                subq = await decompose_query(effective_query)
                out_subqueries = subq
                if len(subq) > 1:
                    queries = [s["query"] for s in subq][:4]
            else:
                out_subqueries = []
                expanded = await expand_query(effective_query)
                queries = expanded[:4]
        ddg_count = max(count * 2 // len(queries), 5)

        async def _ddg_search(q, n, **kw):
            try:
                return await search_ddg(q, n, **kw)
            except Exception:
                return []

        async def _reddit_search(q, n, **kw):
            try:
                return await search_reddit(q, n, **kw)
            except Exception:
                return []

        # Run each engine with the best query variants for broader recall
        async def _multi_search(search_fn, variants, **kw):
            """Run a search function across multiple query variants and merge results.

            Returns a dict with ``results`` (merged list) and ``total_available``
            (0 if the underlying function only returns a plain list).
            """
            results = await asyncio.gather(
                *[search_fn(q, **kw) for q in variants],
                return_exceptions=True,
            )
            out = []
            total_available = 0
            for r in results:
                if isinstance(r, dict):
                    out.extend(r.get("results", []))
                    if r.get("total_available", 0) > total_available:
                        total_available = r["total_available"]
                elif isinstance(r, list):
                    out.extend(r)
            return {"results": out, "total_available": total_available}

        multi_variants = queries[:2]  # original query + best expanded variant

        tr_map = {"d": "day", "w": "week", "m": "month", "y": "year"}
        tavily_tr = tr_map.get(timelimit, "")

        ddg_tasks = {}
        for i, q in enumerate(queries):
            k = f"ddg_{i}"
            ddg_tasks[k] = asyncio.create_task(_ddg_search(
                q, ddg_count, search_type=search_type, backend=backend,
                timelimit=timelimit, page=page, region=region, safesearch=safesearch,
                size=size, color=color, type_image=type_image, layout=layout,
                license_image=license_image, resolution=resolution, duration=duration,
                license_videos=license_videos))

        tasks = {
            "rss": asyncio.create_task(_multi_search(search_google_rss, multi_variants,
                count=count, region=region, timelimit=timelimit)),
            "tavily": asyncio.create_task(_multi_search(search_tavily, multi_variants,
                n=count, topic=tavily_topic, time_range=tavily_tr or tbs,
                search_depth=tavily_depth, include_raw_content=True,
                start_date=start_date, end_date=end_date,
                include_domains=include_domains, exclude_domains=exclude_domains)),
            "wiki": asyncio.create_task(_multi_search(search_wikipedia, multi_variants,
                count=min(count, 8), language=language)),
            "reddit": asyncio.create_task(_multi_search(
                lambda q, **kw: _reddit_search(q, count, **kw), multi_variants,
                subreddit=reddit_subreddit or None, include_comments=reddit_comments,
                include_ai_summary=True)),
            "arxiv": asyncio.create_task(_multi_search(search_arxiv, multi_variants,
                count=min(count, 10))),
            "anysearch": asyncio.create_task(_multi_search(search_anysearch, multi_variants,
                count=min(count, 20), domain=domain,
                tag=anysearch_tag, zone=anysearch_zone, language=anysearch_language,
                params=anysearch_params)),
            "tinyfish": asyncio.create_task(_multi_search(tinyfish_search, multi_variants,
                count=min(count, 50), domain_type=_detect_tinyfish_type(query), goal=query)),
            "brave": asyncio.create_task(_multi_search(search_brave, multi_variants[:1],
                count=min(count, 10), timelimit=timelimit)),
            **ddg_tasks,
        }
        done = await asyncio.wait_for(
            asyncio.gather(*tasks.values(), return_exceptions=True), timeout=90)
        done_map = dict(zip(tasks.keys(), done))
        # D2 need-bias: academic need overrides the detected intent so the
        # scholarly vertical lanes fire; discussion keeps the detected intent
        # (reddit is always in the fan-out already).
        _intent = detect_intent(effective_query)
        if search_need == "academic" and _intent != "paper":
            _intent = "paper"
        _verticals = [v for v in verticals_for(_intent, effective_query)
                      if v not in ("wikipedia", "arxiv", "news")]
        if _verticals:
            try:
                v_results = await asyncio.gather(
                    *[asyncio.wait_for(vertical_run(_v, effective_query), timeout=5) for _v in _verticals],
                    return_exceptions=True,
                )
            except BaseException:
                v_results = [[] for _ in _verticals]
            for _v, r in zip(_verticals, v_results):
                done_map[f"vert_{_v}"] = [] if isinstance(r, BaseException) else r
        for key in list(("rss", "tavily", "reddit", "wiki", "arxiv", "anysearch", "tinyfish", "brave")) + list(ddg_tasks.keys()) + [f"vert_{v}" for v in _verticals]:
            if key not in done_map:
                continue
            val = done_map[key]
            if isinstance(val, BaseException):
                continue
            if isinstance(val, dict):
                results_list = val.get("results", [])
                ta = val.get("total_available", 0)
            elif isinstance(val, list):
                results_list = val
                ta = 0
            else:
                continue
            if ta > 0:
                eng_name = "duckduckgo" if key.startswith("ddg") else (
                    "google-news-rss" if key == "rss" else (
                        "wikipedia" if key == "wiki" else key))
                engine_totals[eng_name] = ta
            for r in results_list:
                if isinstance(r, dict) and "error" not in r and r.get("url"):
                    # Post-merge domain scoping (landscape 🥉): tavily filters
                    # natively, but DDG/brave/rss/reddit/verticals don't —
                    # apply include/exclude uniformly here.
                    _h = _host_of(r["url"])
                    if _h and ((include_domains and not any(
                            d.lower() in _h for d in include_domains)) or
                            (exclude_domains and any(
                                d.lower() in _h for d in exclude_domains))):
                        continue
                    if key != "reddit" or r.get("engine") in ("reddit", "reddit-comment", "reddit-ai-summary"):
                        new_eng = "duckduckgo" if key.startswith("ddg") else (
                            "google-news-rss" if key == "rss" else (
                                "wikipedia" if key == "wiki" else key))
                        if key.startswith("vert_"):
                            new_eng = key[5:]
                        # Preserve sub-engine (e.g., reddit-ai-summary, reddit-comment)
                        cur = r.get("engine")
                        if not cur or cur == key or cur == new_eng:
                            r["engine"] = new_eng
                        entry = dict(r)
                        entry["rank"] = len(per_engine.get(new_eng, []))
                        per_engine.setdefault(new_eng, []).append(entry)
        eng = {"rss": "google-news-rss", "tavily": "tavily", "reddit": "reddit", "wiki": "wikipedia", "arxiv": "arxiv", "anysearch": "anysearch", "tinyfish": "tinyfish", "brave": "brave"}
        for key, name in eng.items():
            val = done_map.get(key)
            if isinstance(val, dict):
                rl = val.get("results", [])
            elif isinstance(val, list):
                rl = val
            else:
                continue
            if any(isinstance(r, dict) and "error" not in r for r in rl):
                engines_used.append(name)
        for _v in _verticals:
            if any(isinstance(r, dict) and "error" not in r
                   for r in done_map.get(f"vert_{_v}", [])):
                engines_used.append(_v)
        if any(isinstance(done_map.get(k), (list, dict)) for k in ddg_tasks):
            engines_used.append("duckduckgo")

        try:
            try:
                completed, _ = await asyncio.wait([gai_future], timeout=300)
            except BaseException:
                completed = set()
            ai_answer = ""
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
                    _seen_urls |= {norm_key(h["url"]) for hits in per_engine.values() for h in hits}
                    for s in rd.get("sources", []):
                        _h = _host_of(s["url"])
                        if _h and ((include_domains and not any(
                                d.lower() in _h for d in include_domains)) or
                                (exclude_domains and any(
                                    d.lower() in _h for d in exclude_domains))):
                            continue
                        if norm_key(s["url"]) not in _seen_urls:
                            _seen_urls.add(norm_key(s["url"]))
                            per_engine.setdefault("google-ai-mode", []).append({
                                "title": s["title"], "url": s["url"],
                                "snippet": s.get("snippet", ""),
                                "source": _netloc_of(s["url"]) if s.get("url") else "",
                                "engine": "google-ai-mode",
                                "rank": len(per_engine.get("google-ai-mode", [])),
                            })
            else:
                gai_future.cancel()
        finally:
            if not gai_future.done():
                gai_future.cancel()
    # Track which engines were expected but didn't contribute
    engine_blocked = [] if google_ai_only else sorted(_ALL_ENGINES - set(engines_used))

    if per_engine:
        intent = detect_intent(effective_query)
        if search_need == "academic" and intent != "paper":
            intent = "paper"
        merged = merge_base(per_engine, query, intent)
        if merged:
            # ponytail: cap the rerank pool at the top 120 by base score —
            # the cross-encoder correlates with the pre-score, and the bottom
            # of a 189-entry pool costs ~4-5s of GPU time for near-zero
            # top-10 impact; raise if long-tail recall ever matters.
            if len(merged) > 120:
                merged = sorted(merged, key=lambda r: r["score"], reverse=True)[:120]
            # Cross-encoder scores from the shared rust worker, blended 60/40
            # ponytail: truncate snippets to 1200 chars before rerank — 94-entry
            # merged pools were feeding ~500K chars (~130K tokens, 15-50s) for
            # a relevance signal that the first ~300 tokens already carry.
            for m in merged:
                s = m.get("snippet") or ""
                if len(s) > 1200:
                    m["snippet"] = s[:1200]
            with_idx = [dict(m, idx=i) for i, m in enumerate(merged)]
            semantic = await _rerank(query, with_idx, top_k=len(with_idx))
            if semantic:
                by_idx = {}
                for s in semantic:
                    if s.get("idx") is not None:
                        v = s.get("_rerank")
                        if v is None:
                            v = s.get("_rel") or s.get("_relevance") or s.get("_hybrid")
                        if v is not None:
                            by_idx[s["idx"]] = v
                blend(merged, [by_idx.get(i) for i in range(len(merged))])
            apply_coverage(query, merged)
            apply_authority(query, intent, merged)
            apply_six_signal(effective_query, intent, merged, ai_answer)
            deduped = finalize(merged, count + 1)
            total_merged = merged_total(per_engine)
            weak = is_weak(deduped, total_merged)
        else:
            deduped = []
            total_merged = 0
            weak = True
        deduped = _normalize_scores(deduped)
        related = _related_queries(query, deduped)
    else:
        deduped = []
        total_merged = 0
        weak = True
        related = []
    out: dict = {"success": True, "engines_used": engines_used,
                 "engine_blocked": engine_blocked,
                 "engine_totals": engine_totals,
                 "ai_answer": ai_answer, "follow_up": follow_up,
                 "related_queries": related,
                 "merged_total": total_merged, "weak": weak,
                 "results": deduped[:count], "total": len(deduped[:count]),
                 "duration_ms": round((time.monotonic() - _start) * 1000)}
    if rewritten_query and rewritten_query != query:
        out["rewritten_query"] = rewritten_query
    out["search_need"] = search_need
    if out_subqueries:
        out["subqueries"] = out_subqueries
    if _reranker_disabled():
        out["reranker"] = "disabled — results are engine-ranked only (not reranked). " \
            "Enable with: rm ~/.local/share/reranker-rust/disabled"
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
        *[fetch_url(url, max_chars=32768, fast=True) for url in urls], return_exceptions=True)
    # donsetch mod.rs:583 — dead links keep their SERP title/snippet with a 0.5
    # score demote; bot walls (challenge verdict) stay untouched at full score.
    # Both stay in the pool (the LLM still sees title+snippet), only real
    # successes carry fetched content and enter dedup/rerank.
    kept: list[dict] = []
    demoted: list[dict] = []
    by_url = {r.get("url"): r for r in results if r.get("url")}
    for f in fetched:
        if isinstance(f, dict) and f.get("success"):
            kept.append(f)
            continue
        url = f.get("url") if isinstance(f, dict) else None
        orig = by_url.get(url)
        if not orig:
            continue
        entry = {
            **orig,
            "content": orig.get("snippet") or "",
            "success": False,
            "dead": True,
        }
        if isinstance(f, dict) and f.get("verdict") in ("CHALLENGE", "TERMINAL"):
            entry["relevance_score"] = orig.get("relevance_score") or 0.5
        else:
            entry["relevance_score"] = round((orig.get("relevance_score") or 0.5) * 0.5, 4)
        demoted.append(entry)
    fetched = kept
    # No Wikipedia summary re-fetch: the full 32K fetch already carries the
    # article, and a summary extract is longer than it almost never (it would
    # only prepend when summary > full content). The old block spent a REST
    # call per wiki page on work that was discarded ~always.
    if len(fetched) > 1:
        texts = [(f.get("content", "") or "")[:300] + " " +
                 (f.get("title", "") or "")[:100] for f in fetched]
        item_emb = await _embed(texts, "passage")
        if item_emb:
            deduped = []
            seen_emb: list[list[float]] = []
            for f, emb in zip(fetched, item_emb):
                is_dup = any(_cosine_sim(emb, se) > 0.92 for se in seen_emb)  # match _dedup_rank default
                if not is_dup:
                    deduped.append(f)
                    seen_emb.append(emb)
            fetched = deduped
        fetched = await _rerank(query, fetched, top_k=depth * 2)
        fetched = _normalize_scores(fetched)
    # reranker saw the full fetch; give the LLM a bounded digest per page
    # (smolagents truncate-with-marker pattern: explicit cut notice so the
    # LLM never mistakes a truncated page for a short one)
    for f in fetched:
        c = f.get("content") or ""
        if len(c) > 5000:
            f["content"] = c[:5000] + "\n\n[Content truncated at 5000 chars]"
        elif c:
            f["content"] = c
    result = {"fetched_content": fetched}
    if demoted:
        result["fetched_content"] = result["fetched_content"] + demoted
    if _reranker_disabled():
        result["reranker"] = "disabled — results are engine-ranked only (not reranked). " \
            "Enable with: rm ~/.local/share/reranker-rust/disabled"
    return result
