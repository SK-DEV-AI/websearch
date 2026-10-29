"""Reddit gatherer — DOM search + CDP comment fetch + AI summary extraction.

Architecture (per-call page, like GAI):
- Each ``search_reddit()`` call creates its own CDP page and works inside it.
- Posts come from scraping the RENDERED search page
  (``div[data-testid="search-post-unit"]`` cards). Reddit's ``/search.json``
  answers 403 to browser-origin requests even with a live session cookie, so
  the JSON search path was removed (measured 2026-10-29).
- Comments still use in-browser ``fetch()`` of ``/r/<sub>/comments/<id>.json``,
  which is alive for some posts (best effort).
- The page is closed in ``finally`` so multiple searches are independent.
- Session cookies are shared at the browser level (Helium CDP), so every
  page inherits the user's Reddit session automatically.
- Falls back gracefully (empty results) if CDP is unavailable.
"""

from __future__ import annotations

import asyncio
import html as _html
import json
import logging
import time
from typing import Any

from urllib.parse import quote as _quote

logger = logging.getLogger("reddit")

_reddit_sem = asyncio.Semaphore(3)


async def _new_reddit_page():
    """Create a hidden CDP page navigated to reddit.com for same-origin CORS."""
    page = None
    try:
        from cdp_client import get_cdp_session
        session = await get_cdp_session()
        if not session:
            return None
        page = await session.create_page()
        # domcontentloaded (not "load") — reddit.com homepage is heavy; fetch()
        # only needs the document interactive, full load adds seconds for zero gain
        await page.goto("https://www.reddit.com/", wait_until="domcontentloaded",
                        referrer="https://www.google.com/")
        deadline = time.monotonic() + 10
        while time.monotonic() < deadline:
            try:
                alive = await page.evaluate("document.readyState")
                if alive in ("interactive", "complete"):
                    break
            except Exception:
                pass
            await asyncio.sleep(0.3)
        return page
    except Exception as e:
        logger.warning("Failed to create Reddit page: %s", e)
        if page is not None:
            try:
                await page.close()
            except Exception:
                pass
        return None


_LAST_FETCH_ERROR = ""


async def _cdp_fetch(page, path: str) -> dict | None:
    """Execute fetch() from within the given CDP page context.

    The page must be on a same-origin URL (reddit.com) so CORS doesn't
    block credentialed requests. Session cookies are inherited from Helium.

    On failure the reason is recorded in ``_LAST_FETCH_ERROR`` so the caller
    can report WHY reddit returned nothing (measured 2026-10-27: reddit
    answers 403 to anonymous /search.json requests, which used to look
    exactly like "no results").
    """
    global _LAST_FETCH_ERROR
    _LAST_FETCH_ERROR = ""
    url = f"https://www.reddit.com{path}"
    try:
        result = await page.evaluate(f'''
            (async () => {{
                try {{
                    const ctl = new AbortController();
                    const timer = setTimeout(() => ctl.abort(), 25000);
                    try {{
                        const r = await fetch({json.dumps(url)}, {{signal: ctl.signal}});
                        if (r.status < 200 || r.status >= 300) {{
                            return JSON.stringify({{_error: "HTTP " + r.status}});
                        }}
                        const text = await r.text();
                        return JSON.stringify({{_data: text}});
                    }} finally {{
                        clearTimeout(timer);
                    }}
                }} catch(e) {{
                    return JSON.stringify({{_error: e.message}});
                }}
            }})()
        ''')
        if not result:
            return None
        parsed = json.loads(result) if isinstance(result, str) else result
        if isinstance(parsed, dict) and parsed.get("_error"):
            logger.warning("CDP fetch error for %s: %s", path, parsed["_error"])
            _LAST_FETCH_ERROR = str(parsed["_error"])
            return None
        data_str = parsed.get("_data") if isinstance(parsed, dict) else None
        if data_str:
            return json.loads(data_str)
        return None
    except Exception as e:
        logger.warning("CDP fetch exception for %s: %s", path, e)
        _LAST_FETCH_ERROR = f"{type(e).__name__}: {e}"
        return None


# ── AI summary extraction ───────────────────────────────────────────────────

async def _get_ai_summary(page, query: str) -> dict | None:
    """Fetch Reddit's AI-generated answer for *query* via *page*.

    1. Navigates to the Reddit search results page where the AI summary renders
       inline as a "What people are saying" section.
    2. If a full answers page is linked ("See More"), navigates there and
       extracts the full AI answer from the ``<guides-response-container-streaming>``
       custom element's open shadow root.

    Returns a dict with keys *summary*, *full_answer*, and *source_subreddits*,
    or *None* if unavailable.
    """
    result: dict[str, Any] = {}

    # --- Step 1: navigate to search results and poll for AI summary ---
    search_url = f"https://www.reddit.com/search/?q={_quote(query)}"
    await page.goto(search_url, wait_until="domcontentloaded", referrer="https://www.google.com/")

    # Poll for the AI summary AND stream settlement (up to 15s).
    # The marker appears while the answer is still streaming — extract only
    # after the section stops growing (3x no-growth), so no partial answers.
    # Some layouts skip the inline section: the AI entity then lives ONLY
    # behind the /answers/ "See More" link (renders ~4s, no marker ever) —
    # watch for that link too, or step 2 never fires and we return None.
    deadline = time.monotonic() + 15
    section_len = 0
    stable = 0
    no_signal = 0
    section_js = (
        "(() => {"
        "  const t = document.body?.innerText || '';"
        "  const seeMore = !!document.querySelector('a[href*=\"/answers/\"]');"
        "  const sIdx = t.indexOf('What people are saying');"
        "  if (sIdx < 0) return {ready: false, len: 0, seeMore};"
        "  const endIdx = t.indexOf('\\nPosts\\n', sIdx);"
        "  const sec = endIdx > 0 ? t.substring(sIdx, endIdx) : t.substring(sIdx, 2500);"
        "  return {ready: true, len: sec.length, seeMore};"
        "})()"
    )
    while time.monotonic() < deadline:
        state = await page.evaluate(section_js)
        if state and state.get("ready"):
            no_signal = 0
            cur = state.get("len", 0)
            if cur == section_len:
                stable += 1
                if stable >= 3:
                    break
            else:
                section_len = cur
                stable = 0
        elif state and state.get("seeMore"):
            # AI entity exists but inline section isn't rendering — the
            # extraction below grabs answers_url and step 2 takes over.
            break
        else:
            # No marker AND no answers link within the grace window: no AI
            # entity for this query — don't burn the deadline on nothing.
            no_signal += 1
            if no_signal >= 8:
                break
        await asyncio.sleep(0.5)

    extracted = await page.evaluate('''
(() => {
    const r = {};
    const body = document.body?.innerText || '';

    // 1b first: the /answers/ link must survive even when the inline
    // section never renders (marker absent) — step 2 needs it.
    // 1b — find the answers-page link
    const seeMore = document.querySelector('a[href*="/answers/"]');
    if (seeMore) r['answers_url'] = seeMore.href;

    // 1a — extract the "What people are saying" section from innerText
    const startIdx = body.indexOf('What people are saying');
    if (startIdx < 0) return JSON.stringify(r);
    const fromStart = body.substring(startIdx);
    // Cut at the "Posts" heading that follows the AI summary
    const endIdx = fromStart.indexOf('\\nPosts\\n');
    const summary = endIdx > 0 ? fromStart.substring(0, endIdx) : fromStart.substring(0, 2500);
    r['summary'] = summary.trim();

    // 1c — parse sources line
    const srcMatch = summary.match(/Sources:\\s*(.+?)(?:\\n|$)/i);
    if (srcMatch) {
        r['source_subreddits'] = srcMatch[1]
            .split(/[,+]/)
            .map(s => s.trim())
            .filter(Boolean);
    }

    return JSON.stringify(r);
})()
''')

    if not extracted:
        return None

    info = json.loads(extracted) if isinstance(extracted, str) else extracted
    result["summary"] = info.get("summary")
    result["source_subreddits"] = info.get("source_subreddits", [])
    result["answers_url"] = info.get("answers_url")

    # --- Step 2: navigate to the full answers page for shadow-root content ---
    answers_url = info.get("answers_url")
    if answers_url:
        try:
            # defense-in-depth: page-controlled URL still passes SSRF gate
            from security import validate_url
            await validate_url(answers_url)
            # "load" would block on reddit's heavy homepage scripts; the
            # shadow-root stability poll below handles the streaming wait
            await page.goto(answers_url, wait_until="domcontentloaded", referrer=search_url)

            # Poll for streaming content to appear in shadow root (up to 20s).
            # After first chunk appears (>100 chars), wait for content-length
            # stability (3x no-growth = stream complete) before extracting.
            full = None
            prev = 0
            stable = 0
            no_chunk = 0
            deadline = time.monotonic() + 20
            while time.monotonic() < deadline:
                chunk = await page.evaluate('''
(() => {
    const sc = document.querySelector('guides-response-container-streaming');
    if (!sc || !sc.shadowRoot) return null;
    const text = sc.shadowRoot.textContent || '';
    return text.length > 100 ? text : null;
})()
''')
                if chunk and isinstance(chunk, str):
                    no_chunk = 0
                    cur = len(chunk)
                    if cur == prev:
                        stable += 1
                        if stable >= 6:
                            full = chunk
                            break
                    else:
                        prev = cur
                        stable = 0
                else:
                    # Fresh /answers/ entities cold-build server-side: the
                    # first chunk can take ~10s. Give it room; the 20s
                    # deadline above stays as the backstop.
                    no_chunk += 1
                    if no_chunk >= 20:
                        break
                await asyncio.sleep(0.5)
            if full and isinstance(full, str) and len(full.strip()) > 100:
                result["full_answer"] = full.strip()
        except Exception as e:
            logger.warning("Answers-page extraction failed: %s", e)

    return result if result.get("summary") or result.get("full_answer") else None


# ── Data model helpers ──────────────────────────────────────────────────────

def _parse_comment(c: dict) -> dict:
    """Convert a Reddit API comment object to our format."""
    return {
        "id": c.get("id", ""),
        "author": c.get("author", ""),
        "body": _html.unescape((c.get("body") or "")[:600]),
        "score": c.get("score", 0),
    }


def _extract_comments(listing: dict, per_post: int) -> list[dict]:
    """Pull top-level comments (one level of replies) from a comments listing."""
    out: list[dict] = []

    def walk(children: list, depth: int):
        for ch in children:
            if not isinstance(ch, dict) or ch.get("kind") != "t1":
                continue
            d = ch.get("data", {})
            body = d.get("body") or ""
            if not body or body in ("[deleted]", "[removed]"):
                continue
            out.append(_parse_comment(d))
            if len(out) >= per_post:
                return
            replies = d.get("replies")
            if depth == 0 and isinstance(replies, dict):
                walk(replies.get("data", {}).get("children", []), depth + 1)

    children = listing.get("data", {}).get("children", []) if isinstance(listing, dict) else []
    walk(children, 0)
    return out[:per_post]


# ── DOM-based search (JSON API is 403 for browser-origin requests) ───────────

async def _dom_search_posts(page, query: str, subreddit: str | None = None,
                            sort: str = "relevance",
                            time_filter: str = "all") -> list[dict]:
    """Scrape rendered Reddit search results from the DOM.

    Reddit's ``/search.json`` (and ``.rss``, pullpush) now answer 403 to
    browser-origin requests even with a live ``reddit_session`` cookie, so the
    JSON path is dead. The rendered search page DOES contain real results in
    ``div[data-testid="search-post-unit"]`` cards, with metadata in a
    ``search-telemetry-tracker``'s ``data-faceplate-tracking-context`` JSON
    and vote/comment counts in ``<faceplate-number number="N">``.

    Returns a list of dicts: id, title, href, subreddit, author, score,
    num_comments. Empty list on any failure (caller reports honest error).
    """
    if subreddit:
        search_url = (f"https://www.reddit.com/r/{subreddit}/search/?q={_quote(query)}"
                      f"&restrict_sr=on&sort={_quote(sort)}&t={_quote(time_filter)}")
    else:
        search_url = (f"https://www.reddit.com/search/?q={_quote(query)}"
                      f"&sort={_quote(sort)}&t={_quote(time_filter)}")
    try:
        await page.goto(search_url, wait_until="domcontentloaded",
                        referrer="https://www.google.com/")
    except Exception as e:
        logger.warning("DOM search navigation failed for %s: %s", query, e)
        return []

    # Poll for result cards (rendered lazily, typically 4-8s).
    deadline = time.monotonic() + 10
    while time.monotonic() < deadline:
        try:
            cnt = await page.evaluate(
                'document.querySelectorAll(\'[data-testid="search-post-unit"]\').length')
            if isinstance(cnt, int) and cnt >= 1:
                break
        except Exception:
            pass
        await asyncio.sleep(0.5)

    js = r'''
(() => {
    const out = [];
    const units = document.querySelectorAll('[data-testid="search-post-unit"]');
    for (const u of units) {
        let ctx = {};
        try {
            const t = u.querySelector('search-telemetry-tracker');
            if (t) ctx = JSON.parse(t.getAttribute('data-faceplate-tracking-context') || '{}');
        } catch (e) { ctx = {}; }
        const titleEl = u.querySelector('[data-testid="post-title"]');
        const title = (titleEl && (titleEl.getAttribute('aria-label') || titleEl.innerText || '')) || '';
        const href = (titleEl && titleEl.getAttribute('href')) || '';
        const nums = Array.from(u.querySelectorAll('faceplate-number'));
        const parseNum = (el) => {
            if (!el) return 0;
            const n = el.getAttribute('number');
            if (n !== null && n !== '') { const v = parseInt(n, 10); if (!isNaN(v)) return v; }
            const v2 = parseInt((el.innerText || '').trim(), 10);
            return isNaN(v2) ? 0 : v2;
        };
        out.push({
            id: String(ctx.post && ctx.post.id ? ctx.post.id : '').replace(/^t3_/, ''),
            title: String(title).trim(),
            href: String(href).trim(),
            subreddit: String((ctx.subreddit && ctx.subreddit.name) || ''),
            author: String((ctx.profile && ctx.profile.name) || ''),
            score: parseNum(nums[0]),
            num_comments: parseNum(nums[1]),
        });
    }
    return JSON.stringify(out);
})()
'''
    try:
        result = await page.evaluate(js)
    except Exception as e:
        logger.warning("DOM search evaluate failed: %s", e)
        return []
    if not result:
        return []
    try:
        parsed = json.loads(result) if isinstance(result, str) else result
    except Exception as e:
        logger.warning("DOM search JSON parse failed: %s", e)
        return []
    if not isinstance(parsed, list):
        return []
    return [p for p in parsed if isinstance(p, dict) and (p.get("title") or p.get("href"))]


# ── Main entry point ────────────────────────────────────────────────────────

async def search_reddit(query: str, count: int = 10, subreddit: str | None = None,
                        comments_per_post: int = 4, include_comments: bool = True,
                        include_ai_summary: bool = True, sort: str = "relevance",
                        time_filter: str = "all") -> list[dict]:
    """Search Reddit posts + comments + AI summary via CDP.

    Each call creates its own hidden CDP page, does all work, and closes it
    in ``finally`` — no shared state means concurrent searches are safe.

    Falls back to empty results if CDP is unavailable.
    """
    async with _reddit_sem:
        page = await _new_reddit_page()
        if not page:
            return {"results": [], "error": "reddit: CDP page unavailable (is Helium running with remote debugging?)"}
        try:
            # --- Phase 1: AI summary (navigates the page for AI content) ------
            ai_summary_result = None
            if include_ai_summary:
                try:
                    ai_summary_result = await _get_ai_summary(page, query)
                except Exception as e:
                    logger.warning("AI summary failed: %s", e)

            # --- Phase 2: search results via DOM scraping -------------------
            # Reddit's /search.json (and .rss, pullpush) now answer 403 to
            # browser-origin requests even with a live reddit_session, so the
            # JSON path is dead. Scrape the rendered search page instead.
            dom_posts = await _dom_search_posts(page, query, subreddit,
                                               sort=sort, time_filter=time_filter)

            # --- Phase 3: fetch comments (concurrent) ------------------------
            post_comments: dict[str, list[dict]] = {}
            if include_comments and dom_posts:
                posts_raw = dom_posts

                async def _cmts(post: dict) -> None:
                    sub = post.get("subreddit") or ""
                    pid = post.get("id") or ""
                    if not sub or not pid:
                        return
                    cdata = await _cdp_fetch(page, f"/r/{sub}/comments/{pid}.json")
                    if isinstance(cdata, list) and len(cdata) >= 2:
                        comments = _extract_comments(cdata[1], comments_per_post)
                        if comments:
                            post_comments[post["id"]] = comments

                await asyncio.gather(*[_cmts(p) for p in posts_raw[:count]], return_exceptions=True)

            # --- Phase 4: build unified results ------------------------------
            results: list[dict] = []

            if ai_summary_result:
                raw = ai_summary_result.get("full_answer") or ai_summary_result.get("summary") or ""
                summary = raw.strip()
                if summary:
                    subreddits = ai_summary_result.get("source_subreddits", [])
                    sub_str = f" (sources: {', '.join(subreddits)})" if subreddits else ""
                    ai_url = ai_summary_result.get("answers_url") or f"https://www.reddit.com/search/?q={_quote(query)}"
                    results.append({
                        "title": f"Reddit AI summary{ sub_str }",
                        "url": ai_url,
                        "snippet": summary[:1600],
                        "source": "reddit.com",
                        "engine": "reddit-ai-summary",
                    })

            if not dom_posts:
                return {"results": results, "error": f"reddit: DOM search returned no posts (page may not have rendered; last fetch err: {_LAST_FETCH_ERROR or 'none'})"}

            posts = dom_posts

            for p in posts:
                permalink = p.get("href", "")
                if permalink and not permalink.startswith("http"):
                    url = f"https://www.reddit.com{permalink}"
                else:
                    url = permalink or ""
                sub = p.get("subreddit", "")
                title = p.get("title", "")
                snippet = title
                if sub:
                    snippet = f"[r/{sub}] {snippet}"
                comments = post_comments.get(p.get("id"), [])
                if comments:
                    lines = [f"\n\nTop comments:"]
                    for c in comments:
                        lines.append(f"  u/{c['author']}: {c['body']}")
                    snippet = (snippet + "\n".join(lines))[:1600]

                entry: dict[str, Any] = {
                    "title": title,
                    "url": url,
                    "snippet": snippet,
                    "source": "reddit.com",
                    "engine": "reddit",
                    "author": p.get("author", ""),
                    "score": p.get("score", 0),
                    "num_comments": p.get("num_comments", 0),
                    "subreddit": sub,
                }
                if comments:
                    entry["comments"] = comments
                results.append(entry)

                for c in comments:
                    cid = c.get("id", "")
                    results.append({
                        "title": f"r/{sub} comment by u/{c['author']}",
                        "url": f"{url}#t1_{cid}" if url else "",
                        "snippet": c.get("body", ""),
                        "source": "reddit.com",
                        "engine": "reddit-comment",
                        "author": c.get("author", ""),
                        "score": c.get("score", 0),
                        "subreddit": sub,
                    })

            return results
        finally:
            try:
                await page.close()
            except Exception:
                pass
            try:
                from search_gai import _cleanup_orphan_tabs
                asyncio.ensure_future(_cleanup_orphan_tabs())
            except Exception:
                pass


    # ── CLI smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    r = asyncio.run(search_reddit("python programming", count=2, comments_per_post=1))
    print(f"Results: {len(r)}")
    for e in r[:6]:
        print(f"  [{e.get('engine','')}] {e.get('title','')[:70]}")
