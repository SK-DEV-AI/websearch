"""Reddit gatherer — CDP in-browser fetch + AI summary extraction.

Architecture (per-call page, like GAI):
- Each ``search_reddit()`` call creates its own CDP page, navigates to
  reddit.com (for same-origin CORS), then uses in-browser ``fetch()`` for
  post/comment data and navigates the same page for the AI summary.
- The page is closed in ``finally`` so multiple searches are independent.
- Session cookies are shared at the browser level (Helium CDP), so every
  page inherits the user's authenticated Reddit session automatically.
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
    try:
        from cdp_client import get_cdp_session
        session = await get_cdp_session()
        if not session:
            return None
        page = await session.create_page()
        await page.goto("https://www.reddit.com/", referrer="https://www.google.com/")
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
        return None


async def _cdp_fetch(page, path: str) -> dict | None:
    """Execute fetch() from within the given CDP page context.

    The page must be on a same-origin URL (reddit.com) so CORS doesn't
    block credentialed requests. Session cookies are inherited from Helium.
    """
    url = f"https://www.reddit.com{path}"
    try:
        result = await page.evaluate(f'''
            (async () => {{
                try {{
                    const r = await fetch({json.dumps(url)});
                    if (r.status < 200 || r.status >= 300) {{
                        return JSON.stringify({{_error: "HTTP " + r.status}});
                    }}
                    const text = await r.text();
                    return JSON.stringify({{_data: text}});
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
            return None
        data_str = parsed.get("_data") if isinstance(parsed, dict) else None
        if data_str:
            return json.loads(data_str)
        return None
    except Exception as e:
        logger.warning("CDP fetch exception for %s: %s", path, e)
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
    await page.goto(search_url, referrer="https://www.google.com/")

    # Poll for the AI summary to appear (up to 15s) instead of fixed sleep
    deadline = time.monotonic() + 15
    while time.monotonic() < deadline:
        ready = await page.evaluate('document.body?.innerText?.includes("What people are saying")')
        if ready:
            break
        await asyncio.sleep(0.5)

    extracted = await page.evaluate('''
(() => {
    const r = {};
    const body = document.body?.innerText || '';

    // 1a — extract the "What people are saying" section from innerText
    const startIdx = body.indexOf('What people are saying');
    if (startIdx < 0) return JSON.stringify(r);
    const fromStart = body.substring(startIdx);
    // Cut at the "Posts" heading that follows the AI summary
    const endIdx = fromStart.indexOf('\\nPosts\\n');
    const summary = endIdx > 0 ? fromStart.substring(0, endIdx) : fromStart.substring(0, 2500);
    r['summary'] = summary.trim();

    // 1b — find the answers-page link
    const seeMore = document.querySelector('a[href*="/answers/"]');
    if (seeMore) r['answers_url'] = seeMore.href;

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
            await page.goto(answers_url, referrer=search_url)

            # Poll for streaming content to appear in shadow root (up to 20s).
            # After first chunk appears (>100 chars), wait for content-length
            # stability (3x no-growth = stream complete) before extracting.
            full = None
            prev = 0
            stable = 0
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
                    cur = len(chunk)
                    if cur == prev:
                        stable += 1
                        if stable >= 3:
                            full = chunk
                            break
                    else:
                        prev = cur
                        stable = 0
                await asyncio.sleep(0.5)
            if full and isinstance(full, str) and len(full.strip()) > 100:
                result["full_answer"] = full.strip()
        except Exception as e:
            logger.warning("Answers-page extraction failed: %s", e)

    return result if result.get("summary") or result.get("full_answer") else None


# ── Data model helpers ──────────────────────────────────────────────────────

def _parse_post(p: dict) -> dict:
    """Convert a Reddit API post object to our result format."""
    return {
        "id": p.get("id"),
        "title": _html.unescape(p.get("title", "")),
        "subreddit": p.get("subreddit", ""),
        "author": p.get("author", ""),
        "score": p.get("score", 0),
        "num_comments": p.get("num_comments", 0),
        "permalink": p.get("permalink", ""),
        "selftext": _html.unescape(p.get("selftext", "") or ""),
        "created_utc": p.get("created_utc", 0),
    }


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
        return []

    try:
        # --- Phase 1: AI summary (navigates the page for AI content) ------
        ai_summary_result = None
        if include_ai_summary:
            try:
                ai_summary_result = await _get_ai_summary(page, query)
            except Exception as e:
                logger.warning("AI summary failed: %s", e)

        # --- Phase 2: fetch search results via in-browser fetch -----------
        limit = max(count, min(count * 5, 100))
        params = {"q": query, "limit": str(limit), "sort": sort, "t": time_filter,
                  "type": "link", "raw_json": "1"}
        if subreddit:
            params["restrict_sr"] = "on"
            path = f"/r/{subreddit}/search.json"
        else:
            path = "/search.json"

        qs = "&".join(f"{k}={_quote(v)}" for k, v in params.items())
        data = await _cdp_fetch(page, f"{path}?{qs}")

        # --- Phase 3: fetch comments (concurrent) ------------------------
        post_comments: dict[str, list[dict]] = {}
        if include_comments and isinstance(data, dict):
            children = data.get("data", {}).get("children", [])
            posts_raw = [c.get("data", {}) for c in children if isinstance(c, dict) and c.get("kind") == "t3"]

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

        if not isinstance(data, dict):
            return results

        children = data.get("data", {}).get("children", [])
        posts = []
        for c in children:
            if not isinstance(c, dict) or c.get("kind") != "t3":
                continue
            posts.append(_parse_post(c.get("data", {})))

        for p in posts:
            permalink = p.get("permalink", "")
            url = f"https://www.reddit.com{permalink}" if permalink else ""
            sub = p.get("subreddit", "")
            snippet = p.get("selftext", "").strip()
            if not snippet:
                snippet = p.get("title", "")
            snippet = f"[r/{sub}] {snippet}"
            comments = post_comments.get(p.get("id"), [])
            if comments:
                lines = [f"\n\nTop comments:"]
                for c in comments:
                    lines.append(f"  u/{c['author']}: {c['body']}")
                snippet = (snippet + "\n".join(lines))[:1600]

            entry: dict[str, Any] = {
                "title": p.get("title", ""),
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


# ── CLI smoke test ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    r = asyncio.run(search_reddit("python programming", count=2, comments_per_post=1))
    print(f"Results: {len(r)}")
    for e in r[:6]:
        print(f"  [{e.get('engine','')}] {e.get('title','')[:70]}")
