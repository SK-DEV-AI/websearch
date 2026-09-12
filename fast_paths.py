"""Fast-path fetches for well-known URL shapes (steal landscape 🥇).

Each path is shape-gated: returns a ready response dict (fetch_url-compatible)
or None to fall through to the generic pipeline. Zero cost when the URL does
not match — the gate is a regex on the URL, no requests issued.

Paths: llms.txt (site root), YouTube transcript (yt-dlp, youtube URLs only),
Reddit .json (comments threads), GitHub API (repo README / raw blob).
"""

import asyncio
import html as html_mod
import json
import re
import time
from urllib.parse import urlparse

from scrapling.fetchers import AsyncFetcher

from security import SecurityError, validate_url as _validate_url

_TXT_CT = "text/plain"


async def _get(url: str, timeout: float = 12) -> str | None:
    # Pre-validate: some callers pass derived URLs (yt-dlp caption c_url)
    # that never went through fetch_url's gate — fail closed here too.
    try:
        await _validate_url(url)
    except SecurityError:
        return None
    try:
        resp = await AsyncFetcher.get(url, timeout=timeout, stealthy_headers=True)
    except Exception:
        return None
    if resp.status != 200:
        return None
    final = getattr(resp, "url", None)
    if final and final != url:
        try:
            await _validate_url(final)
        except SecurityError:
            return None
    body = resp.body
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    return body


def _resp(url: str, content: str, method: str, title: str) -> dict:
    return {"success": True, "url": url, "content": content, "method": method,
            "status": 200, "title": title, "metadata": {}, "content_type": _TXT_CT}


# ── llms.txt ──────────────────────────────────────────────────────────
# LLM-friendly site manifest (TadMSTR/searxng-mcp "llms.txt fast path").
# Probed only for root URLs: one extra request, skipped on 404.

async def _llms_txt(url: str) -> dict | None:
    p = urlparse(url)
    if p.path not in ("", "/"):
        return None
    body = await _get(f"{p.scheme}://{p.netloc}/llms.txt")
    if body is None or len(body) > 2_000_000:
        return None
    return _resp(url, body, "llms.txt", f"llms.txt for {p.netloc}")


# ── YouTube transcript ────────────────────────────────────────────────

_YT_RE = re.compile(
    r"^https?://(?:www\.|m\.)?(?:youtube\.com/watch\?.*?v=|youtu\.be/)([\w-]{11})")


async def _youtube_transcript(url: str) -> dict | None:
    m = _YT_RE.search(url)
    if not m:
        return None
    vid = m.group(1)
    try:
        proc = await asyncio.create_subprocess_exec(
            "yt-dlp", "-J", "--skip-download",
            "--extractor-args", "youtube:player_client=android",
            f"https://www.youtube.com/watch?v={vid}",
            stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.DEVNULL)
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    except Exception:
        return None
    try:
        info = json.loads(out)
    except Exception:
        return None
    caps = info.get("automatic_captions") or info.get("subtitles") or {}
    for lang in ("en", "en-US", "en-GB"):
        if lang in caps:
            c_url = caps[lang][0]["url"]
            break
    else:
        return None
    body = await _get(c_url)
    if body is None:
        return None
    text = _strip_vtt(body)
    if not text.strip():
        return None
    return _resp(url, text, "youtube-transcript",
                 info.get("title") or f"YouTube {vid}")


def _strip_vtt(vtt: str) -> str:
    out: list[str] = []
    buf: list[str] = []
    for line in vtt.splitlines():
        if "-->" in line or line.strip().isdigit() or line.strip() in ("WEBVTT", ""):
            continue
        if line.startswith("["):  # [Music], [Applause]
            continue
        buf.append(line.strip())
        if len(buf) >= 3:
            out.append(" ".join(buf))
            buf = []
    if buf:
        out.append(" ".join(buf))
    return "\n".join(out)


# ── Reddit .json ──────────────────────────────────────────────────────
# Comments threads render as readable markdown via the JSON endpoint.

_REDDIT_RE = re.compile(
    r"^https?://(?:www\.|old\.|new\.)?reddit\.com(/r/[\w-]+/comments/[\w-]+)")


def _clean_body(t: str) -> str:
    t = html_mod.unescape(t or "")
    return re.sub(r"<[^>]+>", "", t).strip()


async def _reddit_json(url: str) -> dict | None:
    m = _REDDIT_RE.search(url)
    if not m:
        return None
    # Same-origin with the www.reddit.com page above: old/new hosts share
    # the same JSON backend, but cross-origin fetch is CORS-blocked
    # ("Failed to fetch") — normalize the host, keep the path.
    json_url = (re.sub(r"^https?://(?:www\.|old\.|new\.)?reddit\.com",
                       "https://www.reddit.com", url.rstrip("/")) + ".json")
    try:
        from cdp_client import get_cdp_session
        session = await get_cdp_session()
        if not session:
            return None
        page = await session.create_page()
        try:
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
            result = await page.evaluate(f'''
                (async () => {{
                    try {{
                        const r = await fetch({json.dumps(json_url)});
                        if (r.status < 200 || r.status >= 300) {{
                            return JSON.stringify({{_error: "HTTP " + r.status}});
                        }}
                        return JSON.stringify({{_data: await r.text()}});
                    }} catch(e) {{
                        return JSON.stringify({{_error: e.message}});
                    }}
                }})()
            ''')
        finally:
            try:
                await page.close()
            except Exception:
                pass
    except Exception:
        return None
    if not result:
        return None
    try:
        parsed = json.loads(result) if isinstance(result, str) else result
        if isinstance(parsed, dict) and parsed.get("_error"):
            return None
        body = parsed.get("_data") if isinstance(parsed, dict) else None
        if not body:
            return None
        listing = json.loads(body)
        post = listing[0]["data"]["children"][0]["data"]
    except Exception:
        return None
    title = post.get("title", "")
    parts = [f"# {title}",
             f"*u/{post.get('author', '?')}* — {post.get('subreddit_name_prefixed', '')}",
             _clean_body(post.get("selftext", ""))]

    def walk(children: list, depth: int) -> None:
        for c in children:
            d = c.get("data", {})
            body_t = _clean_body(d.get("body", ""))
            if not body_t:
                continue
            parts.append(f"{'  ' * depth}- **u/{d.get('author', '?')}**: {body_t[:2000]}")
            replies = d.get("replies")
            if isinstance(replies, dict):
                walk(replies["data"]["children"], depth + 1)

    try:
        walk(listing[1]["data"]["children"], 0)
    except Exception:
        pass
    return _resp(url, "\n\n".join(parts), "reddit-json", title)


# ── GitHub API ────────────────────────────────────────────────────────
# Repo root/tree → metadata + README; blob/raw → file content. Keyless,
# 60 req/hr unauthenticated — fine for occasional fetches.

_GITHUB_RE = re.compile(r"^https?://(?:www\.)?github\.com/([\w.-]+)/([\w.-]+)(/.*)?$")


async def _github_api(url: str) -> dict | None:
    m = _GITHUB_RE.search(url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    rest = (m.group(3) or "").rstrip("/")

    fm = re.match(r"^/(?:blob|raw)/(.+?)/(.*)$", rest)
    if fm:  # blob/<branch>/<path> — branch may contain '/', first split wins
        branch, path = fm.group(1), fm.group(2)
        body = await _get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}")
        if body is None:
            return None
        return _resp(url, body, "github-raw", path.split("/")[-1])

    hdr = {"Accept": "application/vnd.github+json", "User-Agent": "websearch-mcp"}
    try:
        meta = await AsyncFetcher.get(f"https://api.github.com/repos/{owner}/{repo}",
                                      timeout=12, stealthy_headers=False, headers=hdr)
        readme = await AsyncFetcher.get(f"https://api.github.com/repos/{owner}/{repo}/readme",
                                        timeout=12, stealthy_headers=False,
                                        headers={**hdr, "Accept": "application/vnd.github.raw+json"})
    except Exception:
        return None
    if meta.status != 200:
        return None
    try:
        md = json.loads(meta.body if isinstance(meta.body, str)
                        else meta.body.decode("utf-8", errors="replace"))
    except Exception:
        return None
    head = [f"# {md.get('full_name', f'{owner}/{repo}')}",
            (md.get("description") or "").strip()]
    stars = md.get("stargazers_count")
    if stars is not None:
        head.append(f"⭐ {stars} | {md.get('language') or ''} | "
                    f"updated {md.get('updated_at', '')[:10]} | "
                    f"license {((md.get('license') or {}).get('spdx_id') or 'none')}")
    topics = md.get("topics")
    if topics:
        head.append("topics: " + ", ".join(topics))
    body = readme.body if readme.status == 200 else ""
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    if isinstance(body, str) and body.lower().startswith(("<!doctype", "<html")):
        body = ""
    content = "\n\n".join(head + [body]).strip() or f"No README for {owner}/{repo}."
    return _resp(url, content, "github-api", md.get("full_name", f"{owner}/{repo}"))


# ── dispatch ──────────────────────────────────────────────────────────

def _is_root_url(url: str) -> bool:
    return urlparse(url).path in ("", "/")


async def fast_path_fetch(url: str) -> dict | None:
    """Run the applicable fast paths; return a response dict or None."""
    if not url.startswith("http"):
        return None
    probes = []
    if _is_root_url(url):
        probes.append(_llms_txt)
    if _YT_RE.search(url):
        probes.append(_youtube_transcript)
    if _REDDIT_RE.search(url):
        probes.append(_reddit_json)
    if _GITHUB_RE.search(url):
        probes.append(_github_api)
    for fn in probes:
        try:
            r = await fn(url)
        except Exception:
            r = None
        if r:
            return r
    return None