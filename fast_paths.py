"""Fast-path fetches for well-known URL shapes (steal landscape 🥇).

Each path is shape-gated: returns a ready response dict (fetch_url-compatible)
or None to fall through to the generic pipeline. Zero cost when the URL does
not match — the gate is a regex on the URL, no requests issued.

Paths: llms.txt (site root), YouTube transcript (yt-dlp, watch/shorts/live/embed),
Reddit .json (comments threads), GitHub API (repo README / raw blob /
PR+issue threads with diff / tree listing / releases+tags / open lists),
StackOverflow/Stack Exchange (question + answers via API), Hacker News
(Firebase API threads), PyPI/npm/crates.io (registry metadata + README),
arXiv (abs API), HuggingFace (hub API), DOI (Crossref/unpaywall), GitHub gists.
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


async def _md_accept(url: str) -> dict | None:
    """Accept: text/markdown fast path (Parallel docs convention, now
    standard on docs sites): fetch the page itself asking for markdown.
    Sites that honor it (Mintlify, Docusaurus v3, Starlight) return clean
    LLM-ready markdown — no trafilatura mangling. Others ignore the header
    and return HTML, which we reject so the generic pipeline handles it."""
    try:
        await _validate_url(url)
    except SecurityError:
        return None
    try:
        resp = await AsyncFetcher.get(url, timeout=12, stealthy_headers=False,
                                      headers={"Accept": "text/markdown"})
    except Exception:
        return None
    if resp is None or resp.status != 200:
        return None
    ct = ""
    try:
        ct = (resp.headers.get("content-type", "") or "").lower()
    except Exception:
        pass
    if "markdown" not in ct and "text/plain" not in ct:
        return None  # server ignored the header — fall through
    body = resp.body
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    if not body or not body.strip() or len(body) > 2_000_000:
        return None
    return _resp(url, body.strip(), "md-accept",
                  f"markdown via Accept header for {urlparse(url).netloc}")


# ── YouTube transcript ────────────────────────────────────────────────

_YT_RE = re.compile(
    r"^https?://(?:www\.|m\.)?(?:youtube\.com/(?:watch\?.*?v=|shorts/|live/|embed/)|youtu\.be/)([\w-]{11})")


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
    except Exception:
        return None
    try:
        out, _ = await asyncio.wait_for(proc.communicate(), timeout=30)
    except asyncio.TimeoutError:
        # wait_for cancels communicate() but leaves yt-dlp running —
        # kill it or every slow video leaks a zombie process.
        try:
            proc.kill()
        except Exception:
            pass
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
    # Path-only rebuild: drops ?sort=/fragments (which would corrupt the
    # .json suffix) and normalizes old/new hosts to www (same-origin fetch).
    json_url = "https://www.reddit.com" + urlparse(url.rstrip("/")).path + ".json"
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


def _gh_thread_md(item: dict, comments: list[dict], kind_label: str) -> str:
    """Render a PR/issue + its comments as markdown."""
    num = item.get("number", "")
    title = (item.get("title") or "").strip()
    user = ((item.get("user") or {}).get("login") or "?")
    state = item.get("state", "")
    created = (item.get("created_at", "") or "")[:10]
    merged = item.get("merged_at")
    status = f"{state}" + (" (merged)" if merged else "")
    parts = [f"# {kind_label} #{num}: {title}",
             f"*{user}* — {status}, opened {created}",
             html_mod.unescape(item.get("body") or "").strip()]
    for c in comments:
        cu = ((c.get("user") or {}).get("login") or "?")
        cd = (c.get("created_at", "") or "")[:10]
        cb = html_mod.unescape(c.get("body") or "").strip()
        if not cb:
            continue
        parts.append(f"- **{cu}** ({cd}): {cb[:2000]}")
    return "\n\n".join(p for p in parts if p)


def _gh_files_md(files: list[dict]) -> str:
    """Render the PR diff file list (patches capped per file)."""
    lines = [f"## Files changed ({len(files)})"]
    for f in files[:30]:
        name = f.get("filename", "")
        status = f.get("status", "")
        lines.append(f"### `{name}` ({status}, "
                     f"+{f.get('additions', 0)}/-{f.get('deletions', 0)})")
        patch = f.get("patch") or ""
        if patch:
            lines.append("```diff\n" + patch[:3000] + "\n```")
    if len(files) > 30:
        lines.append(f"\u2026and {len(files) - 30} more files")
    return "\n\n".join(lines)


async def _github_thread(owner: str, repo: str, kind: str, num: str) -> dict | None:
    """Fetch a PR or issue thread via the keyless GitHub API.

    PRs also pull the file diff + code-review comments — a PR page
    without its diff is half the page.
    """
    hdr = {"Accept": "application/vnd.github+json", "User-Agent": "websearch-mcp"}
    is_pr = kind in ("pull", "pulls")
    base = f"https://api.github.com/repos/{owner}/{repo}"
    sub = "pulls" if is_pr else "issues"

    async def _api(path: str):
        try:
            r = await AsyncFetcher.get(f"{base}{path}", timeout=12,
                                       stealthy_headers=False, headers=hdr)
        except Exception:
            return None
        if r.status != 200:
            return None
        try:
            return json.loads(r.body if isinstance(r.body, str)
                              else r.body.decode("utf-8", errors="replace"))
        except Exception:
            return None

    item = await _api(f"/{sub}/{num}")
    if not isinstance(item, dict):
        return None
    # Conversation comments live on the issues endpoint for BOTH kinds —
    # /pulls/N/comments holds only inline code-review comments.
    conv = await _api(f"/issues/{num}/comments") or []
    files = rev = []
    if is_pr:
        files = await _api(f"/pulls/{num}/files?per_page=30") or []
        rev = await _api(f"/pulls/{num}/comments?per_page=20") or []

    label = "PR" if is_pr else "Issue"
    url_kind = "pull" if is_pr else "issues"
    sections = [_gh_thread_md(item, conv if isinstance(conv, list) else [], label)]
    if is_pr and isinstance(rev, list) and rev:
        rc = ["## Code review comments"]
        for c in rev[:20]:
            cu = ((c.get("user") or {}).get("login") or "?")
            cd = (c.get("created_at", "") or "")[:10]
            where = c.get("path", "")
            if c.get("line"):
                where += f":{c.get('line')}"
            cb = html_mod.unescape(c.get("body") or "").strip()
            if not cb:
                continue
            rc.append(f"- **{cu}** ({where}, {cd}): {cb[:1000]}")
        if len(rc) > 1:
            sections.append("\n\n".join(rc))
    if is_pr and isinstance(files, list) and files:
        sections.append(_gh_files_md(files))
    md = "\n\n".join(s for s in sections if s).strip()
    if not md:
        return None
    title = (item.get("title", "") or "").strip()
    return _resp(f"https://github.com/{owner}/{repo}/{url_kind}/{num}",
                  md, "github-pr", f"{label} #{num}: {title}")


async def _github_contents(owner: str, repo: str, path: str, branch: str) -> dict | None:
    """Directory listing (or single file) via the contents API."""
    hdr = {"Accept": "application/vnd.github+json", "User-Agent": "websearch-mcp"}
    ep = (f"https://api.github.com/repos/{owner}/{repo}/contents?ref={branch}"
          if not path else
          f"https://api.github.com/repos/{owner}/{repo}/contents/{path}?ref={branch}")
    try:
        r = await AsyncFetcher.get(ep, timeout=12,
                                   stealthy_headers=False, headers=hdr)
    except Exception:
        return None
    if r.status != 200:
        return None
    try:
        data = json.loads(r.body if isinstance(r.body, str)
                          else r.body.decode("utf-8", errors="replace"))
    except Exception:
        return None
    if isinstance(data, dict) and data.get("type") == "file":
        dl = data.get("download_url")
        body = await _get(dl) if dl else None
        if body is None:
            return None
        return _resp(f"https://github.com/{owner}/{repo}/blob/{branch}/{path}",
                      body, "github-raw", data.get("name", path.split("/")[-1]))
    if not isinstance(data, list):
        return None
    head = f"/{path}" if path else ""
    lines = [f"# {owner}/{repo} — {head or '/'} @ {branch}"]
    for e in data[:200]:
        name = e.get("name", "")
        if e.get("type") == "dir":
            lines.append(f"- \U0001F4C1 {name}/")
        else:
            lines.append(f"- `{name}` ({e.get('size', 0)} bytes)")
    if len(data) > 200:
        lines.append(f"\u2026and {len(data) - 200} more")
    url = f"https://github.com/{owner}/{repo}/tree/{branch}/{path}".rstrip("/")
    return _resp(url, "\n\n".join(lines), "github-tree",
                  f"{owner}/{repo} {head or '/'}")


async def _github_releases(owner: str, repo: str, rest: str) -> dict | None:
    """Release/tag pages via the API."""
    hdr = {"Accept": "application/vnd.github+json", "User-Agent": "websearch-mcp"}
    base = f"https://api.github.com/repos/{owner}/{repo}"
    if rest == "/tags":
        ep, mode = f"{base}/tags?per_page=30", "tags"
    elif rest in ("/releases", "/releases/latest"):
        ep = f"{base}/releases/latest" if rest.endswith("latest") else f"{base}/releases?per_page=10"
        mode = "releases"
    else:
        m = re.match(r"^/releases/tag/(.+)$", rest)
        if not m:
            return None
        ep, mode = f"{base}/releases/tags/{m.group(1)}", "release"
    try:
        r = await AsyncFetcher.get(ep, timeout=12,
                                   stealthy_headers=False, headers=hdr)
    except Exception:
        return None
    if r.status != 200:
        return None
    try:
        data = json.loads(r.body if isinstance(r.body, str)
                          else r.body.decode("utf-8", errors="replace"))
    except Exception:
        return None
    if mode == "tags":
        if not isinstance(data, list) or not data:
            return None
        lines = [f"# Tags \u2014 {owner}/{repo}"] + [f"- `{t.get('name', '')}`" for t in data[:30]]
        return _resp(f"https://github.com/{owner}/{repo}/tags",
                      "\n\n".join(lines), "github-tags", f"Tags \u2014 {owner}/{repo}")
    items = [data] if mode == "release" else (data if isinstance(data, list) else [])
    if not items:
        return None
    cap = 3000 if mode == "release" else 800
    parts = [f"# Releases \u2014 {owner}/{repo}"] if mode == "releases" else []
    for rel in items:
        name = (rel.get("name") or rel.get("tag_name") or "").strip()
        tag = rel.get("tag_name", "")
        date = (rel.get("published_at", "") or "")[:10]
        body = (rel.get("body") or "").strip()[:cap]
        parts.append(f"## {name} (`{tag}`, {date})\n\n{body}")
    content = "\n\n".join(p for p in parts if p).strip()
    if not content:
        return None
    return _resp(f"https://github.com/{owner}/{repo}{rest}",
                  content, "github-releases", f"Releases \u2014 {owner}/{repo}")


async def _github_issue_list(owner: str, repo: str, kind: str) -> dict | None:
    """Bare /pulls or /issues list page — open items, not the README."""
    hdr = {"Accept": "application/vnd.github+json", "User-Agent": "websearch-mcp"}
    ep = ("pulls?state=open&per_page=20" if kind == "pulls"
          else "issues?state=open&per_page=20")
    try:
        r = await AsyncFetcher.get(f"https://api.github.com/repos/{owner}/{repo}/{ep}",
                                   timeout=12, stealthy_headers=False, headers=hdr)
    except Exception:
        return None
    if r.status != 200:
        return None
    try:
        items = json.loads(r.body if isinstance(r.body, str)
                           else r.body.decode("utf-8", errors="replace"))
    except Exception:
        return None
    if not isinstance(items, list) or not items:
        return None
    label = "PRs" if kind == "pulls" else "Issues"
    lines = [f"# Open {label} \u2014 {owner}/{repo}"]
    for it in items:
        n = it.get("number", "")
        t = (it.get("title") or "").strip()
        u = ((it.get("user") or {}).get("login") or "?")
        lines.append(f"- #{n}: {t} (*{u}*, {it.get('comments', 0)} comments)")
    return _resp(f"https://github.com/{owner}/{repo}/{kind}",
                  "\n\n".join(lines), "github-list", f"Open {label} \u2014 {owner}/{repo}")


async def _github_api(url: str) -> dict | None:
    m = _GITHUB_RE.search(url)
    if not m:
        return None
    owner, repo = m.group(1), m.group(2)
    # Strip query/fragment: /pulls?q=... must route as /pulls, and
    # /pull/18#discussion_r... as /pull/18.
    rest = (m.group(3) or "").split("?")[0].split("#")[0].rstrip("/")

    fm = re.match(r"^/(?:blob|raw)/(.+?)/(.*)$", rest)
    if fm:  # blob/<branch>/<path> — branch may contain '/', first split wins
        branch, path = fm.group(1), fm.group(2)
        body = await _get(f"https://raw.githubusercontent.com/{owner}/{repo}/{branch}/{path}")
        if body is None:
            return None
        return _resp(url, body, "github-raw", path.split("/")[-1])

    # PR / issue pages: the old code ignored `rest` and returned the repo
    # README, so /pull/18 fetched the README instead of the PR. Serve the
    # actual thread via the keyless GitHub API (same 60 req/hr budget).
    pm = re.match(r"^/(pull|pulls|issues)/(\d+)(/.*)?$", rest)
    if pm:
        kind, num = pm.group(1), pm.group(2)
        pr = await _github_thread(owner, repo, kind, num)
        if pr is not None:
            return pr
        return None

    # REST has no discussions endpoint (GraphQL-only) — generic pipeline,
    # never the README (which would misrepresent the page).
    if rest.startswith("/discussions"):
        return None

    if rest in ("/pulls", "/issues"):
        lst = await _github_issue_list(owner, repo, rest.lstrip("/"))
        if lst is not None:
            return lst
        return None

    tm = re.match(r"^/tree/([^/]+)(?:/(.*))?$", rest)
    if tm:
        branch, path = tm.group(1), (tm.group(2) or "").rstrip("/")
        listing = await _github_contents(owner, repo, path, branch)
        if listing is not None:
            return listing
        return None

    if rest == "/tags" or rest == "/releases" or rest.startswith("/releases/"):
        rel = await _github_releases(owner, repo, rest)
        if rel is not None:
            return rel
        return None

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


# ── StackOverflow / Stack Exchange ────────────────────────────────────
# Q&A pages mangle in trafilatura the same way PRs did (question detached
# from answers, votes/accepted lost). Keyless Stack Exchange API v2.3
# (300 req/day unauthenticated — fine for occasional fetches).

_SE_RE = re.compile(
    r"^https?://((?:www\.)?stackoverflow\.com|(?:[\w-]+\.)?stackexchange\.com|"
    r"superuser\.com|serverfault\.com|askubuntu\.com|mathoverflow\.net)/questions/(\d+)")

_SE_SITES = {"stackoverflow.com": "stackoverflow", "superuser.com": "superuser",
             "serverfault.com": "serverfault", "askubuntu.com": "askubuntu",
             "mathoverflow.net": "mathoverflow"}


def _se_site(host: str) -> str:
    host = host.lower()
    if host in _SE_SITES:
        return _SE_SITES[host]
    # *.stackexchange.com -> the subdomain IS the site param
    # (apple, unix, gaming, ...); www.so redirects are already canonical.
    if host.endswith(".stackexchange.com"):
        return host.split(".")[0]
    return "stackoverflow"


def _se_strip(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html_mod.unescape(html or "")).strip()


async def _se_api(url: str) -> dict | None:
    m = _SE_RE.search(url)
    if not m:
        return None
    host, qid = m.group(1), m.group(2)
    site = _se_site(host)
    # Default filter (!nNPvSNPI7A) drops answer bodies; withbody keeps
    # bodies but drops score/accepted — so: question call + answers call.
    q_ep = (f"https://api.stackexchange.com/2.3/questions/{qid}"
            f"?order=desc&sort=votes&site={site}&filter=withbody")
    a_ep = (f"https://api.stackexchange.com/2.3/questions/{qid}/answers"
            f"?order=desc&sort=votes&site={site}&filter=withbody")
    try:
        q_r = await AsyncFetcher.get(q_ep, timeout=12, stealthy_headers=False)
        a_r = await AsyncFetcher.get(a_ep, timeout=12, stealthy_headers=False)
    except Exception:
        return None
    if q_r.status not in (200, 400):
        return None
    try:
        items = (json.loads(q_r.body if isinstance(q_r.body, str)
                            else q_r.body.decode("utf-8", errors="replace"))
                 .get("items", []))
    except Exception:
        return None
    if not items:
        return None
    q = items[0]
    accepted = q.get("accepted_answer_id")
    title = html_mod.unescape(q.get("title", "")).strip()
    parts = [f"# {title}",
             f"score {q.get('score', 0)} | {q.get('view_count', 0)} views | "
             f"tags: {', '.join(q.get('tags', []))}",
             _se_strip(q.get("body", ""))[:4000]]
    answers: list[dict] = []
    if a_r.status == 200:
        try:
            answers = (json.loads(a_r.body if isinstance(a_r.body, str)
                                  else a_r.body.decode("utf-8", errors="replace"))
                       .get("items", []))
        except Exception:
            answers = []
    for a in (answers or [])[:15]:
        owner = ((a.get("owner") or {}).get("display_name") or "?")
        flag = " ✓ ACCEPTED" if a.get("answer_id") == accepted else ""
        parts.append(f"## Answer by {owner} (score {a.get('score', 0)}){flag}\n\n"
                     + _se_strip(a.get("body", ""))[:3000])
    content = "\n\n".join(p for p in parts if p).strip()
    if not content:
        return None
    canon = f"https://{host}/questions/{qid}"
    return _resp(canon, content, "stackexchange", title)


# ── Hacker News ─────────────────────────────────────────────────────────
# Minimalist DOM threads mangle into pipe rows; Firebase API (no key,
# no rate limit per the official docs) serves the tree as JSON.

_HN_RE = re.compile(r"^https?://news\.ycombinator\.com/item\?id=(\d+)")


def _hn_time(ts: int) -> str:
    import datetime as _dt
    try:
        return _dt.datetime.fromtimestamp(ts, _dt.timezone.utc).strftime("%Y-%m-%d")
    except Exception:
        return ""


async def _hn_api(url: str) -> dict | None:
    m = _HN_RE.search(url)
    if not m:
        return None
    item_id = m.group(1)

    async def _item(iid) -> dict | None:
        try:
            r = await AsyncFetcher.get(
                f"https://hacker-news.firebaseio.com/v0/item/{iid}.json",
                timeout=12, stealthy_headers=False)
        except Exception:
            return None
        if r.status != 200:
            return None
        try:
            return json.loads(r.body if isinstance(r.body, str)
                              else r.body.decode("utf-8", errors="replace"))
        except Exception:
            return None

    top = await _item(item_id)
    if not top:
        return None
    title = top.get("title") or top.get("text", "")[:80] or f"HN {item_id}"
    parts = [f"# {title}",
             f"by {top.get('by', '?')} | score {top.get('score', 0)} | "
             f"{_hn_time(top.get('time', 0))}"]
    if top.get("url"):
        parts.append(top["url"])
    if top.get("text"):
        parts.append(_se_strip(top["text"])[:3000])

    # Breadth-first comment walk, capped (each comment = 1 request).
    seen = 0
    queue = [(kid, 0) for kid in (top.get("kids") or [])[:10]]
    while queue and seen < 30:
        kid, depth = queue.pop(0)
        c = await _item(kid)
        seen += 1
        if not c:
            continue
        text = _se_strip(c.get("text", ""))
        if text:
            parts.append(f"{'  ' * depth}- **{c.get('by', '?')}**: {text[:1500]}")
        if depth < 2:
            queue.extend((k, depth + 1) for k in (c.get("kids") or [])[:5])
    content = "\n\n".join(p for p in parts if p).strip()
    if not content:
        return None
    return _resp(f"https://news.ycombinator.com/item?id={item_id}",
                  content, "hackernews", title)


# ── Package registries (PyPI / npm / crates.io) ─────────────────────────
# Registry pages are metadata + README; scraping them loses structure.
# All three JSON APIs are keyless (codesearch already speaks them).

_PYPI_RE = re.compile(r"^https?://pypi\.org/project/([\w_.-]+)(?:/.*)?$")
_NPM_RE = re.compile(r"^https?://(?:www\.)?npmjs\.com/package/(@?[\w_.-]+(?:/[\w_.-]+)?)(?:/.*)?$")
_CRATES_RE = re.compile(r"^https?://crates\.io/crates/([\w_-]+)(?:/.*)?$")


async def _registry_api(url: str) -> dict | None:
    m = _PYPI_RE.search(url)
    kind = "pypi" if m else None
    name = m.group(1) if m else None
    if not m:
        m = _NPM_RE.search(url)
        if m:
            kind, name = "npm", m.group(1)
    if not m:
        m = _CRATES_RE.search(url)
        if m:
            kind, name = "crates", m.group(1)
    if not m or not kind or not name:
        return None
    try:
        if kind == "pypi":
            r = await AsyncFetcher.get(f"https://pypi.org/pypi/{name}/json",
                                       timeout=12, stealthy_headers=False)
            if r.status != 200:
                return None
            d = json.loads(r.body if isinstance(r.body, str)
                           else r.body.decode("utf-8", errors="replace"))
            info = d.get("info", {})
            lines = [f"# {info.get('name', name)} {info.get('version', '')}",
                     (info.get("summary") or "").strip(),
                     f"license {info.get('license') or 'unknown'} | "
                     f"requires-python {info.get('requires_python') or 'any'} | "
                     f"home {info.get('home_page') or ''}"]
            deps = info.get("requires_dist") or []
            if deps:
                lines.append("## Dependencies\n\n" + "\n".join(f"- `{x}`" for x in deps[:30]))
            desc = (info.get("description") or "").strip()[:4000]
            if desc:
                lines.append(desc)
            return _resp(f"https://pypi.org/project/{name}/", "\n\n".join(
                p for p in lines if p).strip(), "pypi-json", f"{name} {info.get('version', '')}")
        if kind == "npm":
            r = await AsyncFetcher.get(f"https://registry.npmjs.org/{name}/latest",
                                       timeout=12, stealthy_headers=False)
            if r.status != 200:
                return None
            d = json.loads(r.body if isinstance(r.body, str)
                           else r.body.decode("utf-8", errors="replace"))
            lines = [f"# {d.get('name', name)} {d.get('version', '')}",
                     (d.get("description") or "").strip(),
                     f"license {d.get('license') or 'unknown'}"]
            deps = d.get("dependencies") or {}
            if deps:
                lines.append("## Dependencies\n\n" + "\n".join(
                    f"- `{k}@{v}`" for k, v in list(deps.items())[:30]))
            readme = (d.get("readme") or "").strip()[:4000]
            if readme:
                lines.append(readme)
            return _resp(f"https://www.npmjs.com/package/{name}", "\n\n".join(
                p for p in lines if p).strip(), "npm-json", f"{name} {d.get('version', '')}")
        r = await AsyncFetcher.get(
            f"https://crates.io/api/v1/crates/{name}", timeout=12,
            stealthy_headers=False,
            headers={"Accept": "application/json", "User-Agent": "websearch-mcp"})
        if r.status != 200:
            return None
        d = json.loads(r.body if isinstance(r.body, str)
                       else r.body.decode("utf-8", errors="replace"))
        crate = d.get("crate", {})
        lines = [f"# {crate.get('name', name)} {crate.get('max_version', '')}",
                 (crate.get("description") or "").strip(),
                 f"downloads {crate.get('downloads', 0)} | "
                 f"updated {(crate.get('updated_at', '') or '')[:10]} | "
                 f"home {crate.get('homepage') or ''} | repo {crate.get('repository') or ''}"]
        return _resp(f"https://crates.io/crates/{name}", "\n\n".join(
            p for p in lines if p).strip(), "crates-json",
            f"{name} {crate.get('max_version', '')}")
    except Exception:
        return None
    return None


# ── arXiv ─────────────────────────────────────────────────────────────
# Abstract pages are the one shape trafilatura handles — but the API gives
# title/authors/abstract/PDF link as structure, not prose. Keyless.

_ARXIV_RE = re.compile(r"^https?://(?:www\.)?(?:arxiv\.org/(?:abs|html|pdf)|"
                       r"export\.arxiv\.org/api/query)[^\s]*?(\d{4}\.\d{4,5})(?:v\d+)?")


async def _arxiv_api(url: str) -> dict | None:
    m = _ARXIV_RE.search(url)
    if not m:
        return None
    arxiv_id = m.group(1)
    try:
        r = await AsyncFetcher.get(
            f"https://export.arxiv.org/api/query?id_list={arxiv_id}",
            timeout=12, stealthy_headers=False)
    except Exception:
        return None
    if r.status != 200:
        return None
    try:
        body = r.body if isinstance(r.body, str) else r.body.decode("utf-8", errors="replace")
        import xml.etree.ElementTree as _ET
        root = _ET.fromstring(body)
        ns = {"a": "http://www.w3.org/2005/Atom"}
        entry = root.find("a:entry", ns)
        if entry is None:
            return None
        title = (entry.findtext("a:title", "", ns) or "").strip()
        abstract = (entry.findtext("a:summary", "", ns) or "").strip()
        authors = [a.findtext("a:name", "", ns)
                   for a in entry.findall("a:author", ns)]
        cats = [c.get("term", "") for c in entry.findall("a:category", ns)]
        pdf = ""
        for link in entry.findall("a:link", ns):
            if link.get("title") == "pdf":
                pdf = link.get("href", "")
        pub = (entry.findtext("a:published", "", ns) or "")[:10]
        parts = [f"# {title}",
                 f"{' | '.join(a for a in authors if a)} | {pub} | {', '.join(cats)}",
                 abstract]
        if pdf:
            parts.append(f"PDF: {pdf}")
        content = "\n\n".join(p for p in parts if p).strip()
        if not content or not title:
            return None
        return _resp(f"https://arxiv.org/abs/{arxiv_id}", content,
                      "arxiv-api", title)
    except Exception:
        return None


# ── HuggingFace ─────────────────────────────────────────────────────────
# Model/dataset pages are React shells; the Hub API is keyless for public
# repos and returns README + metadata + stats as structure.

_HF_RE = re.compile(r"^https?://huggingface\.co/(?:datasets/|spaces/)?([\w.-]+/[\w.-]+)(?:/(?:blob|tree|resolve)/[^\s]*)?/?(?:\?.*)?$")


async def _hf_api(url: str) -> dict | None:
    m = _HF_RE.search(url)
    if not m:
        return None
    # Group 1 may carry the prefix (datasets/squad) since the regex
    # optionally consumes it — strip to the bare owner/repo slug.
    repo = m.group(1).removeprefix("datasets/").removeprefix("spaces/")
    if repo in ("models", "datasets", "spaces"):
        return None  # listing pages, not repos
    kind = "dataset" if "/datasets/" in url else "model"
    api = (f"https://huggingface.co/api/{'datasets' if kind == 'dataset' else 'models'}/{repo}")
    try:
        r = await AsyncFetcher.get(api, timeout=12, stealthy_headers=False)
    except Exception:
        return None
    if r.status != 200:
        return None
    try:
        d = json.loads(r.body if isinstance(r.body, str)
                       else r.body.decode("utf-8", errors="replace"))
        if not isinstance(d, dict) or "error" in d:
            return None
        repo = d.get("id", repo)  # canonical slug (renames: squad -> rajpurkar/squad)
        mid = repo
        likes = d.get("likes", 0) or 0
        dl = d.get("downloads", 0) or 0
        task = " | ".join(d.get("pipeline_tag", "") and [d["pipeline_tag"]] or [])
        tags = ", ".join((d.get("tags") or [])[:12])
        sibs = [s.get("rfilename", "") for s in (d.get("siblings") or [])]
        readme = ""
        try:
            rr = await AsyncFetcher.get(
                f"https://huggingface.co/{repo}/raw/main/README.md",
                timeout=12, stealthy_headers=False)
            if rr.status == 200:
                readme = (rr.body if isinstance(rr.body, str)
                          else rr.body.decode("utf-8", errors="replace"))[:4000]
        except Exception:
            pass
        parts = [f"# {mid}",
                 f"{task + ' | ' if task else ''}{likes} likes | {dl} downloads | tags: {tags}",
                 readme]
        if sibs:
            parts.append("## Files\n\n" + "\n".join(f"- `{s}`" for s in sibs[:40]))
        content = "\n\n".join(p for p in parts if p).strip()
        if not content:
            return None
        return _resp(f"https://huggingface.co/{repo}", content,
                      f"hf-{kind}", mid)
    except Exception:
        return None


# ── DOI ─────────────────────────────────────────────────────────────────
# doi.org links 302 to publisher pages (often paywalled shells). Crossref
# is keyless and returns title/authors/venue/date; Unpaywall adds the OA
# PDF link when one exists (no email = keyless, rate-limited but fine).

_DOI_RE = re.compile(r"^https?://(?:dx\.)?doi\.org/(10\.\S+?)/?$")


async def _doi_api(url: str) -> dict | None:
    m = _DOI_RE.search(url)
    if not m:
        return None
    doi = m.group(1).rstrip("/")
    try:
        r = await AsyncFetcher.get(
            f"https://api.crossref.org/works/{doi}", timeout=12,
            stealthy_headers=False,
            headers={"User-Agent": "websearch-mcp/1.0 (mailto:none)",
                     "Accept": "application/json"})
    except Exception:
        return None
    if r.status != 200:
        return None
    try:
        msg = (json.loads(r.body if isinstance(r.body, str)
                          else r.body.decode("utf-8", errors="replace"))
               .get("message", {}))
        title = " ".join(msg.get("title") or [])
        authors = ", ".join(
            f"{a.get('family', '')} {a.get('given', '')}".strip()
            for a in (msg.get("author") or [])[:8])
        venue = " ".join(msg.get("container-title") or [])
        year = ((msg.get("issued", {}).get("date-parts") or [[None]])[0][0])
        oa = ""
        try:
            ru = await AsyncFetcher.get(f"https://api.unpaywall.org/v2/{doi}?email=none",
                                        timeout=12, stealthy_headers=False)
            if ru.status == 200:
                um = (json.loads(ru.body if isinstance(ru.body, str)
                                 else ru.body.decode("utf-8", errors="replace")))
                best = um.get("best_oa_location") or {}
                oa = best.get("url_for_pdf") or best.get("url") or ""
        except Exception:
            pass
        parts = [f"# {title}" if title else f"# DOI {doi}",
                 f"{authors} | {venue} | {year}",
                 f"Open-access PDF: {oa}" if oa else ""]
        content = "\n\n".join(p for p in parts if p).strip()
        if not content:
            return None
        return _resp(f"https://doi.org/{doi}", content, "doi-crossref",
                      title or f"DOI {doi}")
    except Exception:
        return None


# ── GitHub gists ────────────────────────────────────────────────────────
# gist pages are React shells; the keyless Gist API returns files as text.

_GIST_RE = re.compile(r"^https?://gist\.github\.com/(?:[\w.-]+/)?([0-9a-f]{20,40})(?:[/?#].*)?$")


async def _gist_api(url: str) -> dict | None:
    m = _GIST_RE.search(url)
    if not m:
        return None
    gid = m.group(1)
    try:
        r = await AsyncFetcher.get(f"https://api.github.com/gists/{gid}",
                                   timeout=12, stealthy_headers=False)
    except Exception:
        return None
    if r.status != 200:
        return None
    try:
        d = json.loads(r.body if isinstance(r.body, str)
                       else r.body.decode("utf-8", errors="replace"))
        desc = (d.get("description") or "").strip()
        owner = ((d.get("owner") or {}).get("login") or "?")
        files = d.get("files") or {}
        parts = [f"# {desc or f'Gist {gid[:8]}'} (by {owner})", desc] \
            if desc else [f"# Gist {gid[:8]} (by {owner})"]
        for fname, f in list(files.items())[:10]:
            text = (f.get("content") or "")[:3000]
            parts.append(f"## {fname}\n\n```\n{text}\n```" if text else f"## {fname}")
        content = "\n\n".join(p for p in parts if p).strip()
        if not content:
            return None
        return _resp(f"https://gist.github.com/{gid}", content,
                      "github-gist", desc or f"Gist {gid[:8]}")
    except Exception:
        return None


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
    if _SE_RE.search(url):
        probes.append(_se_api)
    if _HN_RE.search(url):
        probes.append(_hn_api)
    if _PYPI_RE.search(url) or _NPM_RE.search(url) or _CRATES_RE.search(url):
        probes.append(_registry_api)
    if _ARXIV_RE.search(url):
        probes.append(_arxiv_api)
    if _HF_RE.search(url):
        probes.append(_hf_api)
    if _DOI_RE.search(url):
        probes.append(_doi_api)
    if _GIST_RE.search(url):
        probes.append(_gist_api)
    # Last resort before the generic pipeline: docs sites honoring
    # Accept: text/markdown return clean markdown; others fall through.
    probes.append(_md_accept)
    for fn in probes:
        try:
            r = await fn(url)
        except Exception:
            r = None
        if r:
            return r
    return None