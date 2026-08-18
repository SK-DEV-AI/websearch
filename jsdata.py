"""SPA JSON-in-script content miner (jsdata rescue).

Faithful Python port of donsetch's extract/jsdata.rs. Modern SPAs
(Next.js, Nuxt, Remix, GitHub, YouTube) render client-side but embed
their full content as a JSON blob — a known JS global assignment, a
`<script type="application/json">`/ld+json tag, a GitHub
`data-target="react-*.embeddedData"` script, or Next.js 13+
`self.__next_f.push([k, "..."])` RSC flight frames. Tier-1 can't run
the JS but CAN parse the blob, turning an empty SPA shell into real
content without a browser.

Runs only as a rescue: when the normal pipeline produced a thin
shell (fetch.py gate), this mines the embedded data and, if richer,
wins. Returns None when no meaningful content is recovered.
"""

import html
import json
from html.parser import HTMLParser

KNOWN_GLOBALS = (
    "ytInitialData", "ytInitialPlayerResponse", "__NEXT_DATA__",
    "nextData", "__NUXT__", "window.__NUXT__", "__APOLLO_STATE__",
    "__INITIAL_STATE__", "__PRELOADED_STATE__", "__ASYNC_LOADING_STATE__",
    "__REMARKS_STATE__", "__STATIC_ERRORS__", "window.__INITIAL_DATA__",
    "preloadState", "initialState", "appSettings",
    "window.APPLICATION_STATE", "window.BOOTCAMP_DATA",
    "window.PAYPAL_CHOOSE_JS__", "window.__myCodeIgniterData",
    "window.__DIFFICULTY_DATA__",
)

GOOD_KEYS = (
    "title", "name", "headline", "children", "markup", "description",
    "shortdescription", "overview", "summary", "abstract", "readme",
    "body", "content", "text", "plaintext", "markdown", "articlebody",
    "articletext", "subtitle", "excerpt", "intro", "introduction",
    "conclusion", "details", "transcript", "lyrics", "caption", "quote",
    "review", "snippet", "catchline", "surtitle", "ebyline", "about",
    "bio", "profile_bio", "answer", "question", "explanation", "tagline",
    "message", "descriptiontext", "repositorydescription", "objectives",
    "requirements", "responsibilities", "qualifications",
)

BAD_KEYS = (
    "id", "guid", "url", "uri", "href", "src", "srcset", "permalink",
    "thumbnail", "avatar", "icon", "logo", "image", "imageurl",
    "background", "bgimage", "css", "javascript", "bundle", "chunk",
    "token", "secret", "apikey", "accesstoken", "csrf", "xsrf",
    "signature", "fingerprint", "endpoint", "api", "query", "mutation",
    "fragment", "relay", "typename", "__typename", "props", "style",
    "trackingparams", "clicktracking", "continuation", "ctoken",
    "clicktrackingparams", "attributes", "badges", "chips", "emoji",
    "emoticons", "timestamp", "timesec", "lengthseconds", "duration",
    "starttime", "endtime", "expiry", "date", "datetime", "uuid",
    "videoid", "playlistid", "channelid", "thumbnailoverlay",
    "tracingvector", "cornernedges", "accessibility", "arialabel",
    "tooltip", "button", "likes", "dislike", "subscribe", "notification",
    "settings", "mnemonic", "hotkeydialog", "topbar", "leftcontrols",
    "playeroverlay", "chipcloud", "teaser", "carousel", "shelf",
    "compact", "lockup", "reel", "short", "grid", "menu", "dropdown",
    "watchnext", "relatedvideos", "secondaryvideo", "suggestion",
    "recommendation",
)

TITLE_KEYS = ("title", "name", "headline", "namewithowner", "fullname")

MAX_DEPTH = 24
MAX_ARRAY_ITEMS = 400
KEEP_SCORE = 2.0
KEEP_MIN_CHARS = 25
MIN_TOTAL_CHARS = 400
RENDER_ITEM_CAP = 240
RENDER_CHAR_CAP = 60_000


def extract(html_text: str, url: str) -> str | None:
    """Mine embedded JSON for content. Returns markdown or None."""
    blobs = _find_blobs(html_text)
    if not blobs:
        return None

    items: list[dict] = []
    order = 0
    for raw in blobs:
        text = html.unescape(raw)
        try:
            value = json.loads(text)
        except (json.JSONDecodeError, ValueError):
            continue
        _walk(value, "", items, order, 0)
        order = len(items)
    if not items:
        return None

    kept = [i for i in items if i["score"] >= KEEP_SCORE
            and len(i["text"].strip()) >= KEEP_MIN_CHARS]
    if not kept:
        return None

    _dedupe(kept)
    if sum(len(i["text"]) for i in kept) < MIN_TOTAL_CHARS:
        return None

    md, _title = _render(kept, url)
    return md


# ── blob discovery ───────────────────────────────────────────────

def _find_blobs(html_text: str) -> list[str]:
    out: list[str] = []
    add = lambda raw: out.append(raw) if raw not in out else None  # noqa: E731

    for key in KNOWN_GLOBALS:
        search = f"{key} = "
        from_ = 0
        while True:
            idx = html_text.find(search, from_)
            if idx < 0:
                break
            start = idx + len(search)
            raw = _extract_js_value(html_text[start:])
            if raw is not None:
                add(raw)
                from_ = start + len(raw)
            else:
                from_ = start + 1

    for ty in ("application/json", "application/ld+json"):
        for body in _find_typed_bodies(html_text, ty):
            add(body)

    for body in _find_data_target(html_text, "embeddedData"):
        add(body)

    for frame in _find_next_f(html_text):
        js = _strip_frame_key(frame)
        if js and js[0] in "[{":
            add(js)

    return out


def _extract_js_value(s: str) -> str | None:
    """Balanced JS object/array starting at s (strings + escapes)."""
    start = None
    for i, b in enumerate(s):
        if b in "{[":
            start = i
            break
    if start is None:
        return None
    open_c = s[start]
    close_c = "}" if open_c == "{" else "]"
    depth = 0
    in_str = False
    esc = False
    for i in range(start, len(s)):
        b = s[i]
        if in_str:
            if esc:
                esc = False
            elif b == "\\":
                esc = True
            elif b == '"':
                in_str = False
            continue
        if b == '"':
            in_str = True
        elif b in "{[":
            depth += 1
        elif b in "}]":
            depth -= 1
            if depth == 0 and b == close_c:
                return s[start:i + 1]
    return None


def _find_typed_bodies(html_text: str, ty: str) -> list[str]:
    out = []
    needle = f'type="{ty}"'
    from_ = 0
    while True:
        lt = html_text.find("<script", from_)
        if lt < 0:
            break
        tag_end = html_text.find(">", lt)
        if tag_end < 0:
            break
        if needle in html_text[lt:tag_end]:
            rest = html_text[tag_end + 1:]
            cs = rest.find("</script>")
            if cs >= 0:
                raw = rest[:cs].strip()
                if raw:
                    out.append(raw)
        from_ = tag_end + 1
    return out


def _find_data_target(html_text: str, suffix: str) -> list[str]:
    out = []
    from_ = 0
    while True:
        lt = html_text.find("<script", from_)
        if lt < 0:
            break
        tag_end = html_text.find(">", lt)
        if tag_end < 0:
            break
        tag = html_text[lt:tag_end]
        if 'data-target="' in tag and suffix in tag:
            rest = html_text[tag_end + 1:]
            cs = rest.find("</script>")
            if cs >= 0:
                raw = rest[:cs].strip()
                if raw:
                    out.append(raw)
        from_ = tag_end + 1
    return out


def _find_next_f(html_text: str) -> list[str]:
    out = []
    needle = "self.__next_f.push(["
    from_ = 0
    while True:
        rel = html_text.find(needle, from_)
        if rel < 0:
            break
        body_start = rel + len(needle)
        rest = html_text[body_start:]
        q = rest.find('"')
        if q < 0:
            break
        i = q + 1
        esc = False
        end = len(rest)
        while i < len(rest):
            b = rest[i]
            if esc:
                esc = False
            elif b == "\\":
                esc = True
            elif b == '"':
                end = i
                break
            i += 1
        inner = rest[q + 1:end]
        decoded = _js_unescape(inner)
        if len(decoded) > 8:
            out.append(decoded)
        from_ = body_start + end + 1
    return out


def _js_unescape(s: str) -> str:
    """Decode a JS string literal: \\uXXXX, \\xXX, simple escapes."""
    out = []
    i = 0
    n = len(s)
    while i < n:
        if s[i] != "\\":
            out.append(s[i])
            i += 1
            continue
        j = i + 1
        if j >= n:
            break
        c = s[j]
        if c == "u" and j + 4 < n + 1 and j + 5 <= n:
            h = s[j + 1:j + 5]
            try:
                out.append(chr(int(h, 16)))
                i = j + 5
                continue
            except ValueError:
                pass
        if c == "x" and j + 3 <= n:
            h = s[j + 1:j + 3]
            try:
                out.append(chr(int(h, 16)))
                i = j + 3
                continue
            except ValueError:
                pass
        simple = {"n": "\n", "r": "\r", "t": "\t", "b": "\b",
                  "f": "\f", "/": "/"}
        if c in simple:
            out.append(simple[c])
            i = j + 1
        else:
            out.append(c)
            i = j + 1
    return "".join(out)


def _strip_frame_key(frame: str) -> str:
    t = frame.strip()
    idx = t.find(":")
    if idx < 0:
        return t
    rest = t[idx + 1:].strip()
    if rest[:1] in "[{":
        return rest
    return t


# ── JSON walk ────────────────────────────────────────────────────

def _walk(value, path: str, out: list[dict], order: int, depth: int) -> int:
    """Collect content-bearing strings. Returns the next order index."""
    if depth > MAX_DEPTH:
        return order
    if isinstance(value, dict):
        for k, v in value.items():
            np = f"{path}.{k}" if path else str(k)
            order = _walk(v, np, out, order, depth + 1)
    elif isinstance(value, list):
        for v in value[:MAX_ARRAY_ITEMS]:
            order = _walk(v, path, out, order, depth + 1)
    elif isinstance(value, str):
        score, title_like = _score_string(value, path)
        if score >= 1.5:
            out.append({"text": value, "score": score, "order": order,
                        "title_like": title_like})
            order += 1
    return order


def _has_any(s: str, chars: str) -> bool:
    return any(c in s for c in chars)


def _score_string(s: str, path: str) -> tuple[float, bool]:
    t = s.strip()
    if not t:
        return 0.0, False
    if _has_any(t, "{}"):
        return 0.0, False
    for frag in ("@media", ":root", "@font-face", "url(", "px;", "rgb(", "var(--"):
        if frag in t:
            return 0.0, False
    low = t.lower()
    if ("noopener" in low or "noreferrer" in low or "nofollow" in low) and " " not in t:
        return 0.0, False
    if _looks_like_attr(t):
        return 0.0, False
    if "://" in t or t.startswith("data:"):
        return 0.0, False
    if " " not in t and len(t) >= 40:
        return 0.0, False
    letters = sum(1 for c in t if c.isascii() and c.isalpha())
    total = len(t)
    if letters == 0 or letters / max(total, 1) < 0.25:
        return 0.0, False

    lower = path.lower()
    score = 0.0
    title_like = False
    key_hit = False
    for key in GOOD_KEYS:
        if key in lower:
            score += 2.0
            key_hit = True
            if key in TITLE_KEYS:
                title_like = True
            break
    for key in BAD_KEYS:
        if key in lower:
            score -= 3.0
            break

    if " " in t:
        score += 1.0
    if _has_any(t, ".,!?:"):
        score += 0.5
    if len(t) >= 120:
        score += 1.0
    elif len(t) >= 60:
        score += 0.5
    if "microformat" in lower or "primaryinfo" in lower:
        score += 0.5

    if not key_hit:
        if not (len(t) >= 60 and " " in t
                and ("." in t or "?" in t)) or len(t.split()) < 6:
            return 0.0, False

    return max(score, 0.0), title_like


def _looks_like_attr(t: str) -> bool:
    first = t.split(None, 1)[0] if t else ""
    eq = first.find("=")
    if eq < 2:
        return False
    if len(first[eq + 1:]) < 2:
        return False
    return all(c.isalnum() or c == "-" for c in first[:eq])


def _dedupe(items: list[dict]) -> None:
    seen: set[str] = set()
    kept = []
    for i in items:
        norm = "".join(c for c in i["text"] if not c.isspace()).lower()
        if norm and norm not in seen:
            seen.add(norm)
            kept.append(i)
    kept.sort(key=lambda i: i["order"])
    items[:] = kept


# ── render ───────────────────────────────────────────────────────

def _render(items: list[dict], url: str) -> tuple[str, str | None]:
    md: list[str] = []
    title: str | None = None

    for it in items:
        if it["title_like"] and len(it["text"]) <= 200:
            clean = _strip_html(it["text"])
            if clean:
                title = clean
                md.append(f"# {clean}\n")
                break
    if title is None:
        from urllib.parse import urlsplit
        last = urlsplit(url).path.rstrip("/").split("/")[-1]
        if last:
            t = last.replace("-", " ").replace("+", " ").replace("_", " ")
            title = t
            md.append(f"# {t}\n")

    emitted = 0
    chars = 0
    for it in items:
        if it["title_like"] and title is not None:
            continue
        clean = _strip_html(it["text"]).strip()
        if len(clean) < 25:
            continue
        md.append(clean + "\n")
        emitted += 1
        chars += len(clean)
        if emitted >= RENDER_ITEM_CAP or chars > RENDER_CHAR_CAP:
            break

    return "\n".join(md), title


class _TextParser(HTMLParser):
    """Text of the fragment, script/style content suppressed."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []
        self._skip = 0

    def handle_starttag(self, tag, attrs):
        if tag in ("script", "style"):
            self._skip += 1

    def handle_endtag(self, tag):
        if tag in ("script", "style") and self._skip:
            self._skip -= 1

    def handle_data(self, data):
        if not self._skip:
            self.parts.append(data)


def _strip_html(s: str) -> str:
    if "<" not in s or ">" not in s:
        return s
    p = _TextParser()
    p.feed(s)
    p.close()
    return " ".join(" ".join(p.parts).split())


# ── selfcheck ────────────────────────────────────────────────────

if __name__ == "__main__":
    assert _js_unescape(r"a\u003cb\u003e") == "a<b>"
    assert _js_unescape(r'say \"hi\"') == 'say "hi"'
    assert _js_unescape(r"line1\nline2") == "line1\nline2"
    assert _js_unescape(r"\\ path") == "\\ path"
    assert _js_unescape(r"\u00e9tude") == "étude"
    assert _js_unescape("plain") == "plain"

    frames_html = ('<script>self.__next_f.push([1,"52:[\\"$\\",\\"p\\",null,'
                   '{\\"children\\":\\"Hello from the server\\"}]"]);'
                   'self.__next_f.push([2,"61:[\\"$\\",\\"h2\\",null,'
                   '{\\"children\\":\\"A second paragraph of text\\"}]"]);</script>')
    frames = _find_next_f(frames_html)
    assert len(frames) == 2, frames
    assert "Hello from the server" in frames[0]
    assert "A second paragraph" in frames[1]

    assert _strip_frame_key('["$","p"]') == '["$","p"]'
    assert _strip_frame_key('12:{"a":1}') == '{"a":1}'

    def para(n):
        return ("This is body paragraph number {} of the article -- containing "
                "enough substantial prose across several sentences that it "
                "reads as genuine written content rather than a short label "
                "or fragment.").format(n)
    frames_html = '<div id="root"></div><script>'
    for i in range(6):
        frame = f'{i + 50}:["$","p",null,{{"children":"{para(i)}"}}]'
        frames_html += f'self.__next_f.push([{i + 1},{json.dumps(frame)}]);'
    frames_html += "</script>"
    frames_html += "</script>"
    md = extract(frames_html, "https://example.com/post/1")
    assert md and len(md) >= 400, md
    assert "This is body paragraph number 0" in md
    assert "This is body paragraph number 5" in md

    plain = "<html><body><p>Just a normal server-side page, no embedded JSON anywhere in this document.</p></body></html>"
    assert extract(plain, "https://example.com/x") is None

    noise = '<script type="application/json">{"sessionId":"abc123","tracking":{"url":"/api/t","nonce":"deadbeefdecafbad"}}</script>'
    assert extract(noise, "https://example.com/a") is None

    assert _score_string("@media (prefers-color-scheme:dark){body{color:#000}}", ".footer.style") == (0.0, False)
    assert _score_string("width=device-width,initial-scale=1.0", ".viewport") == (0.0, False)
    sc, ty = _score_string("Making Navigations Instant in v0", ".flattened.title")
    assert sc > 2.0 and ty

    items = [
        {"text": "short description", "score": 4.0, "order": 0, "title_like": True},
        {"text": "a genuinely different long body of text that should survive", "score": 5.0, "order": 1, "title_like": False},
        {"text": "short description", "score": 4.0, "order": 2, "title_like": True},
    ]
    _dedupe(items)
    assert len(items) == 2, items

    assert _strip_html("<p>Hello <b>bold</b> world</p><script>var x=1;</script>") == "Hello bold world"

    print("JSDATA-OK")
