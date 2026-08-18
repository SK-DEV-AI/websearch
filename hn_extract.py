"""Hacker News dedicated extractor (donsetch extract/hn.rs port).

HN's comment tree is a nested-table layout the generic pipeline
renders as pipe-table rows (truncating every comment) or loses to
main-content scoring. This extractor recovers the thread: full
comment text, authors, ages, reply depth — token-efficient. Returns
None for non-HN or non-item pages.
"""

import re
from urllib.parse import urljoin

from lxml import html as lh

MAX_COMMENTS = 150
INDENT = "  "


def _plain(el):
    return " ".join("".join(el.itertext()).split())


def _markdown(el, base):
    """Lightweight inline markdown: links → [t](href), code → backticks."""
    for a in el.xpath(".//a"):
        href = a.get("href") or ""
        t = " ".join("".join(a.itertext()).split())
        abs_href = urljoin(base, href) if href else ""
        a.tag = "span"
        a.attrib.clear()
        if t:
            a.text = "[%s](%s)" % (t, abs_href) if abs_href else t
        else:
            a.text = abs_href or ""
    for code in el.xpath(".//code | .//pre"):
        t = " ".join("".join(code.itertext()).split())
        code.tag = "span"
        code.attrib.clear()
        code.text = "`%s`" % t
    for p in el.xpath(".//p"):
        p.tag = "span"
        p.attrib.clear()
        p.tail = ("\n\n" + (p.tail or "")).strip() or "\n\n"
    for br in el.xpath(".//br"):
        br.tag = "span"
        br.attrib.clear()
        br.text = " "
    for elm in el.xpath(".//*"):
        if elm.tag not in ("span",):
            elm.tag = "span"
            elm.attrib.clear()
    return re.sub(r"\n{3,}", "\n\n", "".join(el.itertext())).strip()


def try_extract(html, url, focus=None):
    """Full thread/permalink markdown for HN item pages, else None."""
    try:
        from urllib.parse import urlparse
        p = urlparse(url)
    except Exception:
        return None
    if p.hostname != "news.ycombinator.com" or not p.path.startswith("/item"):
        return None

    doc = lh.fromstring(html)
    comments = [tr for tr in doc.xpath("//tr[contains(concat(' ', normalize-space(@class), ' '), ' comtr ')]")
                if "noshow" not in (tr.get("class") or "").split()]
    if comments:
        return _thread(doc, url, comments, focus)
    return _permalink(doc, url)


def _permalink(doc, url):
    fat = doc.xpath("//table[contains(concat(' ', normalize-space(@class), ' '), ' fatitem ')]")
    if not fat:
        return None
    fat = fat[0]
    commtext = fat.xpath(".//div[contains(concat(' ', normalize-space(@class), ' '), ' commtext ')]")
    if not commtext:
        return None
    text = _markdown(commtext[0], url)
    if not text:
        return None
    author = _plain(fat.xpath(".//a[contains(concat(' ', normalize-space(@class), ' '), ' hnuser ')]")[0]) \
        if fat.xpath(".//a[contains(concat(' ', normalize-space(@class), ' '), ' hnuser ')]") else ""
    age = _plain(fat.xpath(".//span[contains(concat(' ', normalize-space(@class), ' '), ' age ')]")[0]) \
        if fat.xpath(".//span[contains(concat(' ', normalize-space(@class), ' '), ' age ')]") else ""
    story = None
    on = fat.xpath(".//span[contains(concat(' ', normalize-space(@class), ' '), ' onstory ')]//a")
    if on:
        story = (_plain(on[0]), on[0].get("href"))
    else:
        tl = fat.xpath(".//td[contains(concat(' ', normalize-space(@class), ' '), ' title ')]//a")
        if tl:
            story = (_plain(tl[0]), tl[0].get("href"))

    md = ""
    title = "HN comment"
    if story and story[0]:
        title = "Comment on: %s" % story[0]
        md += "# %s\n" % title
    else:
        md += "# HN comment\n"
    byline = []
    if author:
        byline.append("**%s**" % author)
    if age:
        byline.append(age)
    if byline:
        md += " · ".join(byline) + "\n"
    if story and story[1]:
        md += "thread: %s\n" % urljoin(url, story[1])
    md += "%s\n\n%s\n" % (url, text)
    return md


def _thread(doc, url, comments, focus):
    # ── Story header ──
    title_el = doc.xpath("//tr[contains(concat(' ', normalize-space(@class), ' '), ' athing ')]"
                         "//a[contains(concat(' ', normalize-space(@class), ' '), ' title-link ')]")
    if not title_el:
        title_el = doc.xpath("//tr[contains(concat(' ', normalize-space(@class), ' '), ' athing ')]"
                             "//td[last()]//a")
    title = _plain(title_el[0]) if title_el else ""
    if not title:
        t = doc.xpath("//title")
        if t:
            title = "".join(t[0].itertext()).replace("| Hacker News", "").strip()
    if not title:
        return None

    link = url
    tl = doc.xpath("//tr[contains(concat(' ', normalize-space(@class), ' '), ' submission ')]"
                   "//a[contains(concat(' ', normalize-space(@class), ' '), ' title-link ')]")
    if tl and tl[0].get("href"):
        link = urljoin(url, tl[0].get("href"))

    subtext = doc.xpath("//td[contains(concat(' ', normalize-space(@class), ' '), ' subtext ')]")
    points = author = age = cc = ""
    if subtext:
        sc = subtext[0].xpath(".//span[contains(concat(' ', normalize-space(@class), ' '), ' score ')]")
        if sc:
            points = _plain(sc[0])
        au = subtext[0].xpath(".//a[contains(concat(' ', normalize-space(@class), ' '), ' hnuser ')]")
        if au:
            author = _plain(au[0])
        ag = subtext[0].xpath(".//span[contains(concat(' ', normalize-space(@class), ' '), ' age ')]")
        if ag:
            age = _plain(ag[0])
        for a in subtext[0].xpath(".//a"):
            t = _plain(a)
            if t.endswith("comments") or t.endswith("comment"):
                cc = t
                break

    md = "# %s\n" % title
    byline = [x for x in (points, "by %s" % author if author else "", age, cc) if x]
    if byline:
        md += " · ".join(byline) + "\n"
    md += "%s\n\n" % link

    # Self-post text (Ask HN etc.)
    op = doc.xpath("//div[contains(concat(' ', normalize-space(@class), ' '), ' toptext ')] | "
                   "//td[contains(concat(' ', normalize-space(@class), ' '), ' toptext ')]")
    if op:
        t = _markdown(op[0], url)
        if t:
            md += t + "\n\n"

    # ── Comments ──
    parsed = []
    for tr in comments[:MAX_COMMENTS]:
        img = tr.xpath(".//td[contains(concat(' ', normalize-space(@class), ' '), ' ind ')]//img")
        depth = 0
        if img:
            w = (img[0].get("width") or "")
            if w.isdigit():
                depth = int(w) // 40
        au = tr.xpath(".//a[contains(concat(' ', normalize-space(@class), ' '), ' hnuser ')]")
        author = _plain(au[0]) if au else "[deleted]"
        ag = tr.xpath(".//span[contains(concat(' ', normalize-space(@class), ' '), ' age ')]")
        age = _plain(ag[0]) if ag else ""
        cm = tr.xpath(".//div[contains(concat(' ', normalize-space(@class), ' '), ' commtext ')]")
        text = _markdown(cm[0], url) if cm else ""
        if not text:
            continue
        parsed.append({"depth": depth, "author": author, "age": age, "text": text})

    # Focus filter: keep comments matching any query term; no match
    # → full thread + notice (same contract as the generic pipeline).
    focus_missed = False
    if focus and focus.strip():
        terms = focus.lower().split()
        matched = [c for c in parsed if any(t in c["text"].lower() for t in terms)]
        if matched:
            md += "*(focus \"%s\": showing %d of %d comments)*\n\n" % (focus, len(matched), len(parsed))
            parsed = matched
        else:
            focus_missed = True

    if parsed:
        md += "## Discussion\n\n"
        for c in parsed:
            indent = INDENT * min(c["depth"], 12)
            header = "%s**%s**" % (indent, c["author"])
            if c["age"]:
                header += " · %s" % c["age"]
            md += header + "\n\n"
            for para in c["text"].split("\n"):
                para = para.strip()
                if para:
                    md += "%s%s\n\n" % (indent, para)

    if len(comments) > MAX_COMMENTS:
        md += "*(thread truncated: %d comments total, showing %d)*\n" % (len(comments), MAX_COMMENTS)
    if focus_missed:
        md += "*(focus \"%s\": no comment matched — showing full thread)*\n" % focus
    return md


if __name__ == "__main__":
    thread_html = """<html><body>
<table><tr class="athing submission"><td class="title"><a class="title-link" href="https://example.com/story">Test Story</a></td></tr></table>
<table><tr><td class="subtext"><span class="score">42 points</span> <a class="hnuser">alice</a> <span class="age">2 hours ago</span> <a>17 comments</a></td></tr></table>
<div class="toptext">Self-post body with <a href="https://example.com/l">a link</a>.</div>
<table>
<tr class="athing comtr"><td class="ind"><img width="0"></td><td><a class="hnuser">bob</a> <span class="age">1 hour ago</span><div class="commtext">First reply <a href="https://example.com/x">anchor</a></div></td></tr>
<tr class="athing comtr"><td class="ind"><img width="40"></td><td><a class="hnuser">carol</a> <span class="age">30 min ago</span><div class="commtext">Nested reply</div></td></tr>
<tr class="athing comtr noshow"><td class="ind"><img width="80"></td><td><div class="commtext">deleted branch</div></td></tr>
<tr class="athing comtr"><td class="ind"><img width="40"></td><td><a class="hnuser">dave</a> <span class="age">1 min ago</span><div class="commtext">second branch</div></td></tr>
</table></body></html>"""

    md = try_extract(thread_html, "https://news.ycombinator.com/item?id=999")
    assert md and "# Test Story" in md, md
    assert "https://example.com/story" in md, md
    assert "42 points" in md and "17 comments" in md, md
    assert "Self-post body with [a link](https://example.com/l)." in md, md
    assert "## Discussion" in md, md
    assert "**bob** · 1 hour ago" in md, md
    assert "[anchor](https://example.com/x)" in md, md
    assert "  **carol**" in md, md          # depth 1 = 2-space indent
    assert "**dave**" in md, md             # same depth as bob
    assert "deleted branch" not in md, md   # noshow dropped
    assert "second branch" in md, md

    # focus filter
    md = try_extract(thread_html, "https://news.ycombinator.com/item?id=999", focus="nested")
    assert "*(focus \"nested\": showing 1 of 3 comments)*" in md, md
    assert "**carol**" in md and "**bob**" not in md, md
    md = try_extract(thread_html, "https://news.ycombinator.com/item?id=999", focus="zzzz")
    assert "no comment matched" in md and "**bob**" in md, md

    # non-HN / non-item → None
    assert try_extract(thread_html, "https://example.com/item?id=1") is None, "host gate"
    assert try_extract(thread_html, "https://news.ycombinator.com/news") is None, "path gate"

    # permalink shape
    perm = """<html><body><table class="fatitem">
<tr><td><span class="onstory">on: <a href="item?id=5">A Story</a></span></td></tr>
<tr><td><a class="hnuser">zoe</a> <span class="age">3 hours ago</span>
<div class="commtext">A permalink comment <a href="https://e.com">with</a> links.</div></td></tr>
</table></body></html>"""
    md = try_extract(perm, "https://news.ycombinator.com/item?id=123")
    assert md and "# Comment on: A Story" in md, md
    assert "**zoe** · 3 hours ago" in md, md
    assert "thread: https://news.ycombinator.com/item?id=5" in md, md
    assert "[with](https://e.com)" in md and "links." in md, md
    print("HN-OK")
