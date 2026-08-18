"""Feed extractor: RSS 2.0 / Atom / JSON Feed → markdown (donsetch
extract/feed.rs port). A feed URL returned as a raw XML blob (25K
chars of CDATA soup) is worthless to an agent; this renders the feed
the way a feed reader shows it: channel header + items with
title/link/date/summary. Content-type lies constantly, so the sniff
also probes the payload.
"""

import json
import re
import xml.etree.ElementTree as ET

MAX_ITEMS = 60
SUMMARY_CAP = 400

_NS = "{http://purl.org/dc/elements/1.1/}date"


def _local(tag):
    return tag.rsplit("}", 1)[-1].lower()


def is_feed(content_type, body):
    """Sniff content-type + payload head for RSS/Atom/JSON Feed."""
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    ct = (content_type or "").lower()
    if any(k in ct for k in ("rss", "atom", "feed+json")):
        return True
    head = body[:1024].lower()
    ls = head.lstrip()
    if ls.startswith("<?xml") or ls.startswith("<rss") or ls.startswith("<feed"):
        scan = body[: 16 * 1024].lower()
        return "<rss" in scan or "<feed" in scan
    if (ct and "json" in ct or ls.startswith("{")) and "jsonfeed.org" in head:
        return True
    return False


def _clean_html(raw):
    """Strip tags, keep text (feed summaries often carry inline HTML)."""
    if not raw:
        return ""
    s = re.sub(r"<br\s*/?>|</p>", "\n", raw, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return " ".join(s.split())


def _text(el, name):
    for child in el:
        if _local(child.tag) == name:
            t = "".join(child.itertext()).strip()
            if t:
                return t
    return ""


def _attr(el, name, key="href"):
    for child in el:
        if _local(child.tag) == name:
            v = (child.get(key) or "").strip()
            if v:
                return v
    return ""


def _extract_xml(text, url):
    try:
        doc = ET.fromstring(text)
    except ET.ParseError:
        return None
    items = [c for c in doc.iter() if _local(c.tag) == "item" or _local(c.tag) == "entry"]
    if not items:
        return None

    # Channel header: RSS root=rss > channel; Atom root=feed.
    container = doc
    if _local(doc.tag) == "rss":
        container = next((c for c in doc if _local(c.tag) == "channel"), doc)
    channel_title = _text(container, "title") or "Feed"
    channel_desc = _text(container, "description") or _text(container, "subtitle")
    site_link = _attr(container, "link") or _text(container, "link")

    md = "# %s\n" % channel_title
    if channel_desc:
        md += "> %s\n" % channel_desc
    if site_link:
        md += "%s\n" % site_link
    md += "%s\n\n" % url

    shown = 0
    for item in items[:MAX_ITEMS]:
        title = _text(item, "title")
        link = _attr(item, "link") or _text(item, "link") or _text(item, "guid")
        date = _text(item, "pubdate") or _text(item, "published") or \
            _text(item, "updated") or _text(item, "date")
        summary = _text(item, "description") or _text(item, "summary") or \
            _text(item, "content:encoded") or _text(item, "content")
        summary = _clean_html(summary)[:SUMMARY_CAP]

        if title:
            if link:
                md += "## [%s](%s)\n" % (title, link)
            else:
                md += "## %s\n" % title
        elif link:
            md += "## %s\n" % link
        else:
            continue
        if date:
            md += "%s\n" % date
        if summary:
            md += "%s\n" % summary
        md += "\n"
        shown += 1

    if len(items) > MAX_ITEMS:
        md += "*(feed truncated: %d items total, showing %d)*\n" % (len(items), MAX_ITEMS)
    return md.strip() if shown else None


def _extract_json(text, url):
    try:
        v = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    items = v.get("items")
    if not isinstance(items, list) or not items:
        return None

    channel_title = v.get("title") or "Feed"
    home = v.get("home_page_url")

    md = "# %s\n" % channel_title
    if v.get("description"):
        md += "> %s\n" % v["description"]
    if home:
        md += "%s\n" % home
    md += "%s\n\n" % url

    shown = 0
    for item in items[:MAX_ITEMS]:
        title = item.get("title") or ""
        link = item.get("url") or item.get("external_url") or ""
        date = item.get("date_published") or item.get("date_modified") or ""
        summary = _clean_html(item.get("summary") or item.get("content_text") or
                              item.get("content_html") or "")[:SUMMARY_CAP]
        if title:
            if link:
                md += "## [%s](%s)\n" % (title, link)
            else:
                md += "## %s\n" % title
        elif link:
            md += "## %s\n" % link
        else:
            continue
        if date:
            md += "%s\n" % date
        if summary:
            md += "%s\n" % summary
        md += "\n"
        shown += 1

    if len(items) > MAX_ITEMS:
        md += "*(feed truncated: %d items total, showing %d)*\n" % (len(items), MAX_ITEMS)
    return md.strip() if shown else None


def try_extract(body, url, content_type=""):
    """Return markdown if body is a recognizable feed, else None."""
    if isinstance(body, bytes):
        body = body.decode("utf-8", errors="replace")
    if not is_feed(content_type, body):
        return None
    if body.lstrip().startswith("{"):
        return _extract_json(body, url)
    return _extract_xml(body, url)


if __name__ == "__main__":
    rss = """<?xml version="1.0"?><rss version="2.0"><channel>
<title>Example Feed</title><description>Test channel</description>
<link>https://example.com</link>
<item><title>First Item</title><link>https://example.com/a</link>
<pubDate>Mon, 18 Aug 2026 10:00:00 GMT</pubDate>
<description><![CDATA[<p>Hello <b>world</b></p>]]></description></item>
<item><title>No Link</title><guid>https://example.com/g</guid><description>plain</description></item>
<item><description>no title no link</description></item>
</channel></rss>"""
    assert is_feed("text/xml", rss.encode()), "ct sniff"
    assert is_feed("", rss.encode()), "payload sniff"
    md = try_extract(rss, "https://example.com/feed.xml", "text/xml")
    assert md and "# Example Feed" in md, md
    assert "## [First Item](https://example.com/a)" in md, md
    assert "Hello world" in md and "<b>" not in md, md
    assert "No Link" in md and "https://example.com/g" in md, md
    assert "no title no link" not in md, "skipped item leaked"

    atom = """<?xml version="1.0"?><feed xmlns="http://www.w3.org/2005/Atom">
<title>Atom Feed</title><subtitle>sub</subtitle>
<link href="https://example.com/"/>
<entry><title>Atom Item</title><link href="https://example.com/x"/>
<updated>2026-08-18T10:00:00Z</updated>
<summary type="html">sum &amp; more</summary></entry></feed>"""
    md = try_extract(atom, "https://example.com/atom", "application/atom+xml")
    assert md and "# Atom Feed" in md and "## [Atom Item](https://example.com/x)" in md, md
    assert "sum & more" in md, md

    jf = json.dumps({
        "version": "https://jsonfeed.org/version/1.1",
        "title": "JSON Feed",
        "home_page_url": "https://example.com/",
        "items": [{
            "title": "JF Item",
            "url": "https://example.com/j",
            "date_published": "2026-08-18T10:00:00Z",
            "summary": "<p>jf body</p>",
        }, {"id": "x", "content_text": "no title"}],
    })
    assert is_feed("application/json", jf), "jsonfeed sniff"
    md = try_extract(jf, "https://example.com/jf.json")
    assert md and "# JSON Feed" in md and "## [JF Item](https://example.com/j)" in md, md
    assert "jf body" in md and "no title" not in md, md

    # 60-item truncation note
    many = {"version": "https://jsonfeed.org/version/1.1", "title": "Many",
            "items": [{"id": str(i), "title": "t%d" % i} for i in range(65)]}
    md = try_extract(json.dumps(many), "https://example.com/m.json")
    assert "feed truncated: 65 items" in md and "t59" in md and "t64" not in md, md

    assert try_extract("<html><body>plain</body></html>", "https://e.com") is None, "html passthrough"
    assert try_extract("not a feed", "https://e.com", "text/plain") is None, "junk"
    print("FEED-OK")
