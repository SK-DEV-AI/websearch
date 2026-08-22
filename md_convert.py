"""Markdown conversion via html-to-markdown (steal C1 — Goldziher).

Replaces trafilatura's built-in markdown serializer on the CDP/scrapling/
wayback paths and the primary path's two-stage fallback. trafilatura still
owns *extraction* (main-content selection); this module owns the
HTML→Markdown serialization step, where trafilatura mangles tables and
code fences.

The html-to-markdown engine is Rust-backed (abi3 wheel) with a stable
ATX/GFM output shape; options here mirror the trafilatura flags that the
callers already carry (include_links/include_images/include_tables).
"""
from __future__ import annotations

import re

from html_to_markdown import ConversionOptions, convert

_LINK_RE = re.compile(r"<a\b[^>]*>(.*?)</a>", re.S | re.I)
_TABLE_RE = re.compile(r"<table\b.*?</table>", re.S | re.I)

# Boilerplate dropped on raw-DOM conversions (CDP/scrapling paths).
# trafilatura-extracted HTML rarely contains these, so applying the list
# unconditionally is harmless and keeps one code path.
_NOISE_SELECTORS = [
    "script", "style", "noscript", "nav", "footer", "aside", "form",
    "iframe", "[aria-hidden=true]", "[hidden]",
    ".advert", ".ads", ".advertisement", ".sponsor",
]


def to_markdown(
    html: str,
    *,
    include_links: bool = True,
    include_images: bool = True,
    include_tables: bool = True,
) -> str:
    """Convert HTML to GFM markdown. Returns "" on empty/invalid input."""
    if not html or not html.strip():
        return ""
    if not include_tables:
        html = _TABLE_RE.sub("", html)
    if not include_links:
        html = _LINK_RE.sub(r"\1", html)
    opts = ConversionOptions(
        extract_metadata=False,
        autolinks=include_links,
        skip_images=not include_images,
        exclude_selectors=_NOISE_SELECTORS,
    )
    try:
        return convert(html, opts).content.strip()
    except Exception:
        return ""


def _harvest_links(html: str, url: str = "", cap: int = 300) -> list[tuple[str, str]]:
    """Absolute-resolved (text, href) pairs from source HTML, deduped by URL."""
    try:
        from lxml import html as LH
        from urllib.parse import urljoin
        doc = LH.fromstring(html)
        out: list[tuple[str, str]] = []
        seen: set[str] = set()
        for a in doc.iter("a"):
            href = (a.get("href") or "").strip()
            if not href or href.startswith(("javascript:", "#", "mailto:")):
                continue
            text = " ".join(a.text_content().split())
            if not text:
                continue
            absu = urljoin(url, href) if url else href
            if absu in seen:
                continue
            seen.add(absu)
            out.append((text[:200], absu))
            if len(out) >= cap:
                break
        return out
    except Exception:
        return []


def _ensure_links(md: str, html: str, url: str) -> str:
    """Guarantee link coverage: trafilatura's link support is experimental
    and dies on forum layouts (refs stripped during wild-text recovery,
    verified 2.1.0 on phpBB — 455 anchors → 0 in output). When extraction
    keeps less than half the page's links, append the full inventory."""
    have = len(re.findall(r"\]\(", md))
    links = _harvest_links(html, url)
    if not links or have * 2 >= len(links):
        return md
    extra = "\n\n## Page links\n" + "\n".join(
        f"- [{t}]({h})" for t, h in links[have:])
    return md + extra


def extract_and_convert(html: str, url: str = "", *, fast: bool = False,
                        include_links: bool = True, include_images: bool = True,
                        include_tables: bool = True, deduplicate: bool = True,
                        **traf_kw) -> str:
    """trafilatura-extract main content as HTML, then convert to markdown.

    Fallback chain: extracted-HTML→markdown, else plain extraction to txt,
    else "". with_metadata is forced off for html output — trafilatura's
    build_html_output crashes on list-valued metadata (verified 2.1.0).
    """
    import trafilatura

    kw: dict = dict(traf_kw)
    kw.update(output_format="html", with_metadata=False,
              include_links=include_links, include_tables=include_tables,
              include_images=include_images, deduplicate=deduplicate)
    if fast:
        kw["fast"] = True
    if url:
        kw["url"] = url
    def _run(kws):
        try:
            return trafilatura.extract(html, **kws)
        except Exception:
            return None

    html_ext = _run(kw)
    # fast=True skips trafilatura's readability/jusText fallback chain;
    # on forum/thread layouts the primary pass keeps a tiny fragment
    # (phpBB 137KB → 1.4K chars vs 27K with fallbacks). Escalate to the
    # full chain when the result looks starved relative to page size.
    if not html_ext or len(html_ext) < max(1000, len(html) // 60):
        full = dict(kw)
        full.pop("fast", None)
        alt = _run(full)
        if alt and len(alt) > len(html_ext or ""):
            html_ext = alt
    if html_ext:
        md = to_markdown(html_ext, include_links=include_links,
                         include_images=include_images,
                         include_tables=include_tables)
        if md:
            if include_links:
                md = _ensure_links(md, html, url)
            return md
    # extraction returned nothing usable — fall back to raw text
    try:
        return (trafilatura.extract(html, output_format="txt",
                                    with_metadata=False, url=url or None) or "").strip()
    except Exception:
        return ""
