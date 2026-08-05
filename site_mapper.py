from __future__ import annotations

import gzip
import html as html_mod
import logging
import re
import xml.parsers.expat
import urllib.parse
from typing import Any

import httpx

from config import get_http_client

log = logging.getLogger("site_mapper")

SITEMAP_NS = "http://www.sitemaps.org/schemas/sitemap/0.9"
ATOM_NS = "http://www.w3.org/2005/Atom"

MAX_SITEMAP_BYTES = 50 * 1024 * 1024
MAX_RECURSION = 5

COMMON_SITEMAP_PATHS = [
    "/sitemap.xml", "/sitemap.xml.gz",
    "/sitemap_index.xml", "/sitemap-index.xml",
    "/sitemap_index.xml.gz", "/sitemap-index.xml.gz",
    "/.sitemap.xml",
    "/sitemap", "/sitemap/",
    "/sitemap/sitemap-index.xml", "/sitemap/sitemap.xml",
    "/sitemaps/sitemap.xml", "/sitemaps/sitemap.xml.gz",
    "/sitemap_news.xml", "/sitemap-news.xml",
    "/sitemap_news.xml.gz", "/sitemap-news.xml.gz",
    "/admin/config/search/xmlsitemap",
]

CONTENT_TYPES_UNLIKELY_XML = {
    "text/html", "application/json", "text/plain",
    "image/", "video/", "audio/", "font/",
}


async def map_site(
    url: str,
    max_depth: int = 0,
    max_urls: int = 1000,
    timeout: int = 30,
    include_sitemap: bool = True,
    include_links: bool = True,
    same_domain: bool = True,
    exclude_patterns: list[str] | None = None,
) -> dict[str, Any]:
    parsed = urllib.parse.urlparse(url)
    base = f"{parsed.scheme}://{parsed.netloc}"
    exclude = None
    if exclude_patterns:
        try:
            exclude = re.compile("|".join(exclude_patterns))
        except re.error:
            pass
    seen: set[str] = set()
    results: list[dict[str, Any]] = []
    source = "none"
    c = get_http_client()

    if include_sitemap:
        sitemap_urls: list[str] = []
        try:
            r = await c.get(f"{base}/robots.txt", timeout=10, follow_redirects=True)
            if r.status_code == 200:
                for line in r.text.splitlines():
                    line = line.strip()
                    if line.lower().startswith("sitemap:"):
                        sitemap_urls.append(line.split(":", 1)[1].strip())
        except Exception:
            pass

        for path in COMMON_SITEMAP_PATHS:
            try:
                r = await c.get(f"{base}{path}", timeout=8, follow_redirects=True)
                ctype = (r.headers.get("content-type", "") or "").lower()
                if r.status_code != 200:
                    continue
                if any(u in ctype for u in CONTENT_TYPES_UNLIKELY_XML):
                    continue
                sitemap_urls.append(str(r.url))
                if len(sitemap_urls) >= 5:
                    break
            except Exception:
                pass

        for su in sitemap_urls:
            await _parse_sitemap_recursive(
                su, c, seen, results, exclude,
                base if same_domain else "", max_urls, set(), 0, max_depth if max_depth > 0 else MAX_RECURSION,
            )
        if results:
            source = "sitemap"

    if include_links and len(results) < max_urls:
        html_text = await _fetch_page_html(url, c)
        if html_text:
            links: set[str] = set()
            for m in re.finditer(r'(?:href|src)=["\'](.*?)["\']', html_text, re.IGNORECASE):
                href = html_mod.unescape(m.group(1)).strip()
                if not href or href.startswith("#") or href.startswith("javascript:") or href.startswith("data:"):
                    continue
                absolute = urllib.parse.urljoin(url, href)
                parsed_link = urllib.parse.urlparse(absolute)
                if not parsed_link.scheme.startswith("http"):
                    continue
                normalised = urllib.parse.urlunparse((
                    parsed_link.scheme, parsed_link.netloc,
                    parsed_link.path.rstrip("/") or "/", "", "", "",
                ))
                if exclude and exclude.search(normalised):
                    continue
                if normalised not in seen:
                    seen.add(normalised)
                    links.add(normalised)
            for link in sorted(links):
                if link.startswith(base):
                    results.append({"url": link, "source": "link"})
                    if len(results) >= max_urls:
                        break
            if results and source == "none":
                source = "links"

    return {
        "success": True,
        "url": url,
        "base_domain": base,
        "source": source,
        "total": len(results),
        "urls": results[:max_urls],
    }


async def _parse_sitemap_recursive(
    sitemap_url: str,
    c: httpx.AsyncClient,
    seen: set[str],
    results: list[dict[str, Any]],
    exclude: re.Pattern | None,
    domain_filter: str,
    max_urls: int,
    parent_urls: set[str],
    depth: int,
    max_depth: int = MAX_RECURSION,
) -> None:
    if depth > max_depth:
        return
    if sitemap_url in parent_urls:
        return

    content = await _fetch_sitemap_content(sitemap_url, c)
    if content is None:
        return

    new_parents = parent_urls | {sitemap_url}

    if content.startswith("<"):
        # XML — could be sitemap index, urlset, RSS, or Atom
        try:
            import xml.etree.ElementTree as ET
            root = ET.fromstring(content[:10000])
            root_tag = _strip_ns(root.tag)
        except Exception:
            root_tag = ""

        parsed = False
        if root_tag in ("sitemapindex", "urlset"):
            parsed = await _parse_xml_sitemap(
                content, sitemap_url, seen, results,
                exclude, domain_filter, max_urls, c, new_parents, depth + 1, max_depth,
            )
        elif root_tag == "rss":
            parsed = await _parse_rss_sitemap(
                content, seen, results, exclude, domain_filter, max_urls,
            )
        elif root_tag == "feed":
            parsed = await _parse_atom_sitemap(
                content, seen, results, exclude, domain_filter, max_urls,
            )
        if not parsed:
            # malformed XML — try Expat fallback
            _parse_xml_fallback(content, seen, results, domain_filter, max_urls, exclude)
    else:
        # plain text sitemap (one URL per line)
        _parse_text_sitemap(content, seen, results, domain_filter, max_urls, exclude)


async def _fetch_sitemap_content(url: str, c: httpx.AsyncClient) -> str | None:
    try:
        r = await c.get(url, timeout=15, follow_redirects=True)
        if r.status_code != 200:
            return None
        raw = r.content
        if not raw:
            return None
        if len(raw) > MAX_SITEMAP_BYTES:
            log.warning("Sitemap at %s exceeds %d bytes, skipping", url, MAX_SITEMAP_BYTES)
            return None
        if url.endswith(".gz") or raw[:2] == b"\x1f\x8b":
            try:
                raw = gzip.decompress(raw)
            except Exception:
                pass
        if _looks_like_nonsitemap(url, r.headers.get("content-type", "")):
            return None
        return raw.decode("utf-8", errors="replace")
    except Exception:
        return None


def _looks_like_nonsitemap(url: str, content_type: str) -> bool:
    ct = content_type.lower().split(";")[0].strip()
    if any(u in ct for u in CONTENT_TYPES_UNLIKELY_XML):
        return True
    if ct in ("text/xml", "application/xml", "application/rss+xml", "application/atom+xml", ""):
        return False
    # unknown content type — accept, let content parsing decide
    return False


def _strip_ns(tag: str) -> str:
    return tag.split("}", 1)[-1] if tag.startswith("{") else tag


async def _parse_xml_sitemap(
    content: str,
    source_url: str,
    seen: set[str],
    results: list[dict[str, Any]],
    exclude: re.Pattern | None,
    domain_filter: str,
    max_urls: int,
    c: httpx.AsyncClient,
    parent_urls: set[str],
    depth: int,
    max_depth: int = MAX_RECURSION,
) -> bool:
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(content)
    except Exception:
        return False

    tag = _strip_ns(root.tag)

    if tag == "sitemapindex":
        sub_loc = f"{{{SITEMAP_NS}}}loc"
        for sm in root.findall(f"{{{SITEMAP_NS}}}sitemap"):
            loc = sm.find(sub_loc)
            if loc is not None and loc.text:
                url = _clean_url(loc.text.strip())
                await _parse_sitemap_recursive(
                    url, c, seen, results, exclude,
                    domain_filter, max_urls, parent_urls, depth + 1, max_depth,
                )
        return True

    if tag == "urlset":
        for u in root.findall(f"{{{SITEMAP_NS}}}url"):
            loc = u.find(f"{{{SITEMAP_NS}}}loc")
            if loc is None or not loc.text:
                continue
            url = _clean_url(loc.text.strip())
            if url in seen:
                continue
            seen.add(url)
            if exclude and exclude.search(url):
                continue
            if domain_filter and not url.startswith(domain_filter):
                continue
            entry: dict[str, Any] = {"url": url, "source": "sitemap"}
            lastmod = u.find(f"{{{SITEMAP_NS}}}lastmod")
            if lastmod is not None and lastmod.text:
                entry["last_modified"] = lastmod.text.strip()
            priority = u.find(f"{{{SITEMAP_NS}}}priority")
            if priority is not None and priority.text:
                try:
                    entry["priority"] = float(priority.text.strip())
                except ValueError:
                    pass
            changefreq = u.find(f"{{{SITEMAP_NS}}}changefreq")
            if changefreq is not None and changefreq.text:
                entry["changefreq"] = changefreq.text.strip()
            results.append(entry)
            if len(results) >= max_urls:
                break
        return True

    return False


def _parse_xml_fallback(
    content: str,
    seen: set[str],
    results: list[dict[str, Any]],
    domain_filter: str,
    max_urls: int,
    exclude: re.Pattern | None,
) -> None:
    urls: list[str] = []
    current_tag = ""
    in_loc = False

    def start(name: str, attrs: dict[str, str]) -> None:
        nonlocal in_loc, current_tag
        current_tag = _strip_ns(name)
        in_loc = current_tag == "loc"

    def end(name: str) -> None:
        nonlocal in_loc
        current_tag = _strip_ns(name)
        if current_tag == "loc":
            in_loc = False

    def data(text: str) -> None:
        if in_loc and text.strip():
            url = _clean_url(text.strip())
            if url:
                urls.append(url)

    try:
        p = xml.parsers.expat.ParserCreate()
        p.StartElementHandler = start
        p.EndElementHandler = end
        p.CharacterDataHandler = data
        p.Parse(content, True)
    except Exception:
        return

    for u in urls:
        if u in seen: continue
        seen.add(u)
        if exclude and exclude.search(u): continue
        if domain_filter and not u.startswith(domain_filter): continue
        results.append({"url": u, "source": "sitemap"})
        if len(results) >= max_urls: break


async def _parse_rss_sitemap(
    content: str,
    seen: set[str],
    results: list[dict[str, Any]],
    exclude: re.Pattern | None,
    domain_filter: str,
    max_urls: int,
) -> bool:
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(content)
    except Exception:
        return False

    for item in root.iter("item"):
        link = item.find("link")
        if link is not None and link.text:
            url = _clean_url(link.text.strip())
            if url in seen: continue
            seen.add(url)
            if exclude and exclude.search(url): continue
            if domain_filter and not url.startswith(domain_filter): continue
            results.append({"url": url, "source": "sitemap"})
            if len(results) >= max_urls:
                break
    return True


async def _parse_atom_sitemap(
    content: str,
    seen: set[str],
    results: list[dict[str, Any]],
    exclude: re.Pattern | None,
    domain_filter: str,
    max_urls: int,
) -> bool:
    try:
        import xml.etree.ElementTree as ET
        root = ET.fromstring(content)
    except Exception:
        return False

    for entry in root.iter(f"{{{ATOM_NS}}}entry"):
        for link in entry.findall(f"{{{ATOM_NS}}}link"):
            href = link.get("href", "")
            if href:
                url = _clean_url(href)
                if url in seen: continue
                seen.add(url)
                if exclude and exclude.search(url): continue
                if domain_filter and not url.startswith(domain_filter): continue
                results.append({"url": url, "source": "sitemap"})
                if len(results) >= max_urls:
                    break
    return True


def _parse_text_sitemap(
    content: str,
    seen: set[str],
    results: list[dict[str, Any]],
    domain_filter: str,
    max_urls: int,
    exclude: re.Pattern | None,
) -> None:
    for line in content.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        url = _clean_url(line)
        if not url:
            continue
        if not urllib.parse.urlparse(url).scheme.startswith("http"):
            continue
        if url in seen: continue
        seen.add(url)
        if exclude and exclude.search(url): continue
        if domain_filter and not url.startswith(domain_filter): continue
        results.append({"url": url, "source": "sitemap"})
        if len(results) >= max_urls:
            break


def _clean_url(url: str) -> str:
    url = html_mod.unescape(url)
    url = re.sub(r"\s+", "", url)
    return url


async def _fetch_page_html(url: str, c: httpx.AsyncClient) -> str | None:
    try:
        r = await c.get(url, timeout=10, follow_redirects=True)
        ct = (r.headers.get("content-type", "") or "").lower()
        if r.status_code == 200 and "text/html" in ct:
            return r.text
    except Exception:
        pass
    return None
