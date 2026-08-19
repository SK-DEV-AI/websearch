"""Keyless academic search engines (steal B3 — OpenAlex/Crossref/PubMed/Europe PMC).

All four are free REST APIs with polite-pool headers (no keys). Verified live
2026-08-19: OpenAlex 200/1.2M hits, Crossref 200, PubMed 200, Europe PMC 200.
Each function returns the standard [{title,url,snippet}] engine shape so the
merge treats them like any web engine.
"""

from __future__ import annotations

import json
import urllib.parse

from config import get_http_client

TIMEOUT = 12.0
_POOL_UA = "websearch-mcp/1.0 (academic-research; mailto:none)"

# OpenAlex abstract is an inverted index — rebuild the sentence.
def _openalex_abstract(ainv: dict | None) -> str:
    if not ainv:
        return ""
    pos: dict[int, str] = {}
    for word, idxs in ainv.items():
        for i in idxs:
            pos[i] = word
    return " ".join(pos[i] for i in sorted(pos))


async def search_openalex(query: str, count: int = 5) -> list[dict]:
    url = ("https://api.openalex.org/works?"
           + urllib.parse.urlencode({"search": query, "per-page": min(count, 25)}))
    c = get_http_client()
    try:
        r = await c.get(url, timeout=TIMEOUT, headers={"User-Agent": _POOL_UA})
        if r.status_code != 200:
            return []
        v = r.json()
    except Exception:
        return []
    out = []
    for it in v.get("results", []):
        title = it.get("title") or ""
        if not title:
            continue
        loc = it.get("primary_location") or {}
        url = loc.get("landing_page_url") or it.get("doi") or ""
        if not url:
            continue
        year = it.get("publication_year")
        abstract = _openalex_abstract(it.get("abstract_inverted_index"))
        snippet = f"{abstract[:200] or it.get('type') or ''} · cited {it.get('cited_by_count', 0)}× · {year}"
        out.append({"title": title, "url": url, "snippet": snippet,
                    "source": "openalex", "date": str(year) if year else ""})
    return out[:count]


async def search_crossref(query: str, count: int = 5) -> list[dict]:
    url = ("https://api.crossref.org/works?"
           + urllib.parse.urlencode({"query": query, "rows": min(count, 20),
                                     "select": "title,DOI,issued,author,container-title"}))
    c = get_http_client()
    try:
        r = await c.get(url, timeout=TIMEOUT,
                        headers={"User-Agent": _POOL_UA,
                                 "mailto": "research@example.invalid"})
        if r.status_code != 200:
            return []
        v = r.json()
    except Exception:
        return []
    out = []
    for it in v.get("message", {}).get("items", []):
        titles = it.get("title") or []
        if not titles:
            continue
        year = (it.get("issued", {}).get("date-parts") or [[None]])[0][0]
        authors = ", ".join(
            f"{a.get('family', '')} {a.get('given', '')}".strip()
            for a in (it.get("author") or [])[:3])
        journal = (it.get("container-title") or [""])[0]
        snippet = f"{journal}{' — ' if journal else ''}{authors} · {year}"
        out.append({"title": titles[0], "url": f"https://doi.org/{it['DOI']}",
                    "snippet": snippet, "source": "crossref",
                    "date": str(year) if year else ""})
    return out[:count]


async def search_pubmed(query: str, count: int = 5) -> list[dict]:
    """Two-step: esearch → PMID list, then esummary → titles."""
    base = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    esearch = (base + "esearch.fcgi?"
               + urllib.parse.urlencode({"db": "pubmed", "term": query,
                                         "retmax": min(count, 10), "retmode": "json"}))
    c = get_http_client()
    try:
        r = await c.get(esearch, timeout=TIMEOUT,
                        headers={"User-Agent": _POOL_UA})
        if r.status_code != 200:
            return []
        ids = r.json().get("esearchresult", {}).get("idlist", [])
        if not ids:
            return []
        esummary = (base + "esummary.fcgi?"
                    + urllib.parse.urlencode({"db": "pubmed", "id": ",".join(ids),
                                              "retmode": "json"}))
        r2 = await c.get(esummary, timeout=TIMEOUT, headers={"User-Agent": _POOL_UA})
        if r2.status_code != 200:
            return []
        docs = r2.json().get("result", {})
    except Exception:
        return []
    out = []
    for pmid in ids:
        d = docs.get(pmid) or {}
        title = d.get("title") or ""
        if not title:
            continue
        year = ""
        for k in ("pubdate", "epubdate"):
            if d.get(k):
                year = d[k][:4]
                break
        journal = d.get("fulljournalname") or d.get("source") or ""
        out.append({"title": title, "url": f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/",
                    "snippet": f"{journal} · PMID {pmid}" if journal else f"PMID {pmid}",
                    "source": "pubmed", "date": year})
    return out[:count]


async def search_europepmc(query: str, count: int = 5) -> list[dict]:
    url = ("https://www.ebi.ac.uk/europepmc/webservices/rest/search?"
           + urllib.parse.urlencode({"query": query, "format": "json",
                                     "pageSize": min(count, 20)}))
    c = get_http_client()
    try:
        r = await c.get(url, timeout=TIMEOUT, headers={"User-Agent": _POOL_UA})
        if r.status_code != 200:
            return []
        v = r.json()
    except Exception:
        return []
    out = []
    for it in v.get("resultList", {}).get("result", []):
        title = it.get("title") or ""
        if not title:
            continue
        pmid = it.get("pmid") or ""
        pmcid = it.get("pmcid") or ""
        url = (f"https://europepmc.org/article/PMC/{pmcid}"
               if pmcid else f"https://pubmed.ncbi.nlm.nih.gov/{pmid}/")
        year = it.get("pubYear") or ""
        journal = it.get("journalTitle") or ""
        snippet = f"{journal}{' · ' if journal else ''}full text open" if it.get("inPMC") == "Y" else journal
        out.append({"title": title, "url": url, "snippet": snippet,
                    "source": "europepmc", "date": str(year)})
    return out[:count]


_ENGINES = {"openalex": search_openalex, "crossref": search_crossref,
            "pubmed": search_pubmed, "europepmc": search_europepmc}


async def search_academic(query: str, count: int = 5, source: str = "openalex") -> list[dict]:
    """Dispatch: source in {openalex, crossref, pubmed, europepmc}."""
    fn = _ENGINES.get(source)
    if not fn:
        return []
    return await fn(query, count)