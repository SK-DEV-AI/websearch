"""OSV.dev vuln lane + NVD fallback (proposal 3).

Keyless. OSV serves exact CVE/GHSA lookups (aggregates GHSA+PyPA+RustSec+NVD);
NVD covers per-ID misses and keyword fallback. Advisory-only: every failure
returns [] or None, never raises.
"""

from __future__ import annotations

import logging
import re
import urllib.parse

logger = logging.getLogger("osv_vertical")

_CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.I)
_GHSA_RE = re.compile(r"GHSA-[\w-]+", re.I)
_RUSTSEC_RE = re.compile(r"RUSTSEC-\d{4}-\d+", re.I)
_VULN_WORD = re.compile(
    r"(?i)\b(cve\b|ghsa\b|rustsec\b|vulnerab\w*|security advisory|exploit\b)",
)

_OSV_VULN = "https://api.osv.dev/v1/vulns/"
_NVD_CVES = "https://services.nvd.nist.gov/rest/json/cves/2.0"
_TIMEOUT = 12.0


def is_vuln_query(query: str) -> bool:
    """True when the query carries CVE-ish intent (explicit IDs or vuln words)."""
    q = query or ""
    return bool(
        _CVE_RE.search(q) or _GHSA_RE.search(q)
        or _RUSTSEC_RE.search(q) or _VULN_WORD.search(q)
    )


def extract_ids(query: str) -> list[str]:
    """Explicit vuln IDs (CVE/GHSA/RUSTSEC), uppercased, deduped, order-kept."""
    seen: dict[str, None] = {}
    for m in (
        [m.group(0).upper() for m in _CVE_RE.finditer(query or "")]
        + [m.group(0).lower() for m in _GHSA_RE.finditer(query or "")]
        + [m.group(0).upper() for m in _RUSTSEC_RE.finditer(query or "")]
    ):
        seen.setdefault(m, None)
    return list(seen)


def _shape(rank: int, vid: str, title: str, url: str, snippet: str,
           published: str | None = None) -> dict:
    return {"title": title or vid, "url": url, "snippet": snippet,
            "published": published, "rank": rank}


def _shape_osv(rank: int, vid: str, v: dict) -> dict:
    summary = (v.get("summary") or "").strip()
    details = (v.get("details") or "").strip()
    aliases = ", ".join(a for a in (v.get("aliases") or [])[:4])
    affected = v.get("affected") or []
    pkgs = ", ".join(
        (a.get("package") or {}).get("name", "")
        for a in affected[:3] if isinstance(a, dict)
    )
    fixed = ""
    for a in affected:
        if not isinstance(a, dict):
            continue
        for r in (a.get("ranges") or []):
            for ev in (r.get("events") or [])[-2:]:
                if isinstance(ev, dict) and ev.get("fixed"):
                    fixed = ev["fixed"]
                    break
    bits = [summary or details[:300]]
    if aliases:
        bits.append(f"aliases: {aliases}")
    if pkgs:
        bits.append(f"affected: {pkgs}")
    if fixed:
        bits.append(f"fixed in: {fixed}")
    refs = [r.get("url", "") for r in (v.get("references") or [])[:2]
            if isinstance(r, dict) and r.get("url")]
    if refs:
        bits.append("refs: " + " ".join(refs))
    return _shape(rank, vid, summary or vid,
                  f"https://osv.dev/vulnerability/{urllib.parse.quote(vid)}",
                  " | ".join(b for b in bits if b)[:600],
                  (v.get("published") or "")[:10] or None)


def _shape_nvd(rank: int, item: dict) -> dict | None:
    cve = item.get("cve") or {}
    vid = cve.get("id", "")
    if not vid:
        return None
    descs = cve.get("descriptions") or []
    en = next((d.get("value", "") for d in descs
               if d.get("lang") == "en"), "")
    pub = (cve.get("published") or "")[:10] or None
    return _shape(rank, vid, f"{vid}: {(en[:120] or 'NVD entry')}",
                  f"https://nvd.nist.gov/vuln/detail/{vid}",
                  en[:400], pub)


async def _osv_vuln(vuln_id: str) -> dict | None:
    try:
        from config import get_http_client
        c = get_http_client()
        r = await c.get(_OSV_VULN + urllib.parse.quote(vuln_id),
                        timeout=_TIMEOUT)
        if r.status_code != 200:
            return None
        v = r.json()
        return v if isinstance(v, dict) else None
    except Exception as e:
        logger.debug("osv lookup %s failed: %s", vuln_id, e)
        return None


async def _nvd(params: dict) -> list[dict]:
    try:
        from config import get_http_client
        c = get_http_client()
        r = await c.get(_NVD_CVES, params=params, timeout=_TIMEOUT,
                        headers={"User-Agent": "websearch-mcp/1.0"})
        if r.status_code != 200:
            return []
        v = r.json()
        return v.get("vulnerabilities", []) or []
    except Exception as e:
        logger.debug("nvd query failed: %s", e)
        return []


async def search_vuln(query: str, count: int = 5) -> list[dict]:
    """CVE/GHSA/RUSTSEC lookup via OSV, NVD per-ID fallback, NVD keyword last."""
    out: list[dict] = []
    ids = extract_ids(query)[:max(count, 1)]
    for vid in ids:
        v = await _osv_vuln(vid)
        if isinstance(v, dict) and v.get("id"):
            out.append(_shape_osv(len(out), v.get("id", vid), v))
            continue
        for item in await _nvd({"cveId": vid.upper()}):
            shaped = _shape_nvd(len(out), item)
            if shaped:
                out.append(shaped)
                break
        if len(out) >= count:
            break
    if not ids and is_vuln_query(query):
        for item in await _nvd({"keywordSearch": query,
                                "resultsPerPage": min(count, 20)}):
            shaped = _shape_nvd(len(out), item)
            if shaped:
                out.append(shaped)
            if len(out) >= count:
                break
    return out[:count]


if __name__ == "__main__":
    import asyncio

    async def check():
        assert is_vuln_query("CVE-2024-3094 xz backdoor")
        assert is_vuln_query("GHSA-xxxx-yyyy-zzzz details")
        assert not is_vuln_query("how to fix a leaking kitchen faucet")
        assert extract_ids("cve-2024-3094 and CVE-2024-3094") == ["CVE-2024-3094"]
        hits = await search_vuln("CVE-2024-3094")
        print(f"osv CVE-2024-3094: {len(hits)} hits")
        for h in hits[:2]:
            print(" -", h["title"][:90], "|", h["url"])

    asyncio.run(check())
