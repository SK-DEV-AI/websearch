"""Common Crawl CDX + Hugging Face search engines (steal engines #6).

Both keyless, verified live 2026-08-19:
- CDX: https://index.commoncrawl.org/<CC-MAIN-YYYY-WW>-index?url=...&output=json
- HF:  https://huggingface.co/api/models|datasets?search=...&limit=N

Niche engines — opt-in only via engines=[...] in search_multi, never in the
default fan-out (archival/URL-level + ML-vertical intents respectively).
"""
from __future__ import annotations

import asyncio
import httpx

_COLLECTION_URL = "https://index.commoncrawl.org/collinfo.json"
_CDX_BASE = "https://index.commoncrawl.org"
_HF_API = "https://huggingface.co/api"

_latest_collection: str | None = None
_collection_ts: float = 0.0
_COLLECTION_TTL = 6 * 3600

_DOMAIN_RE = __import__("re").compile(
    r"^(?=.{1,253}$)([a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,}$")


async def _latest_index(client: httpx.AsyncClient) -> str:
    global _latest_collection, _collection_ts
    import time
    now = time.monotonic()
    if _latest_collection and now - _collection_ts < _COLLECTION_TTL:
        return _latest_collection
    r = await client.get(_COLLECTION_URL)
    r.raise_for_status()
    data = r.json()
    _latest_collection = data[0]["id"]
    _collection_ts = now
    return _latest_collection


async def search_commoncrawl(query: str, count: int = 5) -> list[dict]:
    """CDX index: domain match for bare domains, prefix match otherwise."""
    query = (query or "").strip()
    if not query:
        return [{"error": "empty query"}]
    out: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True,
                                     headers={"User-Agent": "websearch-mcp (research)"}) as c:
            coll = await _latest_index(c)
            match = "domain" if _DOMAIN_RE.match(query) else "prefix"
            r = await c.get(f"{_CDX_BASE}/{coll}-index",
                            params={"url": query, "matchType": match,
                                    "output": "json", "limit": str(count)})
            if r.status_code == 200:
                for line in r.text.splitlines():
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        rec = __import__("json").loads(line)
                    except Exception:
                        continue
                    if not isinstance(rec, dict) or not rec.get("url"):
                        continue
                    ts = rec.get("timestamp", "")
                    date = f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}" if len(ts) >= 8 else ""
                    out.append({
                        "title": rec.get("url", ""),
                        "url": rec.get("url", ""),
                        "snippet": (f"Archived {date} · status {rec.get('status', '?')}"
                                    f" · {rec.get('mime', '?')}"
                                    + (f" · digest {rec['digest'][:8]}" if rec.get("digest") else "")),
                        "engine": "commoncrawl",
                    })
                    if len(out) >= count:
                        break
    except Exception as e:
        return [{"error": f"commoncrawl: {e}"}]
    return out


async def search_huggingface(query: str, count: int = 5) -> list[dict]:
    """HF models + datasets search, merged models-first."""
    query = (query or "").strip()
    if not query:
        return [{"error": "empty query"}]
    out: list[dict] = []
    try:
        async with httpx.AsyncClient(timeout=15, follow_redirects=True,
                                     headers={"User-Agent": "websearch-mcp (research)"}) as c:
            async def _one(kind: str) -> list[dict]:
                r = await c.get(f"{_HF_API}/{kind}",
                                params={"search": query, "limit": str(count)})
                if r.status_code != 200:
                    return []
                recs = r.json()
                res = []
                for m in recs:
                    if kind == "models":
                        res.append({
                            "title": m.get("modelId", ""),
                            "url": f"https://huggingface.co/{m.get('modelId', '')}",
                            "snippet": (f"pipeline={m.get('pipeline_tag', '?')} · "
                                        f"downloads={m.get('downloads', 0):,} · "
                                        f"likes={m.get('likes', 0)}"
                                        + (f" · {m['library_name']}" if m.get("library_name") else "")),
                            "engine": "huggingface",
                        })
                    else:
                        res.append({
                            "title": m.get("id", ""),
                            "url": f"https://huggingface.co/datasets/{m.get('id', '')}",
                            "snippet": (f"dataset · downloads={m.get('downloads', 0):,} · "
                                        f"likes={m.get('likes', 0)}"
                                        + (f" · updated {m.get('lastModified', '')[:10]}" if m.get("lastModified") else "")),
                            "engine": "huggingface",
                        })
                return res

            models, datasets = await asyncio.gather(_one("models"), _one("datasets"))
            out = (models + datasets)[:count]
    except Exception as e:
        return [{"error": f"huggingface: {e}"}]
    return out


if __name__ == "__main__":
    async def _demo() -> None:
        cc = await search_commoncrawl("example.com", 3)
        hf = await search_huggingface("bert", 4)
        assert cc and cc[0].get("url"), f"CDX empty: {cc}"
        assert hf and hf[0].get("url"), f"HF empty: {hf}"
        assert hf[0]["engine"] == "huggingface" and cc[0]["engine"] == "commoncrawl"
        print(f"CDX ok ({len(cc)}): {cc[0]['url']}")
        print(f"HF  ok ({len(hf)}): {hf[0]['title']}")

    asyncio.run(_demo())