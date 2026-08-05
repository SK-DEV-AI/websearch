from __future__ import annotations

import base64
import hashlib
import math
import struct
from typing import Any

import httpx

import asyncio

from config import NV_KEY, NV_BASE, NV_EMBED_MODEL, _cached, _set_cache, get_http_client

_NV_KEYS: list[str] = [k.strip() for k in NV_KEY.split(",") if k.strip()] if NV_KEY else []
_nv_idx = 0
_NV_LOCK = asyncio.Lock()


async def _next_nv_key() -> str:
    global _nv_idx
    if not _NV_KEYS:
        raise RuntimeError("no NV_KEY configured")
    async with _NV_LOCK:
        k = _NV_KEYS[_nv_idx % len(_NV_KEYS)]
        _nv_idx = (_nv_idx + 1) % len(_NV_KEYS)
        return k


async def _embed(texts: list[str], input_type: str = "passage",
                 embed_model: str = "", encoding_format: str = "float") -> list[list[float]] | None:
    if not _NV_KEYS:
        return None
    results: list[list[float] | None] = [None] * len(texts)
    uncached = []
    uncached_idx = []
    for i, t in enumerate(texts):
        k = f"emb:{input_type}:{encoding_format}:{hashlib.sha256(t.encode()).hexdigest()}"
        entry = await _cached(k)
        if entry is not None:
            results[i] = entry
        else:
            uncached.append(t)
            uncached_idx.append(i)
    if uncached:
        nv_key = await _next_nv_key()
        model = embed_model or NV_EMBED_MODEL
        body_payload: dict[str, Any] = {
            "model": model, "input": uncached,
            "input_type": input_type, "encoding_format": encoding_format,
            "truncate": "END",
        }
        try:
            c = get_http_client()
            r = await c.post(f"{NV_BASE}/embeddings", json=body_payload,
                             headers={"Authorization": f"Bearer {nv_key}"}, timeout=30)
            if r.status_code == 200:
                data = r.json()
                for idx, row in zip(range(len(uncached)), data.get("data", [])):
                    emb = row.get("embedding")
                    if emb:
                        if encoding_format == "base64" and isinstance(emb, str):
                            decoded = base64.b64decode(emb)
                            emb = list(map(float, struct.unpack(f'{len(decoded)//4}f', decoded)))
                        orig_idx = uncached_idx[idx]
                        results[orig_idx] = emb
                        await _set_cache(f"emb:{input_type}:{encoding_format}:{hashlib.sha256(uncached[idx].encode()).hexdigest()}", emb)
        except (httpx.HTTPError, ValueError, KeyError):
            pass
    return results


def _cosine_sim(a: list[float], b: list[float]) -> float:
    dot = sum(ai * bi for ai, bi in zip(a, b))
    na = math.sqrt(sum(ai * ai for ai in a))
    nb = math.sqrt(sum(bi * bi for bi in b))
    return dot / (na * nb + 1e-10)


def _dedup_rank(items: list[dict], query_embed: list[float] | None, sim_threshold: float = 0.92) -> list[dict]:
    if not items:
        return items
    embeds = [r.get("_embedding") for r in items]
    has_embeds = query_embed is not None and all(e is not None for e in embeds)
    if has_embeds:
        deduped: list[dict] = []
        kept_embeds: list[list[float]] = []
        q_scores = [_cosine_sim(query_embed, e) for e in embeds]
        for i, r in enumerate(items):
            emb = embeds[i]
            is_dup = False
            for ke in kept_embeds:
                if _cosine_sim(emb, ke) > sim_threshold:
                    is_dup = True
                    break
            if not is_dup:
                r["_rel"] = round(q_scores[i], 4)
                deduped.append(r)
                kept_embeds.append(emb)
        deduped.sort(key=lambda x: x["_rel"], reverse=True)
        for r in deduped:
            r.pop("_embedding", None)
        return deduped
    seen: set[str] = set()
    result = []
    for r in items:
        k = str(r.get("title", r.get("url", "")))[:100]
        if k and k not in seen:
            seen.add(k)
            result.append(r)
    return result
