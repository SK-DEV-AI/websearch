"""Query expansion via Groq API — round-robin multi-key, non-blocking.

Generates diverse search query variations by combining:
- Synonym/paraphrase variations
- Broader/narrower scope variations
- Different phrasings for the same intent
"""

from __future__ import annotations

import asyncio
import logging
import re

from config import _KeyRotator

logger = logging.getLogger("query_expand")

_groq_rotator = _KeyRotator("GROQ_API_KEYS")
_next_key = _groq_rotator.next


async def expand_query(query: str, model: str | None = None) -> list[str]:
    """Expand a short/vague query into diverse search variations.

    Returns [original, variation1, variation2, ...] up to 4 total.
    Only expands queries that are short (<5 words or <=80 chars).
    model defaults to the cheap lane (llama-3.1-8b-instant).
    """
    if len(query.split()) >= 8 or len(query) > 80:
        return [query]
    key = await _next_key()
    if not key:
        return [query]
    if model is None:
        from config import GROQ_CHEAP_MODEL
        model = GROQ_CHEAP_MODEL

    prompt = (
        "Output exactly 2 keyword-only search query variations. Each variation "
        "is one line of 3-6 keywords. No questions. No sentences. No numbering. "
        "No prefixes. No explanation.\n\n"
        f"Original query: {query}"
    )
    try:
        from config import get_http_client
        c = get_http_client()
        r = await c.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}", "Content-Type": "application/json"},
            json={
                "model": model,
                "messages": [
                    {"role": "system", "content": "Generate exactly 2 keyword-only search query variations. One query per line. No numbering. No prefixes. No explanations."},
                    {"role": "user", "content": prompt},
                ],
                "temperature": 0.7,
                "max_tokens": 128,
            },
            timeout=15,
        )
        if r.status_code == 200:
            text = r.json()["choices"][0]["message"]["content"].strip()
            lines = []
            for q in text.split("\n"):
                raw = q.strip()
                if not raw or len(raw) <= 5:
                    continue
                cleaned = re.sub(r'^[\s*\-•·>]+|^[\d]+[\.\)]\s*', '', raw).strip().strip('"\'[]')
                if not cleaned or len(cleaned) <= 5:
                    cleaned = raw
                if any(kw in cleaned.lower() for kw in [
                    "here", "variation", "query:", "---", "original",
                    "broad", "specific", "alternative",
                ]):
                    continue
                lines.append(cleaned)
            # Deduplicate and keep only unique queries
            seen = {query.lower()}
            unique = []
            for q in lines:
                ql = q.lower()
                if ql not in seen and len(q) > 5:
                    seen.add(ql)
                    unique.append(q)
            if unique:
                return [query] + unique[:3]
    except Exception as e:
        logger.warning(f"query_expand failed: {type(e).__name__}: {e}")
    return [query]
