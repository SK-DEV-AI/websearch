"""Query expansion — model-generated variations of a short, vague query.

Generates diverse search query variations by combining:
- Synonym/paraphrase variations
- Broader/narrower scope variations
- Different phrasings for the same intent

The model is chosen by llm.chat, which prefers the free in-tool model. This
module used to call Groq directly, so without a Groq key it silently returned
the query unchanged and wasted the capability.
"""

from __future__ import annotations

import asyncio
import logging
import re

logger = logging.getLogger("query_expand")


async def expand_query(query: str, model: str | None = None) -> list[str]:
    """Expand a short/vague query into diverse search variations.

    Returns [original, variation1, variation2, ...] up to 4 total.
    Only expands queries that are short (<5 words or <=80 chars).
    `model` is unused: the endpoint picks its own model, because a Groq model
    name forced onto the in-tool endpoint would fail every call.
    """
    if len(query.split()) >= 8 or len(query) > 80:
        return [query]

    prompt = (
        "Output exactly 2 keyword-only search query variations. Each variation "
        "is one line of 3-6 keywords. No questions. No sentences. No numbering. "
        "No prefixes. No explanation.\n\n"
        f"Original query: {query}"
    )
    try:
        from llm import chat
        text, _finish, err, _att = await chat(
            [
                {"role": "system", "content": "Generate exactly 2 keyword-only search query variations. One query per line. No numbering. No prefixes. No explanations."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=128, attempts=1, timeout=15,
            kind="expand", temperature=0.7,
        )
        if not text:
            if err:
                logger.debug("query_expand: %s", err)
            return [query]
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
