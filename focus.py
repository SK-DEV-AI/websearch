"""BM25 query-relevance content filter.

Splits extracted markdown into blocks (paragraphs/sections), scores
each against the query using BM25, returns only relevant blocks.
Post-cache filter — runs on cached content, doesn't cause re-fetch.

Usage::
    filtered = filter_by_relevance(content, "pricing plans")
"""

from __future__ import annotations

import logging
import math
import re

logger = logging.getLogger("focus")

_STOPWORDS = frozenset({
    "a", "an", "and", "are", "as", "at", "be", "but", "by", "can",
    "could", "do", "does", "for", "from", "had", "has", "have",
    "how", "if", "in", "is", "it", "its", "may", "might", "more",
    "most", "not", "of", "on", "or", "shall", "should", "so", "than",
    "that", "the", "their", "them", "then", "there", "these", "they",
    "this", "to", "us", "was", "we", "were", "what", "when", "where",
    "which", "while", "who", "will", "with", "would", "you", "your",
    "after", "also", "any", "each", "else", "has", "its", "just",
    "like", "many", "much", "no", "nor", "now", "off", "only",
    "other", "our", "out", "over", "own", "same", "some", "such",
    "than", "too", "up", "very",
})


def _tokenize(text: str) -> list[str]:
    """Tokenize text into lowercase words, filter stopwords."""
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9'+-]{1,}", text.lower())
    return [w for w in words if w not in _STOPWORDS and len(w) >= 2]


def _split_blocks(content: str) -> list[dict]:
    """Split content into blocks with heading context.

    Each block starts at a heading or paragraph break.
    Inherits the most recent heading as context.
    Returns list of {heading, lines} dicts.
    """
    lines = content.split("\n")
    blocks = []
    current_heading = ""
    current_lines: list[str] = []

    for line in lines:
        stripped = line.strip()
        if re.match(r"^#{1,6}\s", stripped):
            if current_lines:
                text = "\n".join(current_lines).strip()
                if text:
                    blocks.append({"heading": current_heading, "lines": current_lines})
            current_heading = stripped.lstrip("#").strip()
            current_lines = [line]
        elif stripped == "":
            if current_lines and any(l.strip() for l in current_lines[-3:]):
                text = "\n".join(current_lines).strip()
                if text:
                    blocks.append({"heading": current_heading, "lines": current_lines})
                current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        text = "\n".join(current_lines).strip()
        if text:
            blocks.append({"heading": current_heading, "lines": current_lines})

    return blocks


def filter_by_relevance(content: str, query: str,
                        threshold_pct: float = 0.3) -> str:
    """Filter *content* to blocks most relevant to *query* via BM25.

    Args:
        content: Full extracted text (markdown).
        query: Relevance query (e.g. "pricing", "installation guide").
        threshold_pct: Keep blocks above this fraction of max score.

    Returns:
        Filtered content with relevant blocks + heading context.
    """
    if not query or not content:
        return content

    query_tokens = _tokenize(query)
    if not query_tokens:
        return content

    blocks = _split_blocks(content)
    if len(blocks) <= 1:
        return content  # Already atomic

    block_tokens = [_tokenize(" ".join(blk["lines"])) for blk in blocks]

    k1 = 1.5
    b = 0.75
    N = len(blocks)

    df: dict[str, int] = {}
    for tokens in block_tokens:
        for t in set(tokens):
            df[t] = df.get(t, 0) + 1

    avg_len = sum(len(t) for t in block_tokens) / max(N, 1)

    scores = []
    for tokens in block_tokens:
        score = 0.0
        if not tokens:
            scores.append(0.0)
            continue
        length = len(tokens)
        for q in query_tokens:
            tf = tokens.count(q)
            if tf == 0:
                continue
            idf = math.log(1 + (N - df.get(q, 0) + 0.5) / (df.get(q, 0) + 0.5))
            score += idf * ((tf * (k1 + 1)) / (tf + k1 * (1 - b + b * length / avg_len)))
        scores.append(score)

    if not scores or max(scores) == 0:
        return content

    max_score = max(scores)
    threshold = max_score * threshold_pct

    heading_scores = [
        sum(1 for q in query_tokens if q in blk["heading"].lower())
        for blk in blocks
    ]

    selected: list[str] = []
    seen_headings: set[str] = set()
    block_count = 0

    for i, blk in enumerate(blocks):
        keep = scores[i] >= threshold or heading_scores[i] > 0
        if i == 0:
            keep = True  # Always keep first paragraph (summary/intro)
        if keep:
            h = blk["heading"]
            block_lines = blk["lines"]
            if h:
                if h in seen_headings:
                    # Heading already emitted — keep content, drop only the
                    # duplicated heading line so we don't re-announce it.
                    block_lines = [ln for ln in block_lines
                                   if not ln.lstrip().startswith("#")]
                    if not block_lines:
                        block_lines = blk["lines"]
                else:
                    seen_headings.add(h)
            selected.extend(block_lines)
            selected.append("")
            block_count += 1

    if not selected:
        # Fallback: top 5 by score
        sorted_i = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
        for idx in sorted_i[:5]:
            selected.extend(blocks[idx]["lines"])
            selected.append("")
        block_count = min(5, len(blocks))

    result = "\n".join(selected).strip()
    ratio = block_count / max(len(blocks), 1)
    if ratio < 0.5 and block_count < 20:
        result += f"\n\n[focus filtered: kept {block_count}/{len(blocks)} blocks relevant to \"{query}\"]"

    return result
