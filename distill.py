"""In-tool model: compress a large result before the caller reads it.

Why this exists: a fetched page can be 50k chars. The calling agent then pays
for all of it in its own context even when it only needed one fact. Running a
cheap model inside the tool can return a smaller, more useful result.

The danger, and the reason this module is shaped the way it is: a summary is a
*claim about completeness*. Truncation is honest about being partial; a
summary is not. A summariser that silently drops the one sentence the caller
needed is worse than no summariser, because the caller cannot tell.

So every distilled result must satisfy all four:

1. **Opt-in only.** Nothing is distilled unless the caller asks. Default off.
2. **Threshold skip.** Below THRESHOLD_CHARS a summary costs more (latency and
   risk) than it saves, so it returns the raw text and says it did.
3. **Explicit provenance.** The result states that it is a distillation, the
   raw size, what was dropped, and how to get the untouched text.
4. **Grounding check.** Numbers and quoted spans in the summary are checked
   against the source without another model call. Anything that cannot be
   found is listed as unverified rather than presented as fact.

Point 4 is the important one: it catches the failure mode that matters
(invented facts dressed up as summary) cheaply and deterministically.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
from typing import Any

import httpx

import cache as cache_mod

logger = logging.getLogger(__name__)

# Below this, distilling is not worth the latency or the risk of loss.
THRESHOLD_CHARS = 4000
# Above this, the source is chunked rather than sent in one request.
MAX_INPUT_CHARS = 200_000
CHUNK_CHARS = 60_000
MAX_SUMMARY_CHARS = 6000
TIMEOUT_SECONDS = 90
# Reasoning models spend tokens on reasoning_content BEFORE emitting content, so
# a small max_tokens yields a 200 response with empty content. Budget for it.
MAX_TOKENS = 3000
MAX_TOKENS_CEILING = 12_000
RETRIES = 3
RETRY_BASE_SLEEP = 1.5

DEFAULT_BASE_URL = "http://127.0.0.1:11435/v1"
DEFAULT_MODEL = "reflection/beam:free"
_JCODE_CONFIG = os.path.expanduser("~/.jcode/config.toml")


def _env_int(name: str, default: int, lo: int, hi: int) -> int:
    """Read an int from the environment, clamped. Bad input falls back to the
    default rather than raising - a typo in an env var must not break a tool."""
    raw = os.environ.get(name, "")
    if not str(raw).strip():
        return default
    try:
        v = int(raw)
    except (TypeError, ValueError):
        logger.warning("distill: %s=%r is not an int, using %d", name, raw, default)
        return default
    return max(lo, min(hi, v))


def settings() -> dict:
    """Every tunable, read fresh so an env change applies without a restart.

    Env var                     default  meaning
    DISTILL_RETRIES                 3    attempts per model call (1-10)
    DISTILL_MAX_TOKENS            3000   initial completion budget (256-32000)
    DISTILL_MAX_TOKENS_CEILING   12000   ceiling when escalating (512-64000)
    DISTILL_TIMEOUT                90    per-request seconds (5-600)
    DISTILL_THRESHOLD_CHARS      4000    below this, skip distillation
    """
    initial = _env_int("DISTILL_MAX_TOKENS", MAX_TOKENS, 256, 32_000)
    ceiling = _env_int("DISTILL_MAX_TOKENS_CEILING", MAX_TOKENS_CEILING,
                       initial, 64_000)
    return {
        "retries": _env_int("DISTILL_RETRIES", RETRIES, 1, 10),
        "max_tokens": initial,
        "max_tokens_ceiling": ceiling,
        "timeout": _env_int("DISTILL_TIMEOUT", TIMEOUT_SECONDS, 5, 600),
        "threshold_chars": _env_int("DISTILL_THRESHOLD_CHARS", THRESHOLD_CHARS,
                                    500, 200_000),
        "retry_base_sleep": RETRY_BASE_SLEEP,
    }

_SYSTEM = """You compress web content for another AI agent that will act on your output.

Hard rules, in order of importance:

1. Use ONLY facts present in the source text. Never add outside knowledge.
2. If you are unsure whether something is in the source, leave it out rather
   than paraphrasing it into existence.
3. Never invent or round a number, date, name, URL, version or quote. Copy them
   exactly as written.
4. Keep every number, date, identifier and proper noun that appears in the part
   you summarise. Dropping a figure is a failure; the caller may need it.
5. Prefer the source's own words for anything that reads like a claim, a price,
   a limit, or a specification.
6. Do not editorialise. No "this is important", no recommendations, no summary
   of significance. Report what the page says.

Output exactly these sections and nothing else:

## Summary
Two to six sentences on what the page is and what it contains.

## Key facts
Bullets. Each bullet is one verifiable fact with its figure or name inline.
Include as many as are needed to cover the substantive content.

## Quoted
Up to five short exact quotes that carry the most load, each attributed to the
section it came from if that is visible.

## Dropped
One line naming what you did NOT include (navigation, boilerplate, comments,
repeated tables, ads). If you dropped nothing substantive, write "nothing"."""

_QUESTION_SUFFIX = """

The caller is specifically looking for: {question}

Put anything that answers that question first, in the Summary section, and keep
it verbatim where possible. Still report every relevant figure - do not
summarise away specifics just because they are not the primary question."""


# ───────────────────────── configuration ─────────────────────────


def _config_key() -> str:
    """Convenience fallback: read the local endpoint key if a jcode config
    exists. Purely optional - env wins, and nothing breaks without it."""
    try:
        with open(_JCODE_CONFIG, "r", encoding="utf-8") as fh:
            text = fh.read()
    except Exception:
        return ""
    # Deliberately narrow: only the cmdcode2api provider block.
    block = re.search(r"\[providers\.cmdcode2api\](.*?)(?:\n\[|$)", text, re.S)
    if not block:
        return ""
    m = re.search(r'^\s*api_key\s*=\s*"([^"]+)"', block.group(1), re.M)
    return m.group(1) if m else ""


def resolve() -> dict:
    """Return {ok, base_url, model, api_key, source, error} without leaking the
    key value anywhere except the request itself."""
    base_url = (os.environ.get("DISTILL_BASE_URL") or DEFAULT_BASE_URL).rstrip("/")
    model = os.environ.get("DISTILL_MODEL") or DEFAULT_MODEL
    key = (os.environ.get("DISTILL_API_KEY") or "").strip()
    source = "env"
    if not key:
        key = _config_key()
        source = "jcode config" if key else "none"
    if not key:
        return {
            "ok": False, "base_url": base_url, "model": model, "api_key": "",
            "source": source,
            "error": (
                "No API key for the in-tool model. Set DISTILL_API_KEY (and "
                "optionally DISTILL_BASE_URL / DISTILL_MODEL) for any "
                "OpenAI-compatible endpoint."
            ),
            "fix": "export DISTILL_API_KEY=<key>  # DEFAULT endpoint is "
                   f"{DEFAULT_BASE_URL} with model {DEFAULT_MODEL}",
        }
    return {"ok": True, "base_url": base_url, "model": model,
            "api_key": key, "source": source}


def available() -> dict:
    r = resolve()
    s = settings()
    return {
        "available": r["ok"],
        "base_url": r["base_url"],
        "model": r["model"],
        "key_source": r["source"],
        "threshold_chars": s["threshold_chars"],
        "retries": s["retries"],
        "max_tokens": s["max_tokens"],
        "max_tokens_ceiling": s["max_tokens_ceiling"],
        "timeout": s["timeout"],
        "error": r.get("error", ""),
    }


# ───────────────────────── grounding ─────────────────────────

_NUM = re.compile(r"\d[\d,._]*\d|\d")
# Single-line, non-greedy: pairing across lines produced false alarms by
# matching the close of one quotation to the open of another.
_QUOTE = re.compile(r'["“]([^"”\n]{8,300})["”]')
_QUOTED_SECTION = re.compile(r"^##\s*Quoted\s*$(.*?)(?=^##\s|\Z)", re.S | re.M)


def _norm_num(s: str) -> str:
    return s.replace(",", "").replace("_", "").rstrip(".")


def grounding_check(summary: str, source: str) -> dict:
    """Deterministically check the summary against the source.

    Numbers are the cheapest hallucination detector: a model that invents a
    figure is usually inventing it wholesale, so the digit string will not
    appear in the source at all. Quoted spans are checked the same way.

    This cannot prove a summary is complete, only that what it asserts is
    present in the source. That is the useful direction.
    """
    src_nums = {_norm_num(m.group(0)) for m in _NUM.finditer(source)}
    out_nums: list[str] = []
    missing_nums: list[str] = []
    seen: set[str] = set()
    for m in _NUM.finditer(summary):
        raw = m.group(0)
        n = _norm_num(raw)
        if not n or n in seen:
            continue
        seen.add(n)
        (out_nums if n in src_nums else missing_nums).append(raw)

    src_compact = re.sub(r"\s+", " ", source)

    # Only the Quoted section claims verbatim reproduction. Elsewhere a quote
    # mark is emphasis or paraphrase, and flagging those produced false alarms
    # that would teach the caller to ignore this flag.
    section = _QUOTED_SECTION.search(summary)
    quote_source = section.group(1) if section else ""
    missing_quotes: list[str] = []
    checked_quotes = 0
    if quote_source:
        for m in _QUOTE.finditer(quote_source):
            q = re.sub(r"\s+", " ", m.group(1)).strip()
            if len(q) < 8:
                continue
            checked_quotes += 1
            if q not in src_compact:
                missing_quotes.append(q[:120])

    return {
        "numbers_checked": len(seen),
        "numbers_unverified": missing_nums[:40],
        "quotes_checked": checked_quotes,
        "quotes_unverified": missing_quotes[:10],
        "quoted_section_found": bool(section),
        "ok": not missing_nums and not missing_quotes,
    }


# ───────────────────────── model call ─────────────────────────


async def _ask(text: str, question: str, cfg: dict,
               *, max_tokens: int | None = None) -> tuple[str, str, int]:
    """One model call, with retries. Returns (summary, error, attempts).

    A free/cheap endpoint is flaky in three distinct ways and each needs its
    own handling:

    * transport failure or 5xx/429 -> retry with backoff, it may just be busy
    * HTTP 200 with EMPTY content -> the usual cause is reasoning tokens having
      consumed max_tokens, so retry with a bigger budget before giving up
    * other non-200 -> treat as permanent, do not waste attempts

    `reasoning_content` is deliberately ignored: it is the model's chain of
    thought, not a report, and using it as the summary would be dishonest.
    """
    s = settings()
    retries = s["retries"]
    timeout = s["timeout"]
    ceiling = s["max_tokens_ceiling"]
    base_sleep = s["retry_base_sleep"]
    max_tokens = int(max_tokens or s["max_tokens"])
    system = _SYSTEM + (_QUESTION_SUFFIX.format(question=question) if question else "")
    headers = {"Authorization": f"Bearer {cfg['api_key']}",
               "Content-Type": "application/json"}
    last = "no attempt made"
    for attempt in range(1, retries + 1):
        payload = {
            "model": cfg["model"],
            "temperature": 0,
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": text},
            ],
        }
        try:
            async with httpx.AsyncClient(timeout=timeout) as c:
                r = await c.post(f"{cfg['base_url']}/chat/completions",
                                 json=payload, headers=headers)
        except Exception as e:  # noqa: BLE001
            last = f"endpoint unreachable: {type(e).__name__}: {e}"
            logger.info("distill: attempt %d/%d failed: %s", attempt, retries, last)
            if attempt < retries:
                await asyncio.sleep(base_sleep * attempt)
            continue

        if r.status_code in (429, 500, 502, 503, 504):
            last = f"HTTP {r.status_code}"
            logger.info("distill: attempt %d/%d got %s, retrying", attempt, retries, last)
            if attempt < retries:
                await asyncio.sleep(base_sleep * attempt * 2)
            continue

        if r.status_code != 200:
            return "", f"model returned HTTP {r.status_code}: {r.text[:200]}", attempt

        try:
            d = r.json()
            choice = (d.get("choices") or [{}])[0]
            msg = choice.get("message") or {}
            content = (msg.get("content") or "").strip()
            finish = choice.get("finish_reason")
        except Exception as e:  # noqa: BLE001
            last = f"unparseable response: {type(e).__name__}: {e}"
            if attempt < retries:
                await asyncio.sleep(base_sleep * attempt)
            continue

        if content:
            return content, "", attempt

        # 200 but nothing usable. Almost always the reasoning budget.
        last = f"empty content (finish_reason={finish}, max_tokens={max_tokens})"
        logger.info("distill: attempt %d/%d empty content, finish=%s, max_tokens=%d",
                    attempt, retries, finish, max_tokens)
        if max_tokens < ceiling:
            max_tokens = min(max_tokens * 2, ceiling)
        elif attempt < retries:
            await asyncio.sleep(base_sleep * attempt)
            continue
        if attempt >= retries:
            break
        await asyncio.sleep(base_sleep * attempt)

    return "", f"no usable content after {retries} attempts (last: {last})", retries


# ───────────────────────── public entry ─────────────────────────


async def distill(
    text: str,
    *,
    question: str = "",
    url: str = "",
    use_cache: bool = True,
) -> dict:
    """Compress `text`. Always returns a dict; never raises.

    On ANY failure or skip the caller gets the raw text back untouched plus a
    reason, because returning a broken summary is worse than returning nothing.
    """
    text = text or ""
    chars = len(text)

    base: dict[str, Any] = {
        "source_url": url,
        "raw_chars": chars,
        "question": question,
        "raw_available": True,
        "how_to_get_raw": "call the same tool again with distill=false",
    }

    if chars == 0:
        return {**base, "success": False, "distilled": False,
                "error": "nothing to distil: content was empty"}

    _s = settings()
    if chars < _s["threshold_chars"]:
        return {**base, "success": True, "distilled": False,
                "distill_skipped": f"content is {chars} chars, below the "
                                   f"{_s['threshold_chars']} threshold",
                "content": text}

    cfg = resolve()
    if not cfg["ok"]:
        return {**base, "success": True, "distilled": False,
                "distill_skipped": cfg["error"], "fix": cfg.get("fix", ""),
                "content": text}

    # cache key includes the question and the model: a summary for a different
    # question is a different artefact and must not be reused.
    etag = "distill:" + hashlib.sha256(
        f"{cfg['model']}|{question}|{chars}".encode()).hexdigest()[:16]
    if use_cache and url:
        try:
            hit = await cache_mod.get_cached(url, extraction_type=etag)
            if hit and hit.get("content"):
                return {**base, "success": True, "distilled": True,
                        "cached": True, "model": cfg["model"],
                        "content": hit["content"],
                        "note": "Cached distillation. Pass distill=false for the raw text."}
        except Exception as e:  # noqa: BLE001
            logger.debug("distill cache read failed: %s", e)

    # chunk, so a huge page still fits a modest context window
    chunks = [text[i:i + CHUNK_CHARS] for i in range(0, min(len(text), MAX_INPUT_CHARS), CHUNK_CHARS)]
    if len(text) > MAX_INPUT_CHARS:
        base["input_truncated"] = True
        base["input_chars_seen"] = MAX_INPUT_CHARS

    parts: list[str] = []
    for idx, chunk in enumerate(chunks):
        ask = question if question else ""
        if len(chunks) > 1:
            ask = (ask + "\n\n" if ask else "") + (
                f"This is part {idx + 1} of {len(chunks)} of one page. "
                "Summarise only this part.")
        summary, err, _attempts = await _ask(chunk, ask, cfg)
        if err:
            return {**base, "success": True, "distilled": False,
                    "distill_skipped": f"model failed: {err}",
                    "note": "Returning the raw content instead of a partial summary.",
                    "content": text}
        if summary:
            parts.append(summary)

    merged = "\n\n".join(parts).strip()
    if not merged:
        return {**base, "success": True, "distilled": False,
                "distill_skipped": "model returned an empty summary",
                "content": text}

    if len(merged) > MAX_SUMMARY_CHARS:
        merged = merged[:MAX_SUMMARY_CHARS].rstrip() + "\n\n[distillation truncated]"

    g = grounding_check(merged, text)

    if g["ok"]:
        note = ("Distillation of the page, grounded: every number and every quoted "
                "span was found in the source. Not a complete reproduction - pass "
                "distill=false for the full text.")
    else:
        bits = []
        if g["numbers_unverified"]:
            bits.append(f"{len(g['numbers_unverified'])} number(s) not found in the "
                        f"source: {', '.join(g['numbers_unverified'][:5])}")
        if g["quotes_unverified"]:
            bits.append(f"{len(g['quotes_unverified'])} quoted span(s) not verbatim "
                        f"(likely paraphrased): "
                        f"{'; '.join(q[:60] for q in g['quotes_unverified'][:3])}")
        note = ("Distillation, but the grounding check failed on " + " and ".join(bits)
                + ". Do not treat those as quotes from the page; pass distill=false "
                  "before relying on them.")

    if use_cache and url:
        try:
            await cache_mod.set_cached(url, merged, extraction_type=etag,
                                       status=0, title="", ttl=86_400)
        except Exception as e:  # noqa: BLE001
            logger.debug("distill cache write failed: %s", e)

    return {
        **base,
        "success": True,
        "distilled": True,
        "model": cfg["model"],
        "summary_chars": len(merged),
        "compression_ratio": round(len(merged) / max(1, chars), 4),
        "grounded": g["ok"],
        "grounding": g,
        "content": merged,
        "note": note,
    }


# ───────────────────────── selfcheck ─────────────────────────


async def _selfcheck() -> None:
    checks: list[tuple[str, bool]] = []
    src = "The price is $1,299 and the limit is 4096 tokens. \"exact quoted span here\" appears."

    ok = grounding_check("It costs 1299 and caps at 4096 tokens.", src)
    checks.append(("real numbers ground", ok["ok"]))

    bad = grounding_check("It costs 4999 and caps at 8192 tokens.", src)
    checks.append(("invented numbers are caught",
                   "4999" in bad["numbers_unverified"] and "8192" in bad["numbers_unverified"]))

    q_ok = grounding_check('## Quoted\n- "exact quoted span here"', src)
    checks.append(("real quote grounds",
                   q_ok["quotes_unverified"] == [] and q_ok["quotes_checked"] == 1))

    q_bad = grounding_check('## Quoted\n- "completely invented quotation"', src)
    checks.append(("invented quote caught", len(q_bad["quotes_unverified"]) == 1))

    # A quote mark outside the Quoted section is emphasis, not a verbatim claim,
    # and must not raise a false alarm.
    q_else = grounding_check('He called it the "n times m problem" thing.', src)
    checks.append(("emphasis quote not flagged", q_else["quotes_checked"] == 0))

    # Multi-line pairing must not create phantom quotes.
    q_multi = grounding_check('## Quoted\n- "opening quote here\n- other text\n- "closing"', src)
    checks.append(("no phantom cross-line quotes",
                   len(q_multi["quotes_unverified"]) <= 1))

    checks.append(("comma normalisation", _norm_num("1,299") == "1299"))
    checks.append(("threshold is positive", THRESHOLD_CHARS > 0))
    checks.append(("available() reports a key_source",
                   "key_source" in available()))

    r = await distill("x" * 10)
    checks.append(("short content is not distilled", r["distilled"] is False))
    checks.append(("short content returns raw", r["content"] == "x" * 10))

    for name, okv in checks:
        print(f"{'OK' if okv else 'FAIL'}  {name}")
    print(f"distill selfcheck: {sum(1 for _, v in checks if v)}/{len(checks)} OK")


if __name__ == "__main__":
    asyncio.run(_selfcheck())
