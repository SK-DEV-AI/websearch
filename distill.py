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
import contextvars
import hashlib
import json
import logging
import os
import re
import time
from typing import Any

import httpx

import cache as cache_mod
import prompts

logger = logging.getLogger(__name__)

# Below this, distilling is not worth the latency or the risk of loss.
THRESHOLD_CHARS = 4000
# Above this, the source is chunked rather than sent in one request.
MAX_INPUT_CHARS = 200_000
CHUNK_CHARS = 60_000
MAX_SUMMARY_CHARS = 6000
TIMEOUT_SECONDS = 90
# No completion budget by default: the endpoint decides. A cap here was a guess
# about how much reasoning a model needs, and a wrong guess cost a retry and a
# truncated summary. DISTILL_MAX_TOKENS still allows an explicit cap.
MAX_TOKENS = None
MAX_TOKENS_CEILING = 12_000
RETRIES = 3
RETRY_BASE_SLEEP = 1.5
REASONING_EFFORT = "low"

# Whether the model may be used at all for THIS call. Default off: a model call
# is the expensive, lossy part of a tool that otherwise returns real data, and a
# swarm reading raw results pays less than one reading a summary. Callers opt in
# per call with use_model=true. A contextvar keeps two concurrent requests from
# flipping each other's policy.
USE_MODEL: contextvars.ContextVar[bool] = contextvars.ContextVar(
    "websearch_use_model", default=False)


def set_use_model(allow: bool) -> None:
    """Set the model policy for the current request context."""
    USE_MODEL.set(bool(allow))


def use_model() -> bool:
    """Whether the model may run for the current request context."""
    return bool(USE_MODEL.get())


# Which gateway THIS call asked for. Empty means "the env order". Naming one
# ("local", "proxy") pins the call to it: an explicit choice is a requirement,
# so a failing named gateway reports its failure instead of quietly spending a
# different route. It is a per-call override, never global state.
PREFER: contextvars.ContextVar[str] = contextvars.ContextVar(
    "websearch_model_prefer", default="")


def set_prefer(name: str) -> None:
    PREFER.set((name or "").strip().lower())


def prefer() -> str:
    return PREFER.get()

DEFAULT_BASE_URL = "http://127.0.0.1:11435/v1"
DEFAULT_MODEL = "reflection/beam:free"
_JCODE_CONFIG = os.path.expanduser("~/.jcode/config.toml")

# Beam is served by two local gateways. The cmdcode2api one has a daily request
# quota and answers 429 for the rest of the day once it is spent; the opencode
# proxy serves the same model without that quota. They are the same model, so
# the proxy belongs in the in-tool lane, not in the keyed-cloud fallback lane.
PROXY_BASE_URL = "http://127.0.0.1:8787/v1"
PROXY_MODEL = "beam-free"
PROXY_DEFAULT_KEY = "local-dev-key"  # the local proxy's dev key

# An endpoint that rate-limits us is retired for a while instead of costing a
# round trip on every call. 0 disables retirement.
COOLDOWN_SECONDS = 600.0
_COOLDOWN: dict[str, float] = {}


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
    DISTILL_MAX_TOKENS          unset    completion budget; unset = endpoint decides
    DISTILL_MAX_TOKENS_CEILING   12000   ceiling when escalating (512-64000)
    DISTILL_TIMEOUT                90    per-request seconds (5-600)
    DISTILL_THRESHOLD_CHARS      4000    below this, skip distillation
    DISTILL_REASONING_EFFORT      low    reasoning_effort passed to the model
    """
    raw_budget = str(os.environ.get("DISTILL_MAX_TOKENS") or "").strip()
    initial = _env_int("DISTILL_MAX_TOKENS", 0, 256, 32_000) if raw_budget else None
    ceiling = _env_int("DISTILL_MAX_TOKENS_CEILING", MAX_TOKENS_CEILING,
                       initial or 512, 64_000)
    return {
        "retries": _env_int("DISTILL_RETRIES", RETRIES, 1, 10),
        "max_tokens": initial,
        "max_tokens_ceiling": ceiling,
        "timeout": _env_int("DISTILL_TIMEOUT", TIMEOUT_SECONDS, 5, 600),
        "threshold_chars": _env_int("DISTILL_THRESHOLD_CHARS", THRESHOLD_CHARS,
                                    500, 200_000),
        "reasoning_effort": (os.environ.get("DISTILL_REASONING_EFFORT")
                             or REASONING_EFFORT).strip().lower(),
        "retry_base_sleep": RETRY_BASE_SLEEP,
    }

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
            "ok": False, "provider": "local", "base_url": base_url,
            "model": model, "api_key": "",
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
            "provider": "local", "api_key": key, "source": source}


def proxy_endpoint() -> dict:
    """Beam behind the local opencode proxy, or {} when it is switched off.

    DISTILL_PROXY=off disables it. The proxy is local and needs no account, so
    its default key is a placeholder; DISTILL_PROXY_API_KEY overrides it.
    """
    if (os.environ.get("DISTILL_PROXY") or "on").strip().lower() in (
            "off", "none", "0", "false", "no"):
        return {}
    return {
        "provider": "proxy",
        "base_url": (os.environ.get("DISTILL_PROXY_BASE_URL")
                     or PROXY_BASE_URL).rstrip("/"),
        "model": os.environ.get("DISTILL_PROXY_MODEL") or PROXY_MODEL,
        "api_key": ((os.environ.get("DISTILL_PROXY_API_KEY") or "").strip()
                    or PROXY_DEFAULT_KEY),
        "ok": True,
    }


def _ep_id(ep: dict) -> str:
    return f"{ep.get('base_url', '')}|{ep.get('model', '')}"


def _cooldown_seconds() -> float:
    raw = os.environ.get("DISTILL_RATE_LIMIT_COOLDOWN", "")
    try:
        return max(0.0, float(raw)) if str(raw).strip() else COOLDOWN_SECONDS
    except (TypeError, ValueError):
        return COOLDOWN_SECONDS


def mark_rate_limited(ep: dict) -> None:
    """Retire an endpoint after it answers 429."""
    secs = _cooldown_seconds()
    if secs > 0:
        _COOLDOWN[_ep_id(ep)] = time.monotonic() + secs


def cooled_down(ep: dict) -> bool:
    until = _COOLDOWN.get(_ep_id(ep), 0.0)
    if until and time.monotonic() < until:
        return True
    if until:
        _COOLDOWN.pop(_ep_id(ep), None)
    return False


def reset_cooldowns() -> None:
    """Forget every cooldown (tests, and an explicit re-probe)."""
    _COOLDOWN.clear()


def endpoints() -> list[dict]:
    """In-tool-model endpoints, best first: the cmdcode2api gateway, then the
    proxy. Both serve Beam; only the gateway differs.

    DISTILL_ORDER: comma-separated subset of "local,proxy" (default that order).
    A gateway that recently answered 429 is moved to the back, and left out
    when the other one is still fresh - a spent daily quota is not worth a
    round trip on every call. If both are cooling down both are returned, so a
    mistimed cooldown cannot silence the model.

    A per-call preference (the `model` argument, see set_prefer) wins over the
    order and over cooldown: the caller named a gateway, so it gets that
    gateway and its error if it fails.
    """
    chain = {"local": resolve(), "proxy": proxy_endpoint()}
    want = prefer()
    if want in chain and chain[want].get("ok"):
        return [chain[want]]
    wanted = [n.strip().lower() for n in
              (os.environ.get("DISTILL_ORDER") or "").split(",") if n.strip()]
    order = wanted or ["local", "proxy"]
    eps = [chain[n] for n in order if chain.get(n) and chain[n].get("ok")]
    fresh = [e for e in eps if not cooled_down(e)]
    if fresh:
        eps = fresh + [e for e in eps if cooled_down(e)]
    return eps


def available() -> dict:
    r = resolve()
    s = settings()
    return {
        "available": r["ok"],
        "base_url": r["base_url"],
        "model": r["model"],
        "key_source": r["source"],
        "endpoints": [{k: e[k] for k in ("provider", "base_url", "model")}
                      for e in endpoints()],
        "prompt_version": prompts.DISTILL_VERSION,
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


async def _ask(text: str, question: str, cfgs: list[dict],
               *, max_tokens: int | None = None,
               part: int = 0, total: int = 0) -> tuple[str, str, int, str]:
    """Ask the model, walking the endpoint chain. Returns
    (summary, error, attempts, answering_provider).

    Both gateways serve the same model, so a failure on one is not a failure of
    the lane: the next gateway gets the same request. An endpoint that answers
    429 is retired for a while afterwards, because a spent daily quota would
    otherwise cost a wasted round trip on every call until it resets.
    """
    if isinstance(cfgs, dict):
        cfgs = [cfgs]
    errors: list[str] = []
    total_attempts = 0
    for cfg in cfgs:
        summary, err, attempts = await _ask_one(
            text, question, cfg, max_tokens=max_tokens, part=part, total=total)
        total_attempts += attempts
        if summary:
            return summary, "", total_attempts, cfg.get("provider", "")
        errors.append(f"{cfg.get('provider', 'model')}: {err}")
        if "429" in err:
            mark_rate_limited(cfg)
    return "", " | ".join(errors) or "no endpoint tried", total_attempts, ""


async def _ask_one(text: str, question: str, cfg: dict,
                   *, max_tokens: int | None = None,
                   part: int = 0, total: int = 0) -> tuple[str, str, int]:
    """One endpoint, with retries. Returns (summary, error, attempts).

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
    max_tokens = int(max_tokens or s["max_tokens"] or 0) or None
    effort = s["reasoning_effort"]
    system = prompts.distill_system(question)
    headers = {"Authorization": f"Bearer {cfg['api_key']}",
               "Content-Type": "application/json"}
    last = "no attempt made"
    send_effort = bool(effort)
    # The opencode proxy advertises its own tool set no matter what we send, and
    # a model holding a `bash` tool would rather curl the page than summarise
    # text that is already in the prompt. `tool_choice: none` is the only
    # payload-level way to forbid that; a gateway that rejects the field gets
    # one retry without it.
    send_no_tools = True
    for attempt in range(1, retries + 1):
        payload = {
            "model": cfg["model"],
            "temperature": 0,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": prompts.distill_user(text, part, total)},
            ],
        }
        if max_tokens:
            payload["max_tokens"] = max_tokens
        if send_effort:
            payload["reasoning_effort"] = effort
        if send_no_tools:
            payload["tool_choice"] = "none"
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
            # 429 is usually a spent quota, not a hiccup: one try here, then
            # let _ask hand the same request to the next gateway.
            if r.status_code == 429 and attempt >= 2:
                break
            if attempt < retries:
                await asyncio.sleep(base_sleep * attempt * 2)
            continue

        if r.status_code == 400 and send_effort and "reasoning" in r.text.lower():
            # A gateway that does not know the field must not cost us the lane.
            send_effort = False
            last = f"HTTP 400 on reasoning_effort: {r.text[:120]}"
            logger.info("distill: %s rejected reasoning_effort, retrying without it",
                        cfg.get("provider"))
            continue

        if r.status_code == 400 and send_no_tools:
            send_no_tools = False
            last = f"HTTP 400 on tool_choice: {r.text[:120]}"
            logger.info("distill: %s rejected tool_choice, retrying without it",
                        cfg.get("provider"))
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

        # 200 but nothing usable. With an explicit budget, almost always the
        # reasoning preamble spending it; without one, there is nothing left to
        # raise, so retrying means paying for the same empty answer twice.
        last = (f"empty content (finish_reason={finish}, max_tokens="
                f"{max_tokens or 'unset'})")
        logger.info("distill: attempt %d/%d empty content, finish=%s, max_tokens=%s",
                    attempt, retries, finish, max_tokens or "unset")
        if max_tokens and max_tokens < ceiling:
            max_tokens = min(max_tokens * 2, ceiling)
            continue
        if attempt >= retries or not max_tokens:
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

    if not use_model():
        # Model off by default: raw results cost the caller nothing to trust and
        # nothing to verify, which is what a swarm wants unless it asked for a
        # summary.
        return {**base, "success": True, "distilled": False,
                "distill_skipped": "model disabled for this call; pass "
                                   "use_model=true to let the model compress it",
                "content": text}

    if chars == 0:
        return {**base, "success": False, "distilled": False,
                "error": "nothing to distil: content was empty"}

    _s = settings()
    if chars < _s["threshold_chars"]:
        return {**base, "success": True, "distilled": False,
                "distill_skipped": f"content is {chars} chars, below the "
                                   f"{_s['threshold_chars']} threshold",
                "content": text}

    # Any gateway serving the in-tool model will do: they are the same model,
    # so the first one that answers is the answer.
    eps = endpoints()
    if not eps:
        cfg = resolve()
        return {**base, "success": True, "distilled": False,
                "distill_skipped": cfg["error"], "fix": cfg.get("fix", ""),
                "content": text}
    cfg = eps[0]

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
    served_by = ""
    for idx, chunk in enumerate(chunks):
        # Part position goes into the user message, not the question: appending
        # it to the question made the model read "part 2 of 3" as part of what
        # the caller was asking for.
        summary, err, _attempts, by = await _ask(chunk, question, eps,
                                                 part=idx + 1, total=len(chunks))
        if err:
            return {**base, "success": True, "distilled": False,
                    "distill_skipped": f"model failed: {err}",
                    "note": "Returning the raw content instead of a partial summary.",
                    "content": text}
        if summary:
            parts.append(summary)
            served_by = served_by or by

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
        "provider": served_by or cfg.get("provider", ""),
        "summary_chars": len(merged),
        "compression_ratio": round(len(merged) / max(1, chars), 4),
        "grounded": g["ok"],
        "grounding": g,
        "prompt_version": prompts.DISTILL_VERSION,
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

    # ── the gateway chain ──────────────────────────────────────
    # Two gateways serve the same model, so one being out of quota must not stop
    # summarisation - and the spent one must not be paid for again.
    saved_env = {k: os.environ.get(k) for k in
                 ("DISTILL_PROXY", "DISTILL_ORDER",
                  "DISTILL_RATE_LIMIT_COOLDOWN")}
    orig_one = globals()["_ask_one"]
    try:
        for k in saved_env:
            os.environ.pop(k, None)
        reset_cooldowns()
        checks.append(("both in-tool gateways are offered",
                       [e["provider"] for e in endpoints()] == ["local", "proxy"]))

        # ── a per-call pin beats the order and the cooldown ──────
        reset_cooldowns()
        set_prefer("proxy")
        checks.append(("a named gateway is the only one used",
                       [e["provider"] for e in endpoints()] == ["proxy"]))
        set_prefer("")
        checks.append(("no pin restores the configured order",
                       [e["provider"] for e in endpoints()] == ["local", "proxy"]))

        async def fake_one(text, question, cfg, **_kw):
            if cfg["provider"] == "local":
                return "", "no usable content after 3 attempts (last: HTTP 429)", 3
            return "summary from the proxy", "", 1

        globals()["_ask_one"] = fake_one
        summary, err, _att, by = await _ask("t", "q", endpoints())
        checks.append(("a 429'd gateway falls through to the other",
                       summary == "summary from the proxy" and not err))
        checks.append(("the answering gateway is named", by == "proxy"))
        checks.append(("the spent gateway is retired",
                       [e["provider"] for e in endpoints()] == ["proxy", "local"]))

        async def both_fail(text, question, cfg, **_kw):
            return "", "HTTP 429", 1

        globals()["_ask_one"] = both_fail
        reset_cooldowns()
        summary, err, _att, by = await _ask("t", "q", endpoints())
        checks.append(("a total failure names both gateways",
                       not summary and "local:" in err and "proxy:" in err))
        checks.append(("a total failure names no gateway", by == ""))
    finally:
        globals()["_ask_one"] = orig_one
        reset_cooldowns()
        for k, v in saved_env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    for name, okv in checks:
        print(f"{'OK' if okv else 'FAIL'}  {name}")
    print(f"distill selfcheck: {sum(1 for _, v in checks if v)}/{len(checks)} OK")


if __name__ == "__main__":
    asyncio.run(_selfcheck())
