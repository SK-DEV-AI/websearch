"""One place that decides which model answers, and counts the calls.

Why this module exists: the search path made model calls from three different
places with two different rules. Synthesis and verification preferred the free
in-tool model; query decomposition and expansion used Groq directly and gave up
entirely without a Groq key. So "use the free model" was true for the part the
user reads and false for the part that decides what to search.

Now every caller goes through `chat()`, which:

* prefers the in-tool model and falls back to Groq (configurable with
  WEBSEARCH_SYNTH_PROVIDER, same knob as synthesis),
* retries transient failures and escalates the token budget when a reasoning
  model returns empty content,
* counts the calls it makes, so a search response can report what it cost.

The counter exists because "how many model calls did that cost me?" was
unanswerable from the outside, and it is the first question anyone asks about a
tool that uses a model.
"""

from __future__ import annotations

import asyncio
import contextvars
import logging
import os
import sys
from typing import Any

import httpx

logger = logging.getLogger(__name__)

GROQ_BASE_URL = "https://api.groq.com/openai/v1"

# A reasoning model emits reasoning_content BEFORE content, so a budget sized for
# the answer alone comes back as 200 + empty content. The helpers ask for 64
# tokens (a label, a one-line rewrite), which a reasoning model spends before it
# says anything: classify_need then returned its "general" fallback and looked
# like a confident classification. Raise every request to a floor that survives
# the reasoning preamble, and keep the caller's intent as a lower bound only.
MIN_USEFUL_TOKENS = int(os.environ.get("WEBSEARCH_MIN_MODEL_TOKENS") or 512)

# Reasoning depth asked of the model. Beam spends reasoning tokens before it
# emits content; "low" keeps that preamble short, which is the difference
# between a one-line answer and a 2000-token essay for the same question.
REASONING_EFFORT = (os.environ.get("WEBSEARCH_REASONING_EFFORT") or "low").strip().lower()

# Per-request counter. The question "how many model calls did that search
# cost me?" was unanswerable from outside, and it is the first question anyone
# asks about a tool that spends a model on every call. A contextvar keeps
# concurrent requests from attributing each other's calls, which a module-level
# dict would do.
_COUNTS: contextvars.ContextVar[dict | None] = contextvars.ContextVar(
    "llm_counts", default=None)


def begin() -> None:
    """Start a fresh count for the current request context."""
    _COUNTS.set({})


def set_use_model(on: bool) -> None:
    """Allow or forbid model calls for the current request context.

    Default is OFF: a tool that returns real data is worth more to a caller than
    a model's paraphrase of it, and it costs nothing. Callers opt in per call.
    """
    try:
        import distill
        distill.set_use_model(on)
    except Exception:  # noqa: BLE001
        pass


def use_model() -> bool:
    try:
        import distill
        return distill.use_model()
    except Exception:  # noqa: BLE001
        return False


def set_prefer(name: str) -> None:
    """Pin this request to one gateway or lane: local, proxy, groq (or "")."""
    try:
        import distill
        distill.set_prefer(name)
    except Exception:  # noqa: BLE001
        pass


def prefer_lane() -> str:
    try:
        import distill
        return distill.prefer()
    except Exception:  # noqa: BLE001
        return ""


def counts() -> dict[str, int]:
    d = _COUNTS.get()
    return dict(d) if d else {}


def _bump(kind: str) -> None:
    d = _COUNTS.get()
    if d is None:
        return
    d[kind] = d.get(kind, 0) + 1


def groq_keys() -> list[str]:
    raw = os.environ.get("GROQ_API_KEYS", "")
    return [k.strip() for k in raw.split(",") if k.strip()]


def local_endpoint() -> dict:
    """The in-tool model as an endpoint, or {} when it is not configured."""
    try:
        import distill  # local import: distill imports cache, which is heavier
        d = distill.available()
        if not d.get("available"):
            return {}
        r = distill.resolve()
        return {"provider": "local", "base_url": r["base_url"],
                "model": r["model"], "api_key": r["api_key"], "rotate": False}
    except Exception:  # noqa: BLE001
        return {}


def proxy_endpoint() -> dict:
    """The same Beam model behind the local opencode proxy, or {} when off.

    Same model, different gateway: the cmdcode2api gateway has a daily quota and
    answers 429 once it is spent, the proxy does not. Configuration lives in
    distill (DISTILL_PROXY / DISTILL_PROXY_BASE_URL / _MODEL / _API_KEY).
    """
    try:
        import distill
        ep = distill.proxy_endpoint()
        if not ep:
            return {}
        return {**ep, "rotate": False}
    except Exception:  # noqa: BLE001
        return {}


def _cooled(ep: dict) -> bool:
    try:
        import distill
        return distill.cooled_down(ep)
    except Exception:  # noqa: BLE001
        return False


def _retire(ep: dict) -> None:
    """Retire an endpoint that answered 429 (best effort)."""
    try:
        import distill
        distill.mark_rate_limited(ep)
    except Exception:  # noqa: BLE001
        pass


def endpoints() -> list[dict]:
    """Candidate endpoints, best first.

    The in-tool model wins by default because it is free at the point of use; a
    keyed cloud lane should not be spent on work the local model can do. Groq is
    the fallback for when the local endpoint is down or absent.

    WEBSEARCH_SYNTH_PROVIDER: local (default) | groq | auto
    WEBSEARCH_MODEL_FALLBACK: on (default) | off
      off - never fall through to the keyed cloud lane. "Use the in-tool model"
      then means exactly that: a failing in-tool endpoint reports an error
      instead of silently spending a keyed lane. Set this when the model choice
      is a requirement (cost, privacy, compliance), not a preference.
    DISTILL_ORDER: comma-separated preference, e.g. "proxy,local,groq".

    Beam is reachable through two local gateways (cmdcode2api and the opencode
    proxy). They are the same model, so both count as the in-tool lane; a spent
    daily quota on one must not stop summarisation. An endpoint that just
    answered 429 is moved to the back, and dropped when another is still fresh.
    """
    prefer = (os.environ.get("WEBSEARCH_SYNTH_PROVIDER") or "local").strip().lower()
    keys = groq_keys()
    groq = ({"provider": "groq", "base_url": GROQ_BASE_URL,
             "model": os.environ.get("GROQ_SMART_MODEL") or "openai/gpt-oss-120b",
             "api_key": "", "rotate": True}
            if keys else None)
    local = local_endpoint() or None
    proxy = proxy_endpoint() or None
    chain = {"local": local, "proxy": proxy, "groq": groq}
    # A per-call pin (the `model` argument) is a requirement: that lane only,
    # and its error if it has no keys or fails, rather than a silent substitute.
    pinned = prefer_lane()
    if pinned in chain and chain[pinned]:
        return [chain[pinned]]
    default_order = ["local", "proxy", "groq"]
    if prefer == "groq":
        default_order = ["groq", "local", "proxy"]
    elif prefer == "proxy":
        default_order = ["proxy", "local", "groq"]
    wanted = [n.strip().lower() for n in
              (os.environ.get("DISTILL_ORDER") or "").split(",") if n.strip()]
    names = list(wanted or default_order)
    # A preference reorders the chain, it does not truncate it: anything not
    # named keeps its usual place at the end.
    names += [n for n in ("local", "proxy", "groq") if n not in names]
    eps = [chain[n] for n in names if chain.get(n)]
    fresh = [e for e in eps if not _cooled(e)]
    if fresh:
        eps = fresh + [e for e in eps if _cooled(e)]
    strict = (os.environ.get("WEBSEARCH_MODEL_FALLBACK") or "on").strip().lower()
    if strict in ("off", "none", "0", "false", "no"):
        if prefer == "groq":
            return eps[:1]
        return [e for e in eps if e.get("provider") != "groq"]
    return eps


def _next_groq_key(offset: int = 0) -> str:
    keys = groq_keys()
    if not keys:
        return ""
    return keys[offset % len(keys)]


async def chat(messages: list[dict], *, max_tokens: int | None = None,
               attempts: int = 2, timeout: float = 30,
               kind: str = "chat", temperature: float = 0.0,
               model: str | None = None,
               client: Any = None) -> tuple[str, str, str, int]:
    """One OpenAI-compatible completion, with fallback across endpoints.

    Returns (text, finish_reason, error, attempts). `kind` labels the call in the
    counter so a response can say where the tokens went.
    """
    if not use_model():
        return "", "", "model disabled for this call; pass use_model=true", 0
    eps = endpoints()
    if not eps:
        return "", "", "no model endpoint configured", 0

    last = "no attempt made"
    total = 0
    for ep in eps:
        if model:
            ep = {**ep, "model": model}
        text, finish, err, att = await _one(ep, messages, max_tokens=max_tokens,
                                           attempts=attempts, timeout=timeout,
                                           temperature=temperature,
                                           client=client)
        total += att
        if text:
            _bump(kind)
            return text, finish, "", total
        last = err
        logger.warning("llm: %s call failed on %s: %s", kind, ep.get("provider"), err)
        if "HTTP 429" in err:
            # A spent quota stays spent: do not pay this round trip again.
            _retire(ep)
    return "", "", last, total


async def _one(ep: dict, messages: list[dict], *, max_tokens: int | None,
               attempts: int, timeout: float,
               temperature: float, client: Any = None) -> tuple[str, str, str, int]:
    url = ep["base_url"].rstrip("/") + "/chat/completions"
    last = "no attempt made"
    # No budget means the endpoint decides: a hand-picked cap was a guess about
    # how much reasoning the model needs, and a wrong guess bought a truncated
    # answer. A caller that does pass one gets a reasoning-surviving floor
    # (MIN_USEFUL_TOKENS) and escalation on empty content.
    budget = max(int(max_tokens), MIN_USEFUL_TOKENS) if max_tokens else None
    ceiling = min(max((budget or 0) * 4, 4096), 32_000)
    effort = REASONING_EFFORT
    send_effort = bool(effort)
    # The proxy advertises its own tools regardless of what we send, and a model
    # holding `bash` answers a question with a tool call instead of prose. Ban
    # it in the payload; a gateway that rejects the field gets one retry without.
    send_no_tools = True
    for attempt in range(1, max(1, attempts) + 1):
        key = (_next_groq_key(attempt - 1) if ep.get("rotate")
               else ep.get("api_key", ""))
        body = {"model": ep["model"], "messages": messages,
                "temperature": temperature}
        if budget:
            body["max_tokens"] = budget
        if send_effort:
            body["reasoning_effort"] = effort
        if send_no_tools:
            body["tool_choice"] = "none"
        try:
            if client is not None:
                resp = await client.post(
                    url,
                    headers={"Authorization": f"Bearer {key}",
                             "Content-Type": "application/json"},
                    json=body,
                    timeout=timeout)
            else:
                async with httpx.AsyncClient(timeout=timeout) as c:
                    resp = await c.post(
                        url,
                        headers={"Authorization": f"Bearer {key}",
                                 "Content-Type": "application/json"},
                        json=body)
        except Exception as e:  # noqa: BLE001
            # httpx.ReadTimeout carries no message, which used to render as
            # "ReadTimeout: " - a dangling colon that teaches the caller nothing.
            detail = str(e).strip()
            last = (f"{type(e).__name__}: {detail}" if detail
                    else f"{type(e).__name__} (no message)")
            continue
        if resp.status_code == 400 and send_effort and "reasoning" in resp.text.lower():
            send_effort = False
            last = f"HTTP 400 on reasoning_effort: {resp.text[:120]}"
            logger.info("llm: %s rejected reasoning_effort, retrying without it",
                        ep.get("provider"))
            continue
        if resp.status_code == 400 and send_no_tools:
            send_no_tools = False
            last = f"HTTP 400 on tool_choice: {resp.text[:120]}"
            logger.info("llm: %s rejected tool_choice, retrying without it",
                        ep.get("provider"))
            continue
        if resp.status_code != 200:
            last = f"HTTP {resp.status_code}: {resp.text[:200]}"
            if resp.status_code in (429, 500, 502, 503, 504):
                # 429 is usually a spent quota: one retry here, then chat()
                # hands the request to the next gateway.
                if resp.status_code == 429 and attempt >= 2:
                    break
                continue
            return "", "", last, attempt
        try:
            data = resp.json()
            choice = (data.get("choices") or [{}])[0]
            text = ((choice.get("message") or {}).get("content") or "").strip()
            finish = choice.get("finish_reason") or ""
        except Exception as e:  # noqa: BLE001
            last = f"unparseable response: {type(e).__name__}: {e}"
            continue
        if text:
            return text, finish, "", attempt
        last = (f"empty content (finish_reason={finish}, max_tokens="
                f"{budget or 'unset'})")
        if budget and budget < ceiling:
            budget = min(budget * 2, ceiling)
            body["max_tokens"] = budget
            logger.info("llm: empty content, raising max_tokens to %d", budget)
    return "", "", last, max(1, attempts)


def _selfcheck() -> None:
    """Prove the parts that are cheap to get wrong: ordering, counting, failure.

    No network: every endpoint is a fake and the transport is stubbed, because a
    selfcheck that needs the internet is a selfcheck that fails offline.
    """
    ok = 0
    bad = []

    def check(name: str, cond: bool) -> None:
        nonlocal ok
        if cond:
            ok += 1
        else:
            bad.append(name)

    # ── endpoint ordering ────────────────────────────────────────
    saved = {k: os.environ.get(k) for k in
             ("WEBSEARCH_SYNTH_PROVIDER", "GROQ_API_KEYS", "DISTILL_API_KEY",
              "WEBSEARCH_MODEL_FALLBACK", "DISTILL_PROXY", "DISTILL_ORDER",
              "DISTILL_RATE_LIMIT_COOLDOWN")}
    # The selfcheck drives the model path on purpose, so it opts in the same way
    # a caller does; the gate itself is proven below.
    set_use_model(True)
    orig_local = globals()["local_endpoint"]
    orig_proxy = globals()["proxy_endpoint"]
    try:
        # Keep the proxy out of the way until its own checks below.
        globals()["proxy_endpoint"] = lambda: {}
        globals()["local_endpoint"] = lambda: {"provider": "local", "base_url": "x",
                                               "model": "m", "api_key": "k",
                                               "rotate": False}
        os.environ.pop("WEBSEARCH_SYNTH_PROVIDER", None)
        os.environ["GROQ_API_KEYS"] = "g1,g2"
        eps = endpoints()
        check("local first by default", [e["provider"] for e in eps] == ["local", "groq"])
        os.environ["WEBSEARCH_SYNTH_PROVIDER"] = "groq"
        eps = endpoints()
        check("provider=groq promotes groq", [e["provider"] for e in eps] == ["groq", "local"])
        os.environ.pop("GROQ_API_KEYS")
        os.environ["WEBSEARCH_SYNTH_PROVIDER"] = "auto"
        eps = endpoints()
        check("auto without groq yields local only", [e["provider"] for e in eps] == ["local"])
        # Strict mode is the difference between "prefers the in-tool model" and
        # "uses only the in-tool model". Both must exist: preference is a good
        # default, a guarantee is what a compliance question needs.
        os.environ["GROQ_API_KEYS"] = "g1"
        globals()["local_endpoint"] = lambda: {"provider": "local", "base_url": "x",
                                               "model": "m", "api_key": "k",
                                               "rotate": False}
        os.environ["WEBSEARCH_MODEL_FALLBACK"] = "off"
        check("strict mode drops the fallback",
              [e["provider"] for e in endpoints()] == ["local"])
        os.environ["WEBSEARCH_SYNTH_PROVIDER"] = "groq"
        check("strict mode keeps only the preferred lane",
              [e["provider"] for e in endpoints()] == ["groq"])
        os.environ["WEBSEARCH_MODEL_FALLBACK"] = "on"
        check("fallback returns when re-enabled",
              len(endpoints()) == 2)
        os.environ.pop("WEBSEARCH_MODEL_FALLBACK", None)

        # The proxy is a second gateway for the same model, so it belongs to the
        # in-tool lane: strict mode must keep it (it spends no keyed quota).
        globals()["proxy_endpoint"] = orig_proxy
        os.environ.pop("WEBSEARCH_SYNTH_PROVIDER", None)
        os.environ.pop("DISTILL_ORDER", None)
        os.environ.pop("DISTILL_PROXY", None)
        check("proxy sits behind local by default",
              [e["provider"] for e in endpoints()] == ["local", "proxy", "groq"])
        os.environ["WEBSEARCH_MODEL_FALLBACK"] = "off"
        check("strict mode keeps both in-tool gateways",
              [e["provider"] for e in endpoints()] == ["local", "proxy"])
        os.environ.pop("WEBSEARCH_MODEL_FALLBACK", None)
        os.environ["DISTILL_ORDER"] = "proxy,local"
        check("DISTILL_ORDER overrides preference",
              [e["provider"] for e in endpoints()] == ["proxy", "local", "groq"])
        os.environ["DISTILL_PROXY"] = "off"
        check("DISTILL_PROXY=off removes it",
              "proxy" not in [e["provider"] for e in endpoints()])
        os.environ.pop("DISTILL_PROXY", None)
        os.environ.pop("DISTILL_ORDER", None)
        proxy_ep = [e for e in endpoints() if e["provider"] == "proxy"][0]
        _retire(proxy_ep)
        check("a 429'd endpoint moves to the back",
              [e["provider"] for e in endpoints()] == ["local", "groq", "proxy"])
        import distill as _distill
        _distill.reset_cooldowns()
        globals()["proxy_endpoint"] = lambda: {}

        os.environ.pop("GROQ_API_KEYS", None)
        globals()["local_endpoint"] = lambda: {}
        check("no endpoint yields empty list", endpoints() == [])

        # ── counting is per context and only on success ──────────
        async def no_ep() -> tuple[str, str, str, int]:
            begin()
            text = await chat([{"role": "user", "content": "x"}])
            return text

        text, _f, err, _a = asyncio.run(no_ep())
        check("no endpoint is reported", text == "" and "no model endpoint" in err)
        check("no endpoint counts nothing",
              asyncio.run(no_ep())[0] == "")

        async def success() -> dict:
            begin()
            _bump("synthesis")
            _bump("synthesis")
            _bump("verify")
            return counts()

        c = asyncio.run(success())
        check("counts label kinds", c == {"synthesis": 2, "verify": 1})
        check("fresh context starts empty", counts() == {})

        # Shared reference across tasks in one request, isolated between them.
        async def two_requests() -> tuple[dict, dict]:
            async def one(kind: str) -> dict:
                begin()
                await asyncio.sleep(0)
                _bump(kind)
                return counts()
            return await asyncio.gather(one("synthesis"), one("verify"))

        a, b = asyncio.run(two_requests())
        check("concurrent requests do not share counts",
              a == {"synthesis": 1} and b == {"verify": 1})

        # ── transport failure is classified, not swallowed ──────
        class FakeResp:
            status_code = 429
            text = "rate limited"

        class FakeClient:
            def __init__(self, resp): self.resp = resp
            async def post(self, *a, **kw): return self.resp

        class RaisingClient:
            async def post(self, *a, **kw): raise OSError("connection reset")

        async def fail_once(client, ep, budget, attempts=1) -> tuple[str, str, str, int]:
            begin()
            return await _one(ep, [{"role": "user", "content": "x"}],
                              max_tokens=budget, attempts=attempts, timeout=1,
                              temperature=0.0, client=client)

        ep = {"provider": "local", "base_url": "http://x", "model": "m",
              "api_key": "k", "rotate": False}
        _t, _f, err, att = asyncio.run(fail_once(FakeClient(FakeResp()), ep, 100))
        check("429 is reported with its status", err.startswith("HTTP 429") and att == 1)
        _t, _f, err, _a = asyncio.run(fail_once(RaisingClient(), ep, 100))
        check("transport error is named", "OSError" in err)

        # ── empty content escalates the budget ───────────────────
        class EmptyThenFull:
            def __init__(self): self.budgets = []
            async def post(self, url, headers=None, json=None, timeout=None):
                self.budgets.append(json["max_tokens"])
                class R:
                    def __init__(self, body): self.status_code = 200; self._b = body
                    def json(self): return self._b
                if len(self.budgets) == 1:
                    return R({"choices": [{"message": {"content": ""},
                                           "finish_reason": "length"}]})
                return R({"choices": [{"message": {"content": "answer"},
                                       "finish_reason": "stop"}]})

        c2 = EmptyThenFull()
        text, _f, err, att = asyncio.run(fail_once(c2, ep, 100, attempts=2))
        check("empty content recovers with a bigger budget", text == "answer")
        check("budget was raised, not reused", c2.budgets[1] > c2.budgets[0])
        check("recovery is counted as one call", counts() == {})
        check("a tiny budget is raised to the usable floor",
              c2.budgets[0] >= MIN_USEFUL_TOKENS)

        # The floor is what stops a 64-token label request from coming back as
        # empty content and being mistaken for the caller's fallback.
        class AlwaysEmpty:
            def __init__(self): self.budgets = []
            async def post(self, url, headers=None, json=None, timeout=None):
                self.budgets.append(json["max_tokens"])
                class R:
                    status_code = 200
                    def json(self): return {"choices": [{"message": {"content": ""},
                                                         "finish_reason": "length"}]}
                return R()

        c4 = AlwaysEmpty()
        text, _f, err, att = asyncio.run(fail_once(c4, ep, 64, attempts=3))
        check("empty content is reported, not returned as success", text == "")
        check("the failure names the budget", "max_tokens" in err)
        check("budget grows on each retry", c4.budgets == sorted(c4.budgets) and c4.budgets[-1] > c4.budgets[0])

        # ── groq key rotation is round-robin, never fixed ────────
        os.environ["GROQ_API_KEYS"] = "a,b,c"
        keys = {_next_groq_key(i) for i in range(3)}
        check("groq keys rotate", keys == {"a", "b", "c"})

        async def counted_ep() -> dict:
            begin()
            p = {"provider": "local", "base_url": "http://x", "model": "m",
                 "api_key": "k", "rotate": False}
            await chat([{"role": "user", "content": "x"}],
                       client=FakeClient(type("R", (), {"status_code": 200,
                                                        "json": lambda self: {"choices": [
                                                            {"message": {"content": "hi"}}]}})()),
                       kind="synthesis")
            return counts()

        globals()["endpoints"] = lambda: [{"provider": "local", "base_url": "http://x",
                                           "model": "m", "api_key": "k", "rotate": False}]
        c3 = asyncio.run(counted_ep())
        check("successful chat bumps its kind", c3 == {"synthesis": 1})

        # ── the model is off unless the call asked for it ────────
        globals().pop("endpoints", None)
        set_use_model(False)
        text, _f, err, _a = asyncio.run(
            chat([{"role": "user", "content": "x"}]))
        check("model is off by default",
              text == "" and "model disabled" in err and "use_model" in err)
        set_use_model(True)
        check("use_model turns it back on", use_model() is True)
    finally:
        set_use_model(False)
        globals()["local_endpoint"] = orig_local
        globals()["proxy_endpoint"] = orig_proxy
        try:
            import distill as _d
            _d.reset_cooldowns()
        except Exception:  # noqa: BLE001
            pass
        globals().pop("endpoints", None)
        for k, v in saved.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    if bad:
        print(f"llm selfcheck: {ok}/{ok + len(bad)} FAILED: {', '.join(bad)}")
        raise SystemExit(1)
    print(f"llm selfcheck: {ok}/{ok} OK")


if __name__ == "__main__":
    if "--selfcheck" in sys.argv:
        _selfcheck()
