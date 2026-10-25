"""Multi-step browser flows: fill-and-submit a form, and follow pagination.

These exist for the two interactive jobs agents most often scripted by hand:

* **form_fill_submit** - put values into several fields, click submit, wait, and
  optionally read the result. Doing that with raw `browser_action` calls means
  a multi-step program and a lot of fragile selectors in the caller.
* **paginate** - click "next" until a page limit, collecting a value from each
  page. The alternative was a loop in a throwaway script.

Both operate on a *session* by default, because state is the whole point: a
form filled in one throwaway tab is useless, and pagination across tabs loses
position. If no session_id is given they create a temporary one and close it
afterwards, so a single-call use still works.

Every step is bounded: a per-step timeout, a page/field cap, and a step budget.
A failure returns what was completed rather than nothing, because a partially
submitted form is important information, not an exception.
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

import session as session_mod

logger = logging.getLogger(__name__)

MAX_FIELDS = 25
MAX_PAGES = 25
DEFAULT_STEP_TIMEOUT = 20.0
DEFAULT_WAIT_MS = 600
MAX_TEXT_CHARS = 50_000


def _err(msg: str, *, kind: str = "", fix: str = "") -> dict:
    out: dict[str, Any] = {"success": False, "error": msg}
    if kind:
        out["errorKind"] = kind
    if fix:
        out["fix"] = fix
    return out


async def _acquire(session_id: str) -> tuple[Any, str, bool, dict | None]:
    """Return (page, sid, owns_session, error)."""
    sid = (session_id or "").strip()
    if sid:
        page, err = await session_mod.get_page(sid)
        if page is None:
            return None, sid, False, err or _err(f"session {sid!r} is not usable")
        return page, sid, False, None
    made = await session_mod.create(label="flow")
    if not made.get("success"):
        return None, "", True, made
    sid = made["session_id"]
    page, err = await session_mod.get_page(sid)
    if page is None:
        await session_mod.close(sid)
        return None, sid, True, err or _err("could not open a temporary session")
    return page, sid, True, None


async def _read(page, selector: str, as_text: bool) -> dict:
    """Read either one element's text or the whole page's text."""
    try:
        if selector:
            html = await page.document_html()
            import lxml.html as _lh
            doc = _lh.fromstring(html)
            nodes = doc.cssselect(selector)
            if not nodes:
                return {"found": False, "value": "", "note": f"selector matched nothing: {selector}"}
            txt = "\n".join(" ".join(n.text_content().split()) for n in nodes)
        else:
            txt = await page.inner_text() if not as_text else await page.inner_text()
        return {"found": True, "value": (txt or "")[:MAX_TEXT_CHARS],
                "chars": len(txt or "")}
    except Exception as e:  # noqa: BLE001
        return {"found": False, "value": "", "note": f"{type(e).__name__}: {e}"}


async def form_fill_submit(
    *,
    fields: list[dict],
    session_id: str = "",
    url: str = "",
    submit_selector: str = "",
    wait_state: str = "networkidle",
    read_selector: str = "",
    timeout: float = DEFAULT_STEP_TIMEOUT,
) -> dict:
    """Fill fields, submit, and optionally read the result.

    `fields` is a list of objects: {"selector": "#email", "value": "a@b.c"}
    optionally with "action": "type" (default) or "click" for checkboxes and
    radio buttons, where only the selector matters.
    """
    fields = fields or []
    if not isinstance(fields, list) or not fields:
        return _err("fields must be a non-empty list of {selector, value}")
    if len(fields) > MAX_FIELDS:
        return _err(f"too many fields (max {MAX_FIELDS})", kind="permanent")
    for i, f in enumerate(fields):
        if not isinstance(f, dict) or not str(f.get("selector", "")).strip():
            return _err(f"fields[{i}] needs a non-empty 'selector'", kind="permanent")

    page, sid, owns, err = await _acquire(session_id)
    if page is None:
        return err

    done: list[dict] = []
    try:
        if url:
            await asyncio.wait_for(page.goto(url, wait_until=wait_state),
                                   timeout=max(5.0, timeout))
        for i, f in enumerate(fields):
            sel = str(f["selector"]).strip()
            act = str(f.get("action") or "type").lower()
            try:
                # Both helpers RETURN False for "element not found"; they do not
                # raise. Checking the return value is the difference between a
                # real fill and a confident no-op.
                # type_text() returns True even when the selector matches
                # nothing, so existence has to be checked separately.
                exists = await asyncio.wait_for(page.query_selector(sel),
                                                timeout=max(2.0, timeout / 2))
                if not exists:
                    ok = False
                elif act == "click":
                    ok = await asyncio.wait_for(page.click(sel),
                                                timeout=max(2.0, timeout / 2))
                else:
                    # type_text APPENDS at the cursor. On a prefilled field that
                    # produces "newvalueoldvalue", so focus and select first.
                    await asyncio.wait_for(
                        page.evaluate(
                            "(s => { const el = document.querySelector(s);"
                            " if (el) { el.focus();"
                            " if (el.select) el.select();"
                            " else if (el.setSelectionRange) try { el.setSelectionRange(0, 99999); } catch(e){} }"
                            " return true; })(%r)" % sel),
                        timeout=max(2.0, timeout / 2))
                    ok = await asyncio.wait_for(
                        page.type_text(str(f.get("value", "")), selector=None),
                        timeout=max(2.0, timeout / 2))
                if not ok:
                    done.append({"selector": sel, "action": act, "ok": False,
                                 "error": "element not found"})
                    return {
                        "success": False, "session_id": sid, "filled": done,
                        "error": f"field {i} ({sel}): element not found, so the form "
                                 f"was NOT submitted",
                        "errorKind": "permanent",
                        "note": "The selector matched nothing. Check it against the "
                                "live page with browser_action(action='html').",
                    }
                entry = {"selector": sel, "action": act, "ok": True}
                unverified = False
                if act != "click":
                    want = str(f.get("value", ""))
                    try:
                        got = await asyncio.wait_for(
                            page.evaluate(
                                "(s => { const el = document.querySelector(s);"
                                " return el ? String(el.value ?? el.textContent ?? '') : null; })(%r)"
                                % sel),
                            timeout=max(2.0, timeout / 2))
                    except Exception:
                        got = None
                    if got is not None:
                        entry["value_verified"] = (got == want)
                        if got != want:
                            entry["value_read_back"] = str(got)[:80]
                            unverified = True
                done.append(entry)
                if unverified:
                    return {
                        "success": False, "session_id": sid, "filled": done,
                        "error": (f"field {i} ({sel}): typed "
                                  f"{want[:60]!r} but the element read back "
                                  f"{str(entry.get('value_read_back'))[:60]!r}, so the "
                                  f"form was NOT submitted"),
                        "errorKind": "permanent",
                        "note": ("The element may be controlled by JS which ignores "
                                 "synthetic keystrokes. Set the value with "
                                 "browser_action(action='evaluate') and submit separately."),
                    }
            except asyncio.TimeoutError:
                done.append({"selector": sel, "action": act, "ok": False,
                             "error": "timeout"})
                return {"success": False, "session_id": sid, "filled": done,
                        "error": f"field {i} ({sel}) timed out, so the form was NOT "
                                 f"submitted", "errorKind": "transient"}
            except Exception as e:  # noqa: BLE001
                done.append({"selector": sel, "action": act, "ok": False,
                             "error": f"{type(e).__name__}: {e}"})
                return {
                    "success": False,
                    "session_id": sid,
                    "filled": done,
                    "error": f"field {i} ({sel}) failed, so the form was NOT submitted",
                    "errorKind": "transient",
                    "note": "Fix the selector and retry; a partial form is still open in the session.",
                }

        before_url, before_text = page.url, ""
        try:
            before_text = await page.inner_text()
        except Exception:
            pass

        if submit_selector:
            try:
                ok = await asyncio.wait_for(page.click(submit_selector),
                                            timeout=max(2.0, timeout / 2))
            except Exception as e:  # noqa: BLE001
                return {
                    "success": False, "session_id": sid, "filled": done,
                    "error": f"submit click failed: {type(e).__name__}: {e}",
                    "errorKind": "transient",
                    "note": "The fields are filled and still in the session; click submit yourself.",
                }
            if not ok:
                return {
                    "success": False, "session_id": sid, "filled": done,
                    "error": f"submit control not found: {submit_selector!r}. The form "
                             f"was NOT submitted.",
                    "errorKind": "permanent",
                    "note": "The fields are filled and still in the session; fix the "
                            "selector and retry, or click submit yourself.",
                }
            try:
                await asyncio.wait_for(page.wait_for_load_state(wait_state),
                                       timeout=max(5.0, timeout))
            except Exception:
                pass  # the click landed; the page just did not settle in time
            await asyncio.sleep(DEFAULT_WAIT_MS / 1000)

        # Did anything actually happen? A submit that changes nothing is
        # suspicious: it usually means a validation error or a dead control.
        effect = "none"
        if submit_selector:
            # Some forms ignore a click on the submit control but honour Enter
            # in the focused field. Try that once before reporting "no effect".
            try:
                quick = ""
                try:
                    quick = await page.inner_text()
                except Exception:
                    pass
                if page.url == before_url and quick == before_text:
                    focused = bool(fields)
                    if focused:
                        await asyncio.wait_for(page.press_key("Enter"),
                                               timeout=max(2.0, timeout / 2))
                        try:
                            await asyncio.wait_for(
                                page.wait_for_load_state(wait_state),
                                timeout=max(5.0, timeout))
                        except Exception:
                            pass
                        await asyncio.sleep(DEFAULT_WAIT_MS / 1000)
            except Exception:
                pass
        if submit_selector:
            # Last resort, and the most reliable one for real HTML forms: a
            # synthetic mouse click does not always fire submission, but
            # requestSubmit() runs validation and the submit handlers exactly
            # as a user's Enter would.
            try:
                quick = ""
                try:
                    quick = await page.inner_text()
                except Exception:
                    pass
                if page.url == before_url and quick == before_text:
                    await asyncio.wait_for(
                        page.evaluate(
                            "(s => { const el = document.querySelector(s);"
                            " if (!el) return false;"
                            " const f = el.form || (el.closest && el.closest('form'));"
                            " if (f) { if (f.requestSubmit) { f.requestSubmit(); return 'requestSubmit'; }"
                            " f.submit(); return 'submit'; }"
                            " if (el.click) { el.click(); return 'click'; }"
                            " return false; })(%r)" % submit_selector),
                        timeout=max(3.0, timeout / 2))
                    try:
                        await asyncio.wait_for(page.wait_for_load_state(wait_state),
                                               timeout=max(8.0, timeout))
                    except Exception:
                        pass
                    await asyncio.sleep(DEFAULT_WAIT_MS / 1000)
            except Exception:
                pass
        if submit_selector:
            try:
                after_text = await page.inner_text()
            except Exception:
                after_text = ""
            if page.url != before_url:
                effect = "navigated"
            elif before_text and after_text and before_text != after_text:
                effect = "content_changed"
        out: dict[str, Any] = {
            "success": True,
            "session_id": sid,
            "filled": done,
            "submitted": bool(submit_selector),
            "effect": effect,
            "final_url": page.url,
        }
        if submit_selector and effect == "none":
            out["note"] = ("Submitted, but neither the URL nor the page text changed. "
                           "That usually means a validation error, a disabled control, "
                           "or a form that submits via JS you should confirm with "
                           "browser_action(action='text').")
        if read_selector or read_selector == "":
            rd = await _read(page, read_selector, as_text=True)
            out["result"] = rd
        return out
    except asyncio.TimeoutError:
        return {"success": False, "session_id": sid, "filled": done,
                "error": "form_fill_submit timed out", "errorKind": "transient"}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "session_id": sid, "filled": done,
                "error": f"{type(e).__name__}: {e}", "errorKind": "transient"}
    finally:
        if owns:
            await session_mod.close(sid)


async def paginate(
    *,
    next_selector: str,
    session_id: str = "",
    url: str = "",
    max_pages: int = 5,
    read_selector: str = "",
    wait_state: str = "networkidle",
    wait_ms: int = DEFAULT_WAIT_MS,
    stop_when_unchanged: bool = True,
    timeout: float = DEFAULT_STEP_TIMEOUT,
) -> dict:
    """Click `next_selector` up to `max_pages` times, collecting a value per page."""
    if not str(next_selector or "").strip():
        return _err("next_selector is required", kind="permanent")
    pages_wanted = max(1, min(int(max_pages or 5), MAX_PAGES))

    page, sid, owns, err = await _acquire(session_id)
    if page is None:
        return err

    collected: list[dict] = []
    stop_reason = "completed"
    try:
        if url:
            await asyncio.wait_for(page.goto(url, wait_until=wait_state),
                                   timeout=max(5.0, timeout))

        for i in range(pages_wanted):
            rd = await _read(page, read_selector, as_text=True)
            collected.append({"page": i + 1, "url": page.url,
                              "found": rd.get("found", False),
                              "value": rd.get("value", "")})
            if i == pages_wanted - 1:
                stop_reason = f"max_pages cap ({pages_wanted}) reached"
                break
            url_before = page.url
            try:
                # Returns False (does not raise) when the selector matches nothing.
                ok = await asyncio.wait_for(page.click(next_selector),
                                            timeout=max(2.0, timeout / 2))
            except Exception as e:  # noqa: BLE001
                stop_reason = f"next control not clickable: {type(e).__name__}: {e}"
                break
            if not ok:
                stop_reason = (f"next control not found: {next_selector!r} matched "
                               f"nothing, so no further pages were read")
                break
            try:
                await asyncio.wait_for(page.wait_for_load_state(wait_state),
                                       timeout=max(5.0, timeout))
            except Exception:
                pass
            await asyncio.sleep(max(0, min(wait_ms, 10_000)) / 1000)

            if stop_when_unchanged:
                prev = (collected[-1].get("value") or "")[:400]
                cur = (await _read(page, read_selector, as_text=True)).get("value", "")[:400]
                if not (page.url != url_before or cur != prev):
                    # Neither the URL nor the collected value moved: the click was
                    # accepted but had no effect. Reporting "cap reached" here
                    # would be a false success.
                    collected.append({"page": i + 2, "url": page.url,
                                      "found": bool(cur), "value": cur})
                    stop_reason = ("next control had no effect: URL and content were "
                                   "unchanged after clicking, so pagination stopped")
                    break

        return {
            "success": True,
            "session_id": sid,
            "pages": collected,
            "page_count": len(collected),
            "stop_reason": stop_reason,
            "truncated": stop_reason != "completed",
        }
    except asyncio.TimeoutError:
        return {"success": False, "session_id": sid, "pages": collected,
                "error": "paginate timed out", "errorKind": "transient"}
    except Exception as e:  # noqa: BLE001
        return {"success": False, "session_id": sid, "pages": collected,
                "error": f"{type(e).__name__}: {e}", "errorKind": "transient"}
    finally:
        if owns:
            await session_mod.close(sid)


# ───────────────────────── selfcheck ─────────────────────────


async def _selfcheck() -> None:
    checks: list[tuple[str, bool]] = []
    checks.append(("empty fields rejected",
                   (await form_fill_submit(fields=[]))["success"] is False))
    checks.append(("field without selector rejected",
                   (await form_fill_submit(fields=[{"value": "x"}]))["success"] is False))
    checks.append(("missing next_selector rejected",
                   (await paginate(next_selector=""))["success"] is False))
    checks.append(("caps are sane", 1 <= MAX_PAGES <= 100 and 1 <= MAX_FIELDS <= 100))
    for name, okv in checks:
        print(f"{'OK' if okv else 'FAIL'}  {name}")
    print(f"browser_flows selfcheck: {sum(1 for _, v in checks if v)}/{len(checks)} OK")


if __name__ == "__main__":
    asyncio.run(_selfcheck())
