import asyncio
import extract
import fetch
import llm


async def _fake_fetch(url, **kw):
    fmt = kw.get("output_format", "html")
    if fmt == "html":
        return {"success": True, "content": "<html><body><p>hello</p></body></html>"}
    return {"success": True, "content": "hello world"}


async def _no_groq():
    return None


def test_llm_empty_result_not_success(monkeypatch):
    monkeypatch.setattr(fetch, "fetch_url", _fake_fetch)
    async def fake_chat(*a, **k):
        return "", "", "no endpoint", 0
    monkeypatch.setattr(llm, "chat", fake_chat)
    # The Groq rotator reads its own copy of the env at import time, so
    # setenv() does not disable the legacy fallback — patch the key lookup
    # itself, otherwise this test makes a real network call.
    monkeypatch.setattr(extract, "_groq_key", _no_groq)
    res = asyncio.run(extract.extract_content("http://example.com", strategy="llm"))
    assert res.get("success") is False
    assert "data" not in res
    assert res.get("error")


def test_llm_nonempty_result_success(monkeypatch):
    monkeypatch.setattr(fetch, "fetch_url", _fake_fetch)
    async def fake_chat(*a, **k):
        return '{"items": [1,2,3]}', "stop", "", 1
    monkeypatch.setattr(llm, "chat", fake_chat)
    monkeypatch.setattr(extract, "_groq_key", _no_groq)
    res = asyncio.run(extract.extract_content("http://example.com", strategy="llm"))
    assert res.get("success") is True
    assert res.get("data") == {"items": [1, 2, 3]}
