import asyncio
import wikipedia


class _Resp:
    status_code = 200

    def json(self):
        return {
            "title": "T", "pageid": 1,
            "content_urls": {"desktop": {"page": "https://en.wikipedia.org/wiki/T"}},
            "extract": "hello world",
            "extract_html": "<p>hello world</p>",
            "thumbnail": {"source": "http://img"},
            "description": "desc",
        }


class _Client:
    async def get(self, *a, **k):
        return _Resp()


def test_wikipedia_rest_no_duplicate_html(monkeypatch):
    monkeypatch.setattr(wikipedia, "get_http_client", lambda: _Client())
    res = asyncio.run(wikipedia.fetch_wikipedia_summary_rest("T"))
    assert res is not None
    assert "extract_html" not in res
    assert res["extract"] == "hello world"
