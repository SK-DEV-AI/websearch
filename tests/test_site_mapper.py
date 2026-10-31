import asyncio
import site_mapper


class _Resp:
    def __init__(self, status=404):
        self.status_code = status
        self.text = ""
        self.headers = {}
        self.content = b""


async def _fake_safe_fetch(c, url, timeout=10):
    return _Resp(404)


async def _fake_fetch_page_html(url, c):
    return '<html><body>' + ''.join(f'<a href="/p{i}">p{i}</a>' for i in range(5)) + '</body></html>'


def test_map_site_truncated_flag(monkeypatch):
    monkeypatch.setattr(site_mapper, "safe_fetch", _fake_safe_fetch)
    monkeypatch.setattr(site_mapper, "_fetch_page_html", _fake_fetch_page_html)
    monkeypatch.setattr(site_mapper, "get_http_client", lambda: None)
    res = asyncio.run(site_mapper.map_site("http://example.com", max_urls=2, include_sitemap=False, include_links=True))
    assert res["truncated"] is True
    assert len(res["urls"]) == 2


def test_map_site_not_truncated(monkeypatch):
    monkeypatch.setattr(site_mapper, "safe_fetch", _fake_safe_fetch)
    monkeypatch.setattr(site_mapper, "_fetch_page_html", _fake_fetch_page_html)
    monkeypatch.setattr(site_mapper, "get_http_client", lambda: None)
    res = asyncio.run(site_mapper.map_site("http://example.com", max_urls=10, include_sitemap=False, include_links=True))
    assert res["truncated"] is False
