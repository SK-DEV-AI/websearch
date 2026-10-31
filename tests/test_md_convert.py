import md_convert


def test_strip_link_runs_removes_long_run():
    md = "intro\n" + "\n".join(f"- [l{i}](http://x/{i})" for i in range(7)) + "\noutro"
    out = md_convert.strip_link_runs(md)
    assert "l0" not in out
    assert "intro" in out
    assert "outro" in out


def test_strip_link_runs_keeps_short_run():
    md = "intro\n" + "\n".join(f"- [l{i}](http://x/{i})" for i in range(3)) + "\noutro"
    out = md_convert.strip_link_runs(md)
    assert "l0" in out


def test_ensure_links_caps_appended_list():
    # extraction kept 0 links, harvest 100 short? use long labels to avoid skip
    links_html = "".join(f'<a href="http://x/{i}">thread title number {i}</a>' for i in range(100))
    md = "content with no links"
    out = md_convert._ensure_links(md, links_html, "http://example.com/")
    # capped to 30 appended
    appended = out.split("## Page links", 1)[-1] if "## Page links" in out else ""
    assert appended.count("http://x/") <= 30


def test_ensure_links_skips_nav_dump():
    links_html = "".join(f'<a href="http://x/{i}">home</a>' for i in range(60))
    md = "content with no links"
    out = md_convert._ensure_links(md, links_html, "http://example.com/")
    assert "## Page links" not in out
