from __future__ import annotations

from bs4 import BeautifulSoup

from serpsage.components.extract.markdown.render import render_markdown


def test_inline_code_inside_links_preserves_spaces() -> None:
    html_doc = (
        "<main>"
        "<p>Run <a href='/docs/start'><code>python  -m   pip</code></a> now.</p>"
        '<p>Use <code>--flag="A  B"</code> to continue.</p>'
        "</main>"
    )
    soup = BeautifulSoup(html_doc, "html.parser")

    markdown, stats = render_markdown(root=soup.main, base_url="https://example.com")

    assert "[`python  -m   pip`](https://example.com/docs/start)" in markdown
    assert '`--flag="A  B"`' in markdown
    assert "python  -m   pip" in markdown
    assert int(stats.get("inline_code_count", 0)) >= 2
    assert int(stats.get("link_count", 0)) >= 1
