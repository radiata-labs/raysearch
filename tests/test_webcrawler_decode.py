from __future__ import annotations

from search_core.config import WebFetchConfig
from search_core.crawler import WebCrawler


def test_webcrawler_decodes_shift_jis_html_meta_charset():
    html = (
        "<html><head><meta charset=\"shift_jis\"></head>"
        "<body><p>初音ミク の新作MV が公開された。</p></body></html>"
    )
    data = html.encode("shift_jis", errors="strict")
    crawler = WebCrawler(
        fetch_cfg=WebFetchConfig(),
        user_agent="ua",
        fetcher=lambda _url: data,
    )
    out = crawler.fetch_blocks("https://example.com")
    joined = "\n".join(out.blocks)
    assert "初音ミク" in joined


def test_webcrawler_decodes_gb18030_html_meta_charset():
    html = (
        "<html><head><meta charset=\"gb18030\"></head>"
        "<body><p>中文内容：测试解码能力。</p></body></html>"
    )
    data = html.encode("gb18030", errors="strict")
    crawler = WebCrawler(
        fetch_cfg=WebFetchConfig(),
        user_agent="ua",
        fetcher=lambda _url: data,
    )
    out = crawler.fetch_blocks("https://example.com")
    joined = "\n".join(out.blocks)
    assert "测试" in joined

