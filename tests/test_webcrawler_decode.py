from __future__ import annotations

import anyio

from search_core.config import WebFetchConfig
from search_core.crawler import AsyncWebCrawler, WebCrawler


def test_webcrawler_decodes_shift_jis_html_meta_charset():
    html = (
        "<html><head><meta charset=\"shift_jis\"></head>"
        "<body><p>初音ミク 日本語テスト</p></body></html>"
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
        "<body><p>中文内容：测试解码能力</p></body></html>"
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


def test_async_webcrawler_decodes_shift_jis_html_meta_charset():
    html = (
        "<html><head><meta charset=\"shift_jis\"></head>"
        "<body><p>初音ミク 日本語テスト</p></body></html>"
    )
    data = html.encode("shift_jis", errors="strict")

    async def afetch(_url: str) -> bytes:
        return data

    crawler = AsyncWebCrawler(
        fetch_cfg=WebFetchConfig(),
        user_agent="ua",
        afetcher=afetch,
    )

    async def run() -> str:
        out = await crawler.afetch_blocks("https://example.com")
        return "\n".join(out.blocks)

    joined = anyio.run(run)
    assert "初音ミク" in joined


def test_async_webcrawler_decodes_gb18030_html_meta_charset():
    html = (
        "<html><head><meta charset=\"gb18030\"></head>"
        "<body><p>中文内容：测试解码能力</p></body></html>"
    )
    data = html.encode("gb18030", errors="strict")

    async def afetch(_url: str) -> bytes:
        return data

    crawler = AsyncWebCrawler(
        fetch_cfg=WebFetchConfig(),
        user_agent="ua",
        afetcher=afetch,
    )

    async def run() -> str:
        out = await crawler.afetch_blocks("https://example.com")
        return "\n".join(out.blocks)

    joined = anyio.run(run)
    assert "测试" in joined

