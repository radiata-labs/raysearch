from __future__ import annotations

import pytest

from serpsage.app.response import ResultItem
from serpsage.contracts.lifecycle import ClockBase
from serpsage.contracts.services import FetcherBase
from serpsage.core.runtime import Runtime
from serpsage.domain.enrich import Enricher
from serpsage.extract.html_main import MainContentHtmlExtractor
from serpsage.rank.blend import BlendRanker
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry
from serpsage.text.tokenize import tokenize


class FakeClock(ClockBase):
    def now_ms(self) -> int:
        return 0


class FakeFetchResult:
    def __init__(self, url: str, content: bytes, content_type: str) -> None:
        self._url = url
        self._content = content
        self._content_type = content_type

    @property
    def url(self) -> str:
        return self._url

    @property
    def status_code(self) -> int:
        return 200

    @property
    def content_type(self) -> str:
        return self._content_type

    @property
    def content(self) -> bytes:
        return self._content


class FakeFetcher(FetcherBase):
    def __init__(self, *, rt: Runtime, html: bytes) -> None:
        super().__init__(rt=rt)
        self._html = html

    async def afetch(self, *, url: str) -> FakeFetchResult:
        return FakeFetchResult(url, self._html, "text/html")


@pytest.mark.anyio
async def test_enrich_prunes_leading_boilerplate_blocks():
    keyword = "关键内容"
    boiler = """
      <p>萌娘百科欢迎您参与完善本条目☆这里是与虚拟歌姬一起演出的世界。</p>
      <p>编辑前请阅读Wiki入门或条目编辑规范，并查找相关资料。</p>
      <p>本条目中所使用的数据或歌词，其著作权属于相关著作权人，仅以介绍为目的引用。</p>
      <p>此页面中存在需要长期更新的内容及资料列表，现存条目中资料未必是最新。</p>
    """
    main = f"<p>{('这是正文内容 ' + keyword + '。') * 30}</p>"
    html = f"<!doctype html><html><body><article>{boiler}{main}</article></body></html>".encode()

    settings = AppSettings.model_validate(
        {
            "enrich": {
                "enabled": True,
                "extractor": {"kind": "main_content"},
                "chunking": {
                    "target_chars": 200,
                    "overlap_sentences": 0,
                    "min_chunk_chars": 50,
                    "max_chunks": 20,
                    "max_blocks": 50,
                    "max_sentences": 200,
                    "max_sentence_chars": 400,
                },
                "select": {"min_chunk_score": 0.05},
            },
            "overview": {"enabled": False},
        }
    )
    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=FakeClock())
    enricher = Enricher(
        rt=rt,
        fetcher=FakeFetcher(rt=rt, html=html),
        extractor=MainContentHtmlExtractor(rt=rt),
        ranker=BlendRanker(rt=rt),
    )

    prof = settings.get_profile(settings.pipeline.default_profile)
    query = keyword
    out = await enricher.enrich_one(
        result=ResultItem(url="https://x", domain="example.com"),
        query=query,
        query_tokens=tokenize(query),
        profile=prof,
        top_k=2,
    )

    assert out.error is None
    assert out.chunks
    txt0 = out.chunks[0].text
    assert keyword in txt0
    assert "欢迎" not in txt0
    assert "编辑前请阅读" not in txt0
