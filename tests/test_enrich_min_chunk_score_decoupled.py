from __future__ import annotations

import pytest

from serpsage.app.response import ResultItem
from serpsage.app.runtime import CoreRuntime
from serpsage.domain.enrich import Enricher
from serpsage.extract.html_main import MainContentHtmlExtractor
from serpsage.rank.blend import BlendRanker
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry
from serpsage.text.tokenize import tokenize


class FakeClock:
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


class FakeFetcher:
    def __init__(self, html: bytes) -> None:
        self._html = html

    async def afetch(self, *, url: str) -> FakeFetchResult:
        return FakeFetchResult(url, self._html, "text/html")


@pytest.mark.anyio
async def test_enrich_min_chunk_score_is_not_coupled_to_pipeline_min_score():
    html = b"""<!doctype html>
    <html><body>
      <article>
        <p>python alpha sentence one. python alpha sentence one. python alpha sentence one.</p>
        <p>python beta sentence two. python beta sentence two. python beta sentence two.</p>
        <p>python gamma sentence three. python gamma sentence three. python gamma sentence three.</p>
      </article>
    </body></html>"""

    settings = AppSettings.model_validate(
        {
            "pipeline": {"min_score": 0.95},
            "enrich": {
                "enabled": True,
                "extractor": {"kind": "main_content"},
                "chunking": {
                    "target_chars": 90,
                    "overlap_sentences": 0,
                    "min_chunk_chars": 20,
                    "max_chunks": 20,
                    "max_blocks": 50,
                    "max_sentences": 200,
                    "max_sentence_chars": 300,
                },
                "select": {
                    "min_chunk_score": 0.10,
                    "score_soft_gate_tau": 0.0,
                },
            },
            "overview": {"enabled": False},
        }
    )
    rt = CoreRuntime(settings=settings, telemetry=NoopTelemetry(), clock=FakeClock())
    enricher = Enricher(
        rt=rt,
        fetcher=FakeFetcher(html),
        extractor=MainContentHtmlExtractor(rt=rt),
        ranker=BlendRanker(rt=rt),
    )

    prof = settings.get_profile(settings.pipeline.default_profile)
    query = "python"
    out = await enricher.enrich_one(
        result=ResultItem(url="https://x", domain="example.com"),
        query=query,
        query_tokens=tokenize(query),
        profile=prof,
        top_k=5,
    )

    # Even with a very high pipeline.min_score, enrich should still be able to keep chunks
    # based on enrich.select.min_chunk_score.
    assert out.error is None
    assert len(out.chunks) >= 2

