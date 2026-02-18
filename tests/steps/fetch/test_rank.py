from __future__ import annotations

import anyio

from serpsage.app.bootstrap import build_runtime
from serpsage.app.request import (
    FetchAbstractsRequest,
    FetchOverviewRequest,
    FetchRequest,
)
from serpsage.components.rank.base import RankerBase
from serpsage.models.extract import ExtractedDocument
from serpsage.models.pipeline import (
    FetchStepContext,
    FetchStepOthers,
    PreparedAbstract,
)
from serpsage.settings.models import AppSettings
from serpsage.steps.fetch.rank import FetchAbstractRankStep


class _DummyRanker(RankerBase):
    def __init__(self, *, rt) -> None:
        super().__init__(rt=rt)
        self.queries: list[str] = []

    async def score_texts(
        self,
        *,
        texts: list[str],
        query: str,
        query_tokens: list[str],
    ) -> list[float]:
        del query_tokens
        self.queries.append(query)
        return [1.0 for _ in texts]


def _build_ctx(*, settings: AppSettings, request: FetchRequest) -> FetchStepContext:
    return FetchStepContext(
        settings=settings,
        request=request,
        url="https://example.com/path",
        url_index=0,
        others=FetchStepOthers(),
        prepared_abstracts=[
            PreparedAbstract(text="alpha", heading="h1", position=0),
            PreparedAbstract(text="beta", heading="h2", position=1),
        ],
        extracted=ExtractedDocument(title="DeepSeek V3.2"),
    )


def test_rank_uses_title_query_for_overview_when_abstracts_disabled() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    ranker = _DummyRanker(rt=rt)
    step = FetchAbstractRankStep(rt=rt, ranker=ranker)
    ctx = _build_ctx(
        settings=settings,
        request=FetchRequest(
            urls=["https://example.com/path"],
            content=False,
            abstracts=False,
            overview=True,
        ),
    )
    ctx.abstracts_request = None
    ctx.overview_request = FetchOverviewRequest()

    out = anyio.run(step.run, ctx)

    assert out.scored_abstracts == []
    assert len(out.overview_scored_abstracts) == 2
    assert "DeepSeek V3.2" in ranker.queries


def test_rank_falls_back_to_url_when_query_and_title_missing() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    ranker = _DummyRanker(rt=rt)
    step = FetchAbstractRankStep(rt=rt, ranker=ranker)
    ctx = _build_ctx(
        settings=settings,
        request=FetchRequest(
            urls=["https://example.com/path"],
            abstracts=True,
        ),
    )
    ctx.extracted = ExtractedDocument(title="")
    ctx.abstracts_request = FetchAbstractsRequest(query=None)
    ctx.overview_request = None

    out = anyio.run(step.run, ctx)

    assert len(out.scored_abstracts) == 2
    assert "https://example.com/path" in ranker.queries
