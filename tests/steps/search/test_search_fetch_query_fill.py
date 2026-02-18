from __future__ import annotations

import anyio
from typing_extensions import override

from serpsage.app.bootstrap import build_runtime
from serpsage.app.request import SearchRequest
from serpsage.app.response import FetchResultItem
from serpsage.models.pipeline import FetchStepContext, SearchStepContext
from serpsage.settings.models import AppSettings
from serpsage.steps.base import RunnerBase, StepBase
from serpsage.steps.search.fetch import SearchFetchStep
from serpsage.telemetry.base import SpanBase


class _CaptureFetchStep(StepBase[FetchStepContext]):
    def __init__(self, *, rt) -> None:
        super().__init__(rt=rt)
        self.requests = []

    @override
    async def run_inner(
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        del span
        self.requests.append(ctx.request)
        ctx.result = FetchResultItem(
            url=ctx.url,
            title="",
            content="",
            abstracts=[],
            abstract_scores=[],
        )
        return ctx


def test_search_fetch_fills_query_when_abstracts_and_overview_are_true() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    capture = _CaptureFetchStep(rt=rt)
    runner = RunnerBase[FetchStepContext](rt=rt, steps=[capture])
    step = SearchFetchStep(rt=rt, fetch_runner=runner)
    ctx = SearchStepContext(
        settings=settings,
        request=SearchRequest(
            query="what is deepseek",
            fetchs={"content": True, "abstracts": True, "overview": True},
        ),
        candidate_urls=["https://example.com"],
    )

    anyio.run(step.run, ctx)

    req = capture.requests[0]
    assert req.abstracts is not False
    assert req.overview is not False
    assert req.abstracts.query == "what is deepseek"
    assert req.overview.query == "what is deepseek"


def test_search_fetch_fills_query_when_request_query_is_none() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    capture = _CaptureFetchStep(rt=rt)
    runner = RunnerBase[FetchStepContext](rt=rt, steps=[capture])
    step = SearchFetchStep(rt=rt, fetch_runner=runner)
    ctx = SearchStepContext(
        settings=settings,
        request=SearchRequest(
            query="what is deepseek",
            fetchs={
                "content": True,
                "abstracts": {"query": None},
                "overview": {"query": None},
            },
        ),
        candidate_urls=["https://example.com"],
    )

    anyio.run(step.run, ctx)

    req = capture.requests[0]
    assert req.abstracts is not False
    assert req.overview is not False
    assert req.abstracts.query == "what is deepseek"
    assert req.overview.query == "what is deepseek"
