from __future__ import annotations

from typing_extensions import override

import anyio

from serpsage.app.bootstrap import build_runtime
from serpsage.app.request import SearchRequest
from serpsage.app.response import FetchResultItem, FetchSubpagesResult
from serpsage.models.pipeline import (
    FetchStepContext,
    ScoredAbstract,
    SearchPrefetchState,
    SearchStepContext,
)
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
        ctx.artifacts.overview_scored_abstracts = [
            ScoredAbstract(abstract_id="S1:A1", text="a", score=0.9),
            ScoredAbstract(abstract_id="S1:A2", text="b", score=0.8),
        ]
        ctx.subpages.overview_scores = [[0.4]]
        ctx.subpages.md_for_abstract = ["sub text"]
        ctx.output.result = FetchResultItem(
            url=ctx.url,
            title="",
            content="",
            abstracts=[],
            abstract_scores=[],
            subpages=[
                FetchSubpagesResult(
                    url=f"{ctx.url}/sub",
                    title="sub",
                    content="",
                    abstracts=[],
                    abstract_scores=[],
                )
            ],
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
        prefetch=SearchPrefetchState(urls=["https://example.com"]),
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
        prefetch=SearchPrefetchState(urls=["https://example.com"]),
    )

    anyio.run(step.run, ctx)

    req = capture.requests[0]
    assert req.abstracts is not False
    assert req.overview is not False
    assert req.abstracts.query == "what is deepseek"
    assert req.overview.query == "what is deepseek"


def test_search_fetch_propagates_overview_scores_for_rerank() -> None:
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
        prefetch=SearchPrefetchState(urls=["https://example.com"]),
    )

    out = anyio.run(step.run, ctx)

    assert len(out.fetch.candidates) == 1
    assert out.fetch.candidates[0].main_overview_scores == [0.9, 0.8]
    assert out.fetch.candidates[0].subpages_overview_scores == [[0.4]]
