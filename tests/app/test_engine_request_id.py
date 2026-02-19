from __future__ import annotations

from typing_extensions import override

import anyio

from serpsage.app.bootstrap import build_runtime
from serpsage.app.engine import Engine
from serpsage.app.request import FetchRequest, SearchRequest
from serpsage.models.pipeline import FetchStepContext, SearchStepContext
from serpsage.settings.models import AppSettings
from serpsage.steps.base import RunnerBase, StepBase
from serpsage.telemetry.base import SpanBase


class _CaptureSearchStep(StepBase[SearchStepContext]):
    def __init__(self, *, rt) -> None:
        super().__init__(rt=rt)
        self.request_ids: list[str] = []

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        del span
        self.request_ids.append(ctx.request_id)
        return ctx


class _CaptureFetchStep(StepBase[FetchStepContext]):
    def __init__(self, *, rt) -> None:
        super().__init__(rt=rt)
        self.request_ids: list[str] = []

    @override
    async def run_inner(
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        del span
        self.request_ids.append(ctx.request_id)
        return ctx


def test_engine_search_generates_and_returns_request_id() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    capture_search = _CaptureSearchStep(rt=rt)
    capture_fetch = _CaptureFetchStep(rt=rt)
    engine = Engine(
        rt=rt,
        search_runner=RunnerBase[SearchStepContext](rt=rt, steps=[capture_search]),
        fetch_runner=RunnerBase[FetchStepContext](rt=rt, steps=[capture_fetch]),
    )
    req = SearchRequest(query="q", fetchs={"content": True})

    resp = anyio.run(engine.search, req)

    assert resp.request_id
    assert len(capture_search.request_ids) == 1
    assert capture_search.request_ids[0] == resp.request_id


def test_engine_fetch_generates_request_id_for_all_url_contexts() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    capture_search = _CaptureSearchStep(rt=rt)
    capture_fetch = _CaptureFetchStep(rt=rt)
    engine = Engine(
        rt=rt,
        search_runner=RunnerBase[SearchStepContext](rt=rt, steps=[capture_search]),
        fetch_runner=RunnerBase[FetchStepContext](rt=rt, steps=[capture_fetch]),
    )
    req = FetchRequest(
        urls=["https://example.com/a", "https://example.com/b"],
        content=True,
    )

    resp = anyio.run(engine.fetch, req)

    assert resp.request_id
    assert len(capture_fetch.request_ids) == 2
    assert all(item == resp.request_id for item in capture_fetch.request_ids)
