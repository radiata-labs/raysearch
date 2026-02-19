from __future__ import annotations

import anyio

from serpsage.app.bootstrap import build_runtime
from serpsage.app.request import (
    FetchAbstractsRequest,
    FetchContentRequest,
    FetchOverviewRequest,
    FetchRequest,
)
from serpsage.models.pipeline import FetchRuntimeConfig, FetchStepContext
from serpsage.settings.models import AppSettings
from serpsage.steps.fetch.prepare import FetchPrepareStep


def _build_ctx(*, settings: AppSettings, request: FetchRequest) -> FetchStepContext:
    return FetchStepContext(
        settings=settings,
        request=request,
        url="https://example.com",
        url_index=0,
        runtime=FetchRuntimeConfig(),
    )


def test_prepare_builds_default_requests_when_switches_are_true() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    step = FetchPrepareStep(rt=rt)
    ctx = _build_ctx(
        settings=settings,
        request=FetchRequest(
            urls=["https://example.com"],
            content=False,
            abstracts=True,
            overview=True,
        ),
    )

    out = anyio.run(step.run, ctx)

    assert out.fatal is False
    assert out.resolved.return_content is False
    assert out.resolved.abstracts_request is not None
    assert out.resolved.abstracts_request.query is None
    assert out.resolved.overview_request is not None
    assert out.resolved.overview_request.query is None


def test_prepare_normalizes_blank_queries_to_none() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    step = FetchPrepareStep(rt=rt)
    ctx = _build_ctx(
        settings=settings,
        request=FetchRequest(
            urls=["https://example.com"],
            abstracts=FetchAbstractsRequest(query=" "),
            overview=FetchOverviewRequest(query=" "),
        ),
    )

    out = anyio.run(step.run, ctx)

    assert out.fatal is False
    assert out.resolved.abstracts_request is not None
    assert out.resolved.abstracts_request.query is None
    assert out.resolved.overview_request is not None
    assert out.resolved.overview_request.query is None


def test_prepare_maps_content_detail_to_internal_depth() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    step = FetchPrepareStep(rt=rt)
    ctx = _build_ctx(
        settings=settings,
        request=FetchRequest(
            urls=["https://example.com"],
            content=FetchContentRequest(detail="full"),
        ),
    )

    out = anyio.run(step.run, ctx)

    assert out.fatal is False
    assert out.resolved.content_request.detail == "full"
    assert out.resolved.content_options.detail == "full"
