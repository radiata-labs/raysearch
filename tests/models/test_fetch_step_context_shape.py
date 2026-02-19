from __future__ import annotations

from serpsage.app.request import FetchRequest
from serpsage.models.errors import AppError
from serpsage.models.pipeline import FetchRuntimeConfig, FetchStepContext
from serpsage.settings.models import AppSettings


def _build_ctx() -> FetchStepContext:
    return FetchStepContext(
        settings=AppSettings(),
        request=FetchRequest(urls=["https://example.com"], content=True),
        url="https://example.com",
        url_index=0,
        runtime=FetchRuntimeConfig(),
    )


def test_fetch_step_context_supports_minimal_grouped_construction() -> None:
    ctx = _build_ctx()

    assert ctx.fatal is False
    assert ctx.errors == []
    assert ctx.resolved.return_content is True
    assert ctx.artifacts.fetch_result is None
    assert ctx.subpages.results == []
    assert ctx.output.result is None


def test_fetch_step_context_mutable_defaults_are_not_shared() -> None:
    first = _build_ctx()
    second = _build_ctx()

    first.subpages.keywords.append("docs")
    first.output.others.links.append("https://example.com/a")

    assert second.subpages.keywords == []
    assert second.output.others.links == []


def test_fetch_step_context_keeps_fatal_and_errors_on_top_level() -> None:
    ctx = _build_ctx()

    ctx.fatal = True
    ctx.errors.append(AppError(code="x", message="y"))

    assert ctx.fatal is True
    assert len(ctx.errors) == 1
