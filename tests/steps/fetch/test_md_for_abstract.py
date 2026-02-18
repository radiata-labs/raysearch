from __future__ import annotations

import anyio

from serpsage.app.bootstrap import build_runtime
from serpsage.app.request import FetchContentRequest, FetchRequest
from serpsage.app.response import FetchResultItem
from serpsage.models.extract import ExtractedDocument
from serpsage.models.pipeline import FetchStepContext, FetchStepOthers, ScoredAbstract
from serpsage.settings.models import AppSettings
from serpsage.steps.fetch.finalize import FetchFinalizeStep
from serpsage.steps.fetch.subpages import _to_subpage_result


def test_fetch_finalize_does_not_expose_md_for_abstract() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    step = FetchFinalizeStep(rt=rt)
    ctx = FetchStepContext(
        settings=settings,
        request=FetchRequest(urls=["https://example.com"], content=True),
        url="https://example.com",
        url_index=0,
        others=FetchStepOthers(),
        content_request=FetchContentRequest(),
        extracted=ExtractedDocument(
            title="title",
            markdown="markdown",
            md_for_abstract="clean markdown",
        ),
        scored_abstracts=[
            ScoredAbstract(abstract_id="S1:A1", text="abs", score=0.9),
        ],
    )

    out = anyio.run(step.run, ctx)

    assert out.result is not None
    assert "md_for_abstract" not in out.result.model_dump()


def test_subpage_result_does_not_expose_md_for_abstract() -> None:
    value = FetchResultItem(
        url="https://example.com/sub",
        title="sub",
        content="",
        abstracts=[],
        abstract_scores=[],
    )

    result = _to_subpage_result(value)

    assert "md_for_abstract" not in result.model_dump()
