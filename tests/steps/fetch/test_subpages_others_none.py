from __future__ import annotations

import anyio

from serpsage.app.bootstrap import build_runtime
from serpsage.app.request import FetchRequest, FetchSubpagesRequest
from serpsage.app.response import FetchSubpagesResult
from serpsage.components.extract.base import ExtractorBase
from serpsage.models.extract import (
    ExtractContentOptions,
    ExtractedDocument,
    ExtractedLink,
)
from serpsage.models.fetch import FetchResult
from serpsage.models.pipeline import FetchRuntimeConfig, FetchStepContext
from serpsage.settings.models import AppSettings
from serpsage.steps.fetch.extract import FetchExtractStep
from serpsage.steps.fetch.finalize import FetchFinalizeStep
from serpsage.steps.fetch.prepare import FetchPrepareStep


class _DummyExtractor(ExtractorBase):
    def __init__(self, *, rt) -> None:
        super().__init__(rt=rt)
        self.collect_links_used = False

    def extract(
        self,
        *,
        url: str,
        content: bytes,
        content_type: str | None,
        content_options: ExtractContentOptions | None = None,
        include_secondary_content: bool = False,
        collect_links: bool = False,
        collect_images: bool = False,
    ) -> ExtractedDocument:
        del content, content_type, content_options, include_secondary_content, collect_images
        self.collect_links_used = collect_links
        return ExtractedDocument(
            title="page",
            markdown="main content",
            md_for_abstract="main content",
            content_kind="html",
            extractor_used="dummy",
            links=[
                ExtractedLink(
                    url=f"{url.rstrip('/')}/sub",
                    anchor_text="sub",
                    section="primary",
                )
            ],
        )


def _build_ctx(*, settings: AppSettings, request: FetchRequest) -> FetchStepContext:
    return FetchStepContext(
        settings=settings,
        request=request,
        url="https://example.com",
        url_index=0,
        runtime=FetchRuntimeConfig(
            crawl_mode=request.crawl_mode,
            crawl_timeout_s=float(request.crawl_timeout or 0.0),
            max_links=request.others.max_links if request.others is not None else None,
            max_image_links=(
                request.others.max_image_links if request.others is not None else None
            ),
        ),
    )


def test_subpages_still_collect_links_when_others_is_none() -> None:
    settings = AppSettings()
    settings.fetch.extract.min_text_chars = 1
    rt = build_runtime(settings=settings)
    prepare = FetchPrepareStep(rt=rt)
    extractor = _DummyExtractor(rt=rt)
    extract = FetchExtractStep(rt=rt, extractor=extractor)
    req = FetchRequest(
        urls=["https://example.com"],
        content=True,
        subpages=FetchSubpagesRequest(max_subpages=2, subpage_keywords="docs"),
        others=None,
    )
    ctx = _build_ctx(settings=settings, request=req)

    prepared = anyio.run(prepare.run, ctx)
    prepared.artifacts.fetch_result = FetchResult(
        url="https://example.com",
        status_code=200,
        content_type="text/html",
        content=b"<html></html>",
        fetch_mode="curl_cffi",
        content_kind="html",
    )
    out = anyio.run(extract.run, prepared)

    assert extractor.collect_links_used is True
    assert len(out.subpages.links) == 1
    assert out.output.others.links == []


def test_finalize_hides_others_but_keeps_subpages_when_others_is_none() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    step = FetchFinalizeStep(rt=rt)
    req = FetchRequest(
        urls=["https://example.com"],
        content=True,
        subpages=FetchSubpagesRequest(max_subpages=1, subpage_keywords="docs"),
        others=None,
    )
    ctx = _build_ctx(settings=settings, request=req)
    ctx.artifacts.extracted = ExtractedDocument(
        title="title",
        markdown="main content",
        md_for_abstract="main content",
        content_kind="html",
        extractor_used="dummy",
    )
    ctx.subpages.results = [
        FetchSubpagesResult(
            url="https://example.com/sub",
            title="sub",
            content="sub content",
            abstracts=[],
            abstract_scores=[],
        )
    ]
    ctx.output.others.links = ["https://example.com/should-not-show"]

    out = anyio.run(step.run, ctx)

    assert out.output.result is not None
    assert len(out.output.result.subpages) == 1
    assert out.output.result.others is None
    assert "others" not in out.output.result.model_dump()
