from __future__ import annotations

from typing_extensions import override

import anyio

from serpsage.app.bootstrap import build_runtime
from serpsage.app.request import FetchRequest, FetchSubpagesRequest
from serpsage.app.response import FetchResultItem
from serpsage.components.rank.base import RankerBase
from serpsage.models.extract import ExtractedDocument, ExtractedLink
from serpsage.models.pipeline import (
    FetchRuntimeConfig,
    FetchStepContext,
    ScoredAbstract,
)
from serpsage.settings.models import AppSettings
from serpsage.steps.base import RunnerBase, StepBase
from serpsage.steps.fetch.subpages import FetchSubpageStep
from serpsage.telemetry.base import SpanBase


class _LinkRanker(RankerBase):
    async def score_texts(
        self,
        *,
        texts: list[str],
        query: str,
        query_tokens: list[str],
    ) -> list[float]:
        del query, query_tokens
        return [1.0 - (idx * 0.1) for idx, _ in enumerate(texts)]


class _ChildFetchStep(StepBase[FetchStepContext]):
    @override
    async def run_inner(
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        del span
        if ctx.url.endswith("/a"):
            ctx.artifacts.overview_scored_abstracts = [
                ScoredAbstract(abstract_id="S1:A1", text="a1", score=0.9),
                ScoredAbstract(abstract_id="S1:A2", text="a2", score=0.8),
            ]
        else:
            ctx.artifacts.overview_scored_abstracts = [
                ScoredAbstract(abstract_id="S1:A1", text="b1", score=0.7),
            ]
        ctx.artifacts.extracted = ExtractedDocument(
            title="sub",
            markdown="sub content",
            md_for_abstract="sub content",
        )
        ctx.output.result = FetchResultItem(
            url=ctx.url,
            title="sub",
            content="",
            abstracts=[],
            abstract_scores=[],
        )
        return ctx


def test_fetch_subpages_collects_overview_scores_in_aligned_order() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    child_runner = RunnerBase[FetchStepContext](rt=rt, steps=[_ChildFetchStep(rt=rt)])
    step = FetchSubpageStep(
        rt=rt,
        fetch_runner=child_runner,
        ranker=_LinkRanker(rt=rt),
    )
    request = FetchRequest(
        urls=["https://parent.com"],
        content=True,
        subpages=FetchSubpagesRequest(max_subpages=2, subpage_keywords="docs"),
    )
    ctx = FetchStepContext(
        settings=settings,
        request=request,
        url="https://parent.com",
        url_index=0,
        runtime=FetchRuntimeConfig(
            crawl_mode=request.crawl_mode,
            crawl_timeout_s=float(request.crawl_timeout or 0.0),
            max_links=None,
            max_image_links=None,
        ),
    )
    ctx.subpages.enabled = True
    ctx.subpages.max_count = 2
    ctx.subpages.query = "docs"
    ctx.subpages.keywords = ["docs"]
    ctx.subpages.links = [
        ExtractedLink(
            url="https://parent.com/a",
            anchor_text="A",
            section="primary",
        ),
        ExtractedLink(
            url="https://parent.com/b",
            anchor_text="B",
            section="primary",
        ),
    ]

    out = anyio.run(step.run, ctx)

    assert [item.url for item in out.subpages.results] == [
        "https://parent.com/a",
        "https://parent.com/b",
    ]
    assert out.subpages.overview_scores == [[0.9, 0.8], [0.7]]
    assert "subpages_overview_scores" not in out.subpages.results[0].model_dump()
