from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.request import (
    FetchAbstractsRequest,
    FetchOverviewRequest,
    FetchRequest,
)
from serpsage.models.pipeline import (
    FetchRuntimeConfig,
    FetchStepContext,
    SearchFetchedCandidate,
    SearchStepContext,
)
from serpsage.steps.base import StepBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.steps.base import RunnerBase
    from serpsage.telemetry.base import SpanBase


class SearchFetchStep(StepBase[SearchStepContext]):
    span_name = "step.search_fetch"

    def __init__(
        self,
        *,
        rt: Runtime,
        fetch_runner: RunnerBase[FetchStepContext],
    ) -> None:
        super().__init__(rt=rt)
        self._fetch_runner = fetch_runner
        self.bind_deps(fetch_runner)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        urls = list(ctx.prefetch.urls or [])
        if not urls:
            ctx.fetch.candidates = []
            ctx.output.results = []
            span.set_attr("fetch_candidates", 0)
            return ctx

        to_fetch: list[FetchStepContext] = []
        for index, url in enumerate(urls):
            req = self._build_fetch_request(ctx=ctx, url=url)
            to_fetch.append(
                FetchStepContext(
                    settings=ctx.settings,
                    request=req,
                    request_id=ctx.request_id,
                    url=url,
                    url_index=index,
                    runtime=FetchRuntimeConfig(
                        crawl_mode=req.crawl_mode,
                        crawl_timeout_s=float(req.crawl_timeout or 0.0),
                        max_links=(
                            req.others.max_links if req.others is not None else None
                        ),
                        max_image_links=(
                            req.others.max_image_links if req.others is not None else None
                        ),
                    ),
                )
            )

        out = await self._fetch_runner.run_batch(to_fetch)

        fetched_candidates: list[SearchFetchedCandidate] = []
        for item in out:
            ctx.errors.extend(item.errors)
            if item.output.result is None or item.fatal:
                continue
            main_md_for_abstract = (
                str(item.artifacts.extracted.md_for_abstract or "")
                if item.artifacts.extracted is not None
                else ""
            )
            main_overview_scores = [
                float(scored.score)
                for scored in list(item.artifacts.overview_scored_abstracts or [])
            ]
            fetched_candidates.append(
                SearchFetchedCandidate(
                    result=item.output.result,
                    main_md_for_abstract=main_md_for_abstract,
                    subpages_md_for_abstract=list(item.subpages.md_for_abstract or []),
                    main_overview_scores=main_overview_scores,
                    subpages_overview_scores=[
                        [float(score) for score in list(subpage_scores or [])]
                        for subpage_scores in list(item.subpages.overview_scores or [])
                    ],
                )
            )

        ctx.fetch.candidates = fetched_candidates
        ctx.output.results = [candidate.result for candidate in fetched_candidates]
        span.set_attr("fetch_candidates", int(len(urls)))
        span.set_attr("fetch_success_count", int(len(fetched_candidates)))
        span.set_attr(
            "fetch_failure_count",
            int(len(urls) - len(fetched_candidates)),
        )
        return ctx

    def _build_fetch_request(self, *, ctx: SearchStepContext, url: str) -> FetchRequest:
        template = ctx.request.fetchs
        search_query = ctx.request.query

        abstracts = template.abstracts
        if isinstance(abstracts, bool):
            if abstracts:
                abstracts_out: bool | FetchAbstractsRequest = FetchAbstractsRequest(
                    query=search_query
                )
            else:
                abstracts_out = False
        else:
            abstracts_query = abstracts.query or search_query
            abstracts_out = abstracts.model_copy(update={"query": abstracts_query})

        overview = template.overview
        if isinstance(overview, bool):
            if overview:
                overview_out: bool | FetchOverviewRequest = FetchOverviewRequest(
                    query=search_query
                )
            else:
                overview_out = False
        else:
            overview_query = overview.query or search_query
            overview_out = overview.model_copy(update={"query": overview_query})

        return FetchRequest(
            urls=[url],
            crawl_mode=template.crawl_mode,
            crawl_timeout=template.crawl_timeout,
            content=template.content,
            abstracts=abstracts_out,
            subpages=template.subpages,
            overview=overview_out,
            others=template.others,
        )


__all__ = ["SearchFetchStep"]
