from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.request import FetchRequest
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


class SearchFetchStep(StepBase[SearchStepContext]):
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
    async def run_inner(self, ctx: SearchStepContext) -> SearchStepContext:
        if bool(ctx.deep.aborted):
            ctx.fetch.candidates = []
            ctx.output.results = []
            return ctx
        urls = list(ctx.prefetch.urls or [])
        if not urls:
            ctx.fetch.candidates = []
            ctx.output.results = []
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
                            req.others.max_image_links
                            if req.others is not None
                            else None
                        ),
                    ),
                )
            )
        out = await self._fetch_runner.run_batch(to_fetch)
        fetched_candidates: list[SearchFetchedCandidate] = []
        for item in out:
            if item.output.result is None or item.fatal:
                await self.emit_tracking_event(
                    event_name="search.fetch.error",
                    request_id=ctx.request_id,
                    stage="search_fetch",
                    status="error",
                    error_code="search_fetch_failed",
                    attrs={
                        "url": str(item.url),
                        "url_index": int(item.url_index),
                    },
                )
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
        return ctx

    def _build_fetch_request(self, *, ctx: SearchStepContext, url: str) -> FetchRequest:
        template = ctx.request.fetchs
        return FetchRequest(
            urls=[url],
            crawl_mode=template.crawl_mode,
            crawl_timeout=template.crawl_timeout,
            content=template.content,
            abstracts=template.abstracts,
            subpages=template.subpages,
            overview=template.overview,
            others=template.others,
        )


__all__ = ["SearchFetchStep"]
