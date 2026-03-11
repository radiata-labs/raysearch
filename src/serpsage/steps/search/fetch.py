from __future__ import annotations

from typing_extensions import override

from serpsage.dependencies import FETCH_RUNNER, Depends
from serpsage.models.app.request import FetchRequest
from serpsage.models.app.response import FetchResponse
from serpsage.models.steps.fetch import FetchStepContext
from serpsage.models.steps.search import SearchFetchedCandidate, SearchStepContext
from serpsage.steps.base import RunnerBase, StepBase


class SearchFetchStep(StepBase[SearchStepContext]):
    fetch_runner: RunnerBase[FetchStepContext] = Depends(FETCH_RUNNER)

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
            main_links_limit = (
                int(req.others.max_links)
                if req.others is not None and req.others.max_links is not None
                else None
            )
            fetch_ctx = FetchStepContext(
                settings=ctx.settings,
                request=req,
                request_id=ctx.request_id,
                response=FetchResponse(
                    request_id=ctx.request_id,
                    results=[],
                    statuses=[],
                ),
                url=url,
                url_index=index,
            )
            fetch_ctx.related.enabled = True
            fetch_ctx.page.crawl_mode = req.crawl_mode
            fetch_ctx.page.crawl_timeout_s = float(req.crawl_timeout or 0.0)
            fetch_ctx.related.link_limit = (
                req.others.max_links if req.others is not None else None
            )
            fetch_ctx.related.image_limit = (
                req.others.max_image_links if req.others is not None else None
            )
            fetch_ctx.related.subpages.candidate_limit = _derive_subpage_links_limit(
                main_links_limit
            )
            to_fetch.append(fetch_ctx)
        out = await self.fetch_runner.run_batch(to_fetch)
        fetched_candidates: list[SearchFetchedCandidate] = []
        for item in out:
            if item.result is None or item.error.failed:
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
            main_abstract_text = (
                str(item.page.doc.content.abstract_text or "")
                if item.page.doc is not None
                else ""
            )
            main_overview_scores = [
                float(scored.score)
                for scored in list(item.analysis.overview.ranked or [])
            ]
            fetched_candidates.append(
                SearchFetchedCandidate(
                    result=item.result,
                    links=(
                        list(item.page.doc.refs.links)
                        if item.page.doc is not None
                        else []
                    ),
                    subpage_links=[
                        list(subpage.doc.refs.links)
                        for subpage in list(item.related.subpages.items or [])
                        if subpage.doc is not None
                    ],
                    main_abstract_text=main_abstract_text,
                    subpage_abstract_texts=[
                        str(subpage.doc.content.abstract_text or "")
                        for subpage in list(item.related.subpages.items or [])
                        if subpage.doc is not None
                    ],
                    main_overview_scores=main_overview_scores,
                    subpages_overview_scores=[
                        [float(score) for score in list(subpage.overview_scores or [])]
                        for subpage in list(item.related.subpages.items or [])
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


def _derive_subpage_links_limit(main_links_limit: int | None) -> int | None:
    if main_links_limit is None:
        return None
    return max(8, int(round(float(main_links_limit) * 0.30)))


__all__ = ["SearchFetchStep"]
