from __future__ import annotations

from typing_extensions import override

from raysearch.dependencies import FETCH_RUNNER, Depends
from raysearch.models.app.request import FetchRequest
from raysearch.models.app.response import FetchResponse
from raysearch.models.steps.fetch import FetchStepContext
from raysearch.models.steps.search import SearchFetchedCandidate, SearchStepContext
from raysearch.steps.base import RunnerBase, StepBase
from raysearch.utils import (
    clean_whitespace,
    pick_earliest_published_date,
    published_date_in_range,
)


class SearchFetchStep(StepBase[SearchStepContext]):
    fetch_runner: RunnerBase[FetchStepContext] = Depends(FETCH_RUNNER)

    @override
    async def should_run(self, ctx: SearchStepContext) -> bool:
        """Execute unless aborted or no URLs to fetch."""
        if bool(ctx.plan.aborted):
            return False
        urls = list(ctx.retrieval.urls or [])
        return bool(urls)

    @override
    async def run_inner(self, ctx: SearchStepContext) -> SearchStepContext:
        # Pre-condition: should_run() verified URLs exist and not aborted
        urls = list(ctx.retrieval.urls or [])
        if not urls:
            ctx.fetch.candidates = []
            ctx.output.results = []
            return ctx
        pre_fetched_items = dict(ctx.retrieval.pre_fetched_items or {})
        to_fetch: list[FetchStepContext] = []
        for index, url in enumerate(urls):
            req = self._build_fetch_request(ctx=ctx, url=url)
            main_links_limit = (
                int(req.others.max_links)
                if req.others is not None and req.others.max_links is not None
                else None
            )
            fetch_ctx = FetchStepContext(
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
            pre_item = pre_fetched_items.get(url)
            if pre_item is not None:
                fetch_ctx.page.pre_fetched_title = pre_item.title
                fetch_ctx.page.pre_fetched_content = pre_item.content
                fetch_ctx.page.pre_fetched_author = pre_item.author
            to_fetch.append(fetch_ctx)
        out = await self.fetch_runner.run_batch(to_fetch)
        fetched_candidates: list[SearchFetchedCandidate] = []
        for item in out:
            if item.result is None or item.error.failed:
                await self.tracker.error(
                    name="search.fetch.failed",
                    request_id=ctx.request_id,
                    step="search.fetch",
                    error_code="search_fetch_failed",
                    data={
                        "url": str(item.url),
                        "url_index": int(item.url_index),
                    },
                )
                continue
            provider_published_date = clean_whitespace(
                ctx.retrieval.published_dates.get(str(item.url), "")
            )
            result = item.result.model_copy(
                update={
                    "published_date": pick_earliest_published_date(
                        provider_published_date,
                        item.result.published_date,
                    )
                }
            )
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
                    result=result,
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
        fetched_candidates = self._filter_candidates_by_published_date(
            candidates=fetched_candidates,
            start_published_date=ctx.request.start_published_date,
            end_published_date=ctx.request.end_published_date,
        )
        ctx.fetch.candidates = fetched_candidates
        ctx.output.results = [candidate.result for candidate in fetched_candidates]
        await self.tracker.info(
            name="search.fetch.completed",
            request_id=ctx.request_id,
            step="search.fetch",
            data={
                "candidate_count": len(fetched_candidates),
                "url_count": len(urls),
            },
        )
        await self.tracker.debug(
            name="search.fetch.detail",
            request_id=ctx.request_id,
            step="search.fetch",
            data={
                "fetched_urls": [
                    {
                        "url": c.result.url,
                        "title": c.result.title[:100] if c.result.title else "",
                        "published_date": c.result.published_date,
                    }
                    for c in fetched_candidates[:20]
                ],
                "crawl_mode": ctx.request.fetchs.crawl_mode,
                "overview_enabled": ctx.request.fetchs.overview,
                "abstracts_enabled": ctx.request.fetchs.abstracts,
            },
        )
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

    def _filter_candidates_by_published_date(
        self,
        *,
        candidates: list[SearchFetchedCandidate],
        start_published_date: str | None,
        end_published_date: str | None,
    ) -> list[SearchFetchedCandidate]:
        if not clean_whitespace(
            str(start_published_date or "")
        ) and not clean_whitespace(str(end_published_date or "")):
            return candidates
        return [
            candidate
            for candidate in candidates
            if published_date_in_range(
                candidate.result.published_date,
                start_published_date=str(start_published_date or ""),
                end_published_date=str(end_published_date or ""),
            )
        ]


def _derive_subpage_links_limit(main_links_limit: int | None) -> int | None:
    if main_links_limit is None:
        return None
    return max(8, int(round(float(main_links_limit) * 0.30)))


__all__ = ["SearchFetchStep"]
