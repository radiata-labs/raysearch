"""Research search step module.

This module contains the ResearchSearchStep class for executing search operations.
All utility functions have been consolidated into utils.py.
"""

from __future__ import annotations

from typing_extensions import override

from raysearch.dependencies import SEARCH_RUNNER, Depends
from raysearch.models.app.request import (
    FetchContentRequest,
    FetchOthersRequest,
    FetchSubpagesRequest,
    SearchFetchRequest,
    SearchRequest,
)
from raysearch.models.app.response import FetchResultItem, SearchResponse
from raysearch.models.steps.research import RoundStepContext
from raysearch.models.steps.research.payloads import PlanSearchJobPayload
from raysearch.models.steps.search import (
    SearchFetchedCandidate,
    SearchRuntimeState,
    SearchStepContext,
)
from raysearch.steps.base import RunnerBase, StepBase
from raysearch.steps.research.utils import canonicalize_url
from raysearch.tokenize import tokenize_for_query


class ResearchSearchStep(StepBase[RoundStepContext]):
    """Step for executing search operations in the research pipeline."""

    search_runner: RunnerBase[SearchStepContext] = Depends(SEARCH_RUNNER)

    @override
    async def run_inner(self, ctx: RoundStepContext) -> RoundStepContext:
        if ctx.run.stop or ctx.run.current is None:
            return ctx
        round_action = (ctx.run.current.round_action or "search").casefold()
        if ctx.run.current.round_index <= 1:
            round_action = "search"
        if round_action != "search":
            return ctx
        if (
            ctx.run.current.search_fetched_candidates
            and not ctx.run.current.pending_search_jobs
        ):
            return ctx
        return await self._run_search_action(ctx)

    async def _run_search_action(self, ctx: RoundStepContext) -> RoundStepContext:
        """Execute search jobs and process results."""
        assert ctx.run.current is not None
        jobs = list(ctx.run.current.pending_search_jobs or ctx.run.current.search_jobs)
        if not jobs:
            return ctx
        remaining_search_budget = ctx.run.allocation.search_remaining
        if remaining_search_budget <= 0:
            ctx.run.current.waiting_for_budget = True
            ctx.run.current.waiting_reason = "max_search_calls"
            return ctx
        executable_jobs = min(len(jobs), remaining_search_budget)
        jobs = jobs[:executable_jobs]
        if not jobs:
            return ctx
        fetch_cfg = self.settings.fetch
        main_links_limit = max(1, int(fetch_cfg.extract.link_max_count))
        contexts: list[SearchStepContext] = []
        for idx, job in enumerate(jobs):
            max_subpages = max(0, ctx.run.limits.round_fetch_budget - 1)
            subpages_request = (
                FetchSubpagesRequest(
                    max_subpages=max_subpages,
                    subpage_keywords=tokenize_for_query(job.query.query),
                )
                if max_subpages > 0
                else None
            )
            req = self._build_search_request(
                ctx=ctx,
                query_job=job,
                subpages_request=subpages_request,
                main_links_limit=main_links_limit,
            )
            contexts.append(
                SearchStepContext(
                    request=req,
                    response=SearchResponse(
                        request_id=(
                            f"{ctx.request_id}:research:{ctx.run.current.round_index}:{idx}"
                        ),
                        search_mode=req.mode,
                        results=[],
                    ),
                    runtime=SearchRuntimeState(
                        disable_internal_llm=True,
                        engine_selection_subsystem="research",
                        additional_queries=[
                            item.model_copy(deep=True)
                            for item in list(job.additional_queries or [])
                        ],
                    ),
                    request_id=f"{ctx.request_id}:research:{ctx.run.current.round_index}:{idx}",
                )
            )
        out = await self.search_runner.run_batch(contexts)
        prepared_candidates: list[SearchFetchedCandidate] = list(
            ctx.run.current.search_fetched_candidates
        )
        for search_ctx in out:
            results = list(search_ctx.output.results or [])
            candidates = list(search_ctx.fetch.candidates or [])
            consumed_indexes: set[int] = set()
            for result in results:
                candidate = self._match_candidate_for_result(
                    result=result,
                    candidates=candidates,
                    consumed_indexes=consumed_indexes,
                )
                if candidate is None:
                    candidate = SearchFetchedCandidate(result=result)
                else:
                    candidate = candidate.model_copy(update={"result": result})
                prepared_candidates.append(candidate.model_copy(deep=True))
        ctx.run.current.search_fetched_candidates = [
            item.model_copy(deep=True) for item in prepared_candidates
        ]
        ctx.run.allocation.search_used += len(jobs)
        ctx.run.current.pending_search_jobs = [
            item.model_copy(deep=True)
            for item in list(
                (ctx.run.current.pending_search_jobs or ctx.run.current.search_jobs)[
                    len(jobs) :
                ]
            )
        ]
        ctx.run.current.waiting_for_budget = False
        ctx.run.current.waiting_reason = ""
        if ctx.run.current.pending_search_jobs:
            ctx.run.current.waiting_for_budget = True
            ctx.run.current.waiting_reason = "max_search_calls"
        return ctx

    def _build_search_request(
        self,
        *,
        ctx: RoundStepContext,
        query_job: PlanSearchJobPayload,
        subpages_request: FetchSubpagesRequest | None,
        main_links_limit: int,
    ) -> SearchRequest:
        """Build a search request from the query job."""
        return SearchRequest(
            query=query_job.query.query,
            user_location="US",
            additional_queries=(
                [item.query for item in list(query_job.additional_queries or [])]
                if query_job.mode == "deep"
                else None
            ),
            mode=query_job.mode,
            max_results=ctx.run.limits.max_results_per_search,
            fetchs=SearchFetchRequest(
                crawl_mode="fallback",
                crawl_timeout=30.0,
                content=FetchContentRequest(
                    detail="full",
                    max_chars=max(
                        ctx.run.limits.fetch_page_max_chars,
                        ctx.run.limits.report_source_batch_chars,
                    ),
                    include_markdown_links=False,
                    include_html_tags=False,
                ),
                abstracts=False,
                subpages=subpages_request,
                overview=True,
                others=FetchOthersRequest(max_links=main_links_limit),
            ),
        )

    def _match_candidate_for_result(
        self,
        *,
        result: FetchResultItem,
        candidates: list[SearchFetchedCandidate],
        consumed_indexes: set[int],
    ) -> SearchFetchedCandidate | None:
        """Find a matching candidate for a search result by URL."""
        target_url = result.url
        target_key = canonicalize_url(target_url) or target_url.casefold()
        for index, candidate in enumerate(candidates):
            if index in consumed_indexes:
                continue
            candidate_url = candidate.result.url
            candidate_key = canonicalize_url(candidate_url) or candidate_url.casefold()
            if candidate_key != target_key:
                continue
            consumed_indexes.add(index)
            return candidate
        return None


__all__ = ["ResearchSearchStep"]
