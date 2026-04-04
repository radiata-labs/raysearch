"""Research search step module."""

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
from raysearch.models.steps.research import ResearchSource, RoundStepContext
from raysearch.models.steps.research.payloads import PlanSearchJobPayload
from raysearch.models.steps.search import (
    SearchFetchedCandidate,
    SearchRuntimeState,
    SearchStepContext,
)
from raysearch.steps.base import RunnerBase, StepBase
from raysearch.tokenize import tokenize_for_query


class ResearchSearchStep(StepBase[RoundStepContext]):
    search_runner: RunnerBase[SearchStepContext] = Depends(SEARCH_RUNNER)

    @override
    async def should_run(self, ctx: RoundStepContext) -> bool:
        """Execute only for search round action (not explore)."""
        if ctx.run.stop or ctx.run.current is None:
            return False
        round_action = (ctx.run.current.round_action or "search").casefold()
        # First round is always search; explore only possible after round 1.
        if ctx.run.current.round_index <= 1:
            round_action = "search"
        return round_action == "search"

    @override
    async def run_inner(self, ctx: RoundStepContext) -> RoundStepContext:
        # Pre-condition: should_run() already verified round_action == "search"
        assert ctx.run.current is not None
        if (
            ctx.run.current.search_fetched_candidates
            and not ctx.run.current.pending_search_jobs
        ):
            return self._run_search_fetch(ctx)
        ctx = await self._run_search_action(ctx)
        if ctx.run.current is None or ctx.run.current.pending_search_jobs:
            return ctx
        if not ctx.run.current.search_fetched_candidates:
            return ctx
        return self._run_search_fetch(ctx)

    async def _run_search_action(self, ctx: RoundStepContext) -> RoundStepContext:
        assert ctx.run.current is not None
        jobs = list(ctx.run.current.pending_search_jobs or ctx.run.current.search_jobs)
        if not jobs:
            return ctx
        remaining_search_budget = ctx.run.allocation.search_remaining
        if remaining_search_budget <= 0:
            ctx.run.current.waiting_for_budget = True
            ctx.run.current.waiting_reason = "max_search_calls"
            return ctx
        jobs = jobs[:remaining_search_budget]
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
                        # QuerySourceSpec fields are primitives or list[str]; shallow copy suffices.
                        additional_queries=[
                            item.model_copy()
                            for item in list(job.additional_queries or [])
                        ],
                    ),
                    request_id=f"{ctx.request_id}:research:{ctx.run.current.round_index}:{idx}",
                )
            )
        out = await self.search_runner.run_batch(contexts)
        prepared_candidates = list(ctx.run.current.search_fetched_candidates)
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
        ctx.run.current.waiting_for_budget = bool(ctx.run.current.pending_search_jobs)
        ctx.run.current.waiting_reason = (
            "max_search_calls" if ctx.run.current.pending_search_jobs else ""
        )
        return ctx

    def _run_search_fetch(self, ctx: RoundStepContext) -> RoundStepContext:
        assert ctx.run.current is not None
        candidates = [
            item.model_copy(deep=True)
            for item in list(ctx.run.current.search_fetched_candidates)
        ]
        remaining_fetch_budget = ctx.run.allocation.fetch_remaining
        if remaining_fetch_budget <= 0:
            ctx.run.current.waiting_for_budget = True
            ctx.run.current.waiting_reason = "max_fetch_calls"
            return ctx
        if ctx.run.current.pending_search_jobs:
            return ctx
        selected_candidates, deferred_candidates, per_round_fetch_calls = (
            self._select_candidates_for_fetch_budget(
                candidates=candidates,
                remaining_fetch_budget=remaining_fetch_budget,
            )
        )
        round_link_candidates: dict[int, SearchFetchedCandidate] = {}
        result_count = 0
        for candidate in selected_candidates:
            main_source_id = self._append_sources_from_fetch_result(
                ctx=ctx,
                result=candidate.result,
                round_index=ctx.run.current.round_index,
            )
            round_link_candidates[main_source_id] = candidate.model_copy(deep=True)
            result_count += 1
        ctx.run.current.search_fetched_candidates = [
            item.model_copy(deep=True) for item in deferred_candidates
        ]
        ctx.run.current.waiting_for_budget = bool(deferred_candidates)
        ctx.run.current.waiting_reason = (
            "max_fetch_calls" if deferred_candidates else ""
        )
        self._finalize_round_state(
            ctx=ctx,
            result_count=result_count,
            per_round_fetch_calls=per_round_fetch_calls,
            next_round_link_candidates=round_link_candidates,
        )
        if (
            not ctx.run.current.waiting_for_budget
            and not ctx.run.current.pending_search_jobs
        ):
            ctx.run.current.search_fetched_candidates = []
        return ctx

    def _build_search_request(
        self,
        *,
        ctx: RoundStepContext,
        query_job: PlanSearchJobPayload,
        subpages_request: FetchSubpagesRequest | None,
        main_links_limit: int,
    ) -> SearchRequest:
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
        for index, candidate in enumerate(candidates):
            if index in consumed_indexes:
                continue
            if candidate.result.url != result.url:
                continue
            consumed_indexes.add(index)
            return candidate
        return None

    def _trim_candidate_to_fetch_budget(
        self,
        *,
        candidate: SearchFetchedCandidate,
        remaining_fetch_budget: int,
    ) -> tuple[SearchFetchedCandidate, int]:
        pages_left = max(0, remaining_fetch_budget)
        if pages_left <= 0:
            return (
                candidate.model_copy(
                    update={
                        "result": candidate.result.model_copy(update={"subpages": []}),
                        "subpage_links": [],
                    }
                ),
                0,
            )
        allowed_subpages = max(0, pages_left - 1)
        trimmed = candidate.model_copy(
            update={
                "result": candidate.result.model_copy(
                    update={
                        "subpages": list(candidate.result.subpages or [])[
                            :allowed_subpages
                        ]
                    }
                ),
                "subpage_links": list(candidate.subpage_links or [])[:allowed_subpages],
            }
        )
        return trimmed, 1 + len(trimmed.result.subpages)

    def _select_candidates_for_fetch_budget(
        self,
        *,
        candidates: list[SearchFetchedCandidate],
        remaining_fetch_budget: int,
    ) -> tuple[list[SearchFetchedCandidate], list[SearchFetchedCandidate], int]:
        pages_left = max(0, remaining_fetch_budget)
        if pages_left <= 0 or not candidates:
            return [], [item.model_copy(deep=True) for item in candidates], 0
        main_selected = min(len(candidates), pages_left)
        selected_slots = [
            1 if idx < main_selected else 0 for idx in range(len(candidates))
        ]
        pages_left -= main_selected
        subpage_indexes = [0] * len(candidates)
        while pages_left > 0:
            progressed = False
            for idx, candidate in enumerate(candidates):
                available_subpages = list(candidate.result.subpages or [])
                if selected_slots[idx] <= 0 or subpage_indexes[idx] >= len(
                    available_subpages
                ):
                    continue
                selected_slots[idx] += 1
                subpage_indexes[idx] += 1
                pages_left -= 1
                progressed = True
                if pages_left <= 0:
                    break
            if not progressed:
                break
        selected: list[SearchFetchedCandidate] = []
        deferred: list[SearchFetchedCandidate] = []
        consumed = 0
        for idx, candidate in enumerate(candidates):
            pages_for_candidate = selected_slots[idx]
            if pages_for_candidate <= 0:
                deferred.append(candidate.model_copy(deep=True))
                continue
            trimmed, candidate_consumed = self._trim_candidate_to_fetch_budget(
                candidate=candidate,
                remaining_fetch_budget=pages_for_candidate,
            )
            consumed += candidate_consumed
            selected.append(trimmed)
            remaining_subpages = list(candidate.result.subpages or [])[
                max(0, pages_for_candidate - 1) :
            ]
            remaining_subpage_links = list(candidate.subpage_links or [])[
                max(0, pages_for_candidate - 1) :
            ]
            if remaining_subpages:
                deferred.append(
                    candidate.model_copy(
                        update={
                            "result": candidate.result.model_copy(
                                update={"subpages": remaining_subpages}
                            ),
                            "subpage_links": remaining_subpage_links,
                        },
                        deep=True,
                    )
                )
        return selected, deferred, consumed

    def _finalize_round_state(
        self,
        *,
        ctx: RoundStepContext,
        result_count: int,
        per_round_fetch_calls: int,
        next_round_link_candidates: dict[int, SearchFetchedCandidate],
    ) -> None:
        assert ctx.run.current is not None
        ctx.run.link_candidates = {
            source_id: item.model_copy(deep=True)
            for source_id, item in next_round_link_candidates.items()
        }
        ctx.run.link_candidates_round = ctx.run.current.round_index
        ctx.run.current.result_count = max(
            0, ctx.run.current.result_count + max(0, result_count)
        )
        self._refresh_source_state(ctx=ctx)
        ctx.run.allocation.fetch_used += max(0, per_round_fetch_calls)

    def _append_sources_from_fetch_result(
        self,
        *,
        ctx: RoundStepContext,
        result: FetchResultItem,
        round_index: int,
    ) -> int:
        main_source_id = self._append_source(
            ctx=ctx,
            url=result.url,
            title=result.title,
            overview=str(result.overview or ""),
            content=str(result.content or ""),
            round_index=round_index,
            is_subpage=False,
        )
        for sub in list(result.subpages or []):
            self._append_source(
                ctx=ctx,
                url=sub.url,
                title=sub.title,
                overview=str(sub.overview or ""),
                content=str(sub.content or ""),
                round_index=round_index,
                is_subpage=True,
            )
        return main_source_id

    def _append_source(
        self,
        *,
        ctx: RoundStepContext,
        url: str,
        title: str,
        overview: str,
        content: str,
        round_index: int,
        is_subpage: bool,
    ) -> int:
        source_id = len(ctx.knowledge.sources) + 1
        ctx.knowledge.sources.append(
            ResearchSource(
                source_id=source_id,
                url=url,
                canonical_url=url,
                title=title,
                overview=overview,
                content=content,
                round_index=round_index,
                is_subpage=is_subpage,
            )
        )
        return source_id

    def _refresh_source_state(self, *, ctx: RoundStepContext) -> None:
        source_ids_by_url: dict[str, list[int]] = {}
        for source in ctx.knowledge.sources:
            ids = list(source_ids_by_url.get(source.canonical_url or source.url, []))
            ids.append(source.source_id)
            source_ids_by_url[source.canonical_url or source.url] = ids
        ctx.knowledge.source_ids_by_url = source_ids_by_url
        ctx.knowledge.ranked_source_ids = [
            source.source_id for source in reversed(ctx.knowledge.sources)
        ]
        ctx.knowledge.source_scores = {
            source.source_id: 1.0 for source in ctx.knowledge.sources
        }


__all__ = ["ResearchSearchStep"]
