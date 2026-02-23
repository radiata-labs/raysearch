from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.request import (
    FetchAbstractsRequest,
    FetchContentRequest,
    FetchRequestBase,
    FetchSubpagesRequest,
    SearchRequest,
)
from serpsage.models.pipeline import ResearchStepContext, SearchStepContext
from serpsage.steps.base import StepBase
from serpsage.steps.research.utils import (
    add_error,
    upsert_source_from_fetch_result,
)

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.steps.base import RunnerBase
    from serpsage.telemetry.base import SpanBase


class ResearchSearchStep(StepBase[ResearchStepContext]):
    span_name = "step.research_search"

    def __init__(
        self,
        *,
        rt: Runtime,
        search_runner: RunnerBase[SearchStepContext],
    ) -> None:
        super().__init__(rt=rt)
        self._search_runner = search_runner
        self.bind_deps(search_runner)

    @override
    async def run_inner(
        self, ctx: ResearchStepContext, *, span: SpanBase
    ) -> ResearchStepContext:
        if ctx.runtime.stop or ctx.current_round is None:
            return ctx

        jobs = list(ctx.work.search_jobs)
        if not jobs:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "no_search_jobs"
            ctx.current_round.stop = True
            ctx.current_round.stop_reason = "no_search_jobs"
            return ctx

        contexts: list[SearchStepContext] = []
        for idx, job in enumerate(jobs):
            req = SearchRequest(
                query=job.query,
                additional_queries=None,
                mode=job.mode,
                max_results=int(ctx.runtime.budget.max_results_per_search),
                include_domains=(list(job.include_domains) or None),
                exclude_domains=(list(job.exclude_domains) or None),
                include_text=(list(job.include_text) or None),
                exclude_text=(list(job.exclude_text) or None),
                fetchs=FetchRequestBase(
                    crawl_mode="fallback",
                    content=FetchContentRequest(detail="full"),
                    abstracts=FetchAbstractsRequest(query=ctx.request.themes, max_chars=2200),
                    subpages=FetchSubpagesRequest(
                        max_subpages=max(1, min(4, int(ctx.runtime.budget.max_fetch_per_round))),
                        subpage_keywords=None,
                    ),
                    overview=False,
                    others=None,
                ),
            )
            contexts.append(
                SearchStepContext(
                    settings=ctx.settings,
                    request=req,
                    request_id=f"{ctx.request_id}:research:{ctx.current_round.round_index}:{idx}",
                )
            )

        out = await self._search_runner.run_batch(contexts)
        all_results = []
        new_source_ids: list[int] = []
        per_round_fetch_calls = 0
        for item in out:
            if item.errors:
                ctx.errors.extend(item.errors)
            results = list(item.output.results or [])
            all_results.extend(results)
            per_round_fetch_calls += int(len(results))
            for result in results:
                source_ids = upsert_source_from_fetch_result(
                    ctx=ctx,
                    result=result,
                    round_index=ctx.current_round.round_index,
                )
                for source_id in source_ids:
                    if source_id not in new_source_ids:
                        new_source_ids.append(source_id)

        ctx.work.search_results = all_results
        ctx.current_round.result_count = int(len(all_results))
        ctx.current_round.new_source_ids = list(new_source_ids)
        ctx.runtime.search_calls += int(len(jobs))
        ctx.runtime.fetch_calls += int(per_round_fetch_calls)

        if int(ctx.runtime.fetch_calls) > int(ctx.runtime.budget.max_fetch_calls):
            add_error(
                ctx,
                code="research_fetch_budget_soft_exceeded",
                message="search pipeline returned more fetched pages than logical budget",
                details={
                    "fetch_calls": int(ctx.runtime.fetch_calls),
                    "max_fetch_calls": int(ctx.runtime.budget.max_fetch_calls),
                    "round_index": int(ctx.current_round.round_index),
                },
            )

        span.set_attr("round_index", int(ctx.current_round.round_index))
        span.set_attr("jobs", int(len(jobs)))
        span.set_attr("results", int(len(all_results)))
        span.set_attr("new_source_ids", int(len(new_source_ids)))
        span.set_attr("search_calls", int(ctx.runtime.search_calls))
        span.set_attr("fetch_calls", int(ctx.runtime.fetch_calls))
        return ctx


__all__ = ["ResearchSearchStep"]

