from __future__ import annotations

import json
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.request import (
    FetchAbstractsRequest,
    FetchContentRequest,
    FetchRequestBase,
    FetchSubpagesRequest,
    SearchRequest,
)
from serpsage.app.response import FetchResultItem, FetchSubpagesResult
from serpsage.models.errors import AppError
from serpsage.models.pipeline import (
    ResearchSource,
    ResearchStepContext,
    SearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.utils import (
    merge_strings,
    normalize_strings,
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
                    abstracts=FetchAbstractsRequest(
                        query=ctx.request.themes, max_chars=2200
                    ),
                    subpages=FetchSubpagesRequest(
                        max_subpages=max(
                            1, min(4, int(ctx.runtime.budget.max_fetch_per_round))
                        ),
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
                source_ids = self._upsert_source_from_fetch_result(
                    ctx=ctx,
                    result=result,
                    round_index=ctx.current_round.round_index,
                )
                for source_id in source_ids:
                    if source_id not in new_source_ids:
                        new_source_ids.append(source_id)

        ctx.current_round.result_count = int(len(all_results))
        ctx.current_round.new_source_ids = list(new_source_ids)
        ctx.runtime.search_calls += int(len(jobs))
        ctx.runtime.fetch_calls += int(per_round_fetch_calls)
        for idx, result in enumerate(all_results, start=1):
            print(
                "[research.search.result]",
                json.dumps(
                    {
                        "round_index": int(ctx.current_round.round_index),
                        "result_index": int(idx),
                        "url": str(result.url),
                        "title": str(result.title or ""),
                        "abstracts_count": int(len(result.abstracts or [])),
                        "content_chars": int(len(str(result.content or ""))),
                        "subpages_count": int(len(result.subpages or [])),
                    },
                    ensure_ascii=False,
                ),
            )
            for sub_idx, sub in enumerate(result.subpages or [], start=1):
                print(
                    "[research.search.subpage]",
                    json.dumps(
                        {
                            "round_index": int(ctx.current_round.round_index),
                            "result_index": int(idx),
                            "subpage_index": int(sub_idx),
                            "url": str(sub.url),
                            "title": str(sub.title or ""),
                            "abstracts_count": int(len(sub.abstracts or [])),
                            "content_chars": int(len(str(sub.content or ""))),
                        },
                        ensure_ascii=False,
                    ),
                )
        print(
            "[research.search]",
            json.dumps(
                {
                    "round_index": int(ctx.current_round.round_index),
                    "queries": list(ctx.current_round.queries),
                    "search_job_count": int(len(jobs)),
                    "result_count": int(len(all_results)),
                    "new_source_ids": list(new_source_ids),
                    "search_calls": int(ctx.runtime.search_calls),
                    "fetch_calls": int(ctx.runtime.fetch_calls),
                },
                ensure_ascii=False,
            ),
        )

        if int(ctx.runtime.fetch_calls) > int(ctx.runtime.budget.max_fetch_calls):
            ctx.errors.append(
                AppError(
                    code="research_fetch_budget_soft_exceeded",
                    message="search pipeline returned more fetched pages than logical budget",
                    details={
                        "fetch_calls": int(ctx.runtime.fetch_calls),
                        "max_fetch_calls": int(ctx.runtime.budget.max_fetch_calls),
                        "round_index": int(ctx.current_round.round_index),
                    },
                )
            )

        span.set_attr("round_index", int(ctx.current_round.round_index))
        span.set_attr("jobs", int(len(jobs)))
        span.set_attr("results", int(len(all_results)))
        span.set_attr("new_source_ids", int(len(new_source_ids)))
        span.set_attr("search_calls", int(ctx.runtime.search_calls))
        span.set_attr("fetch_calls", int(ctx.runtime.fetch_calls))
        return ctx

    def _upsert_source_from_fetch_result(
        self,
        *,
        ctx: ResearchStepContext,
        result: FetchResultItem,
        round_index: int,
    ) -> list[int]:
        created: list[int] = []
        source_id, is_new = self._upsert_source(
            ctx=ctx,
            url=str(result.url),
            title=str(result.title),
            abstracts=list(result.abstracts or []),
            content=str(result.content or ""),
            round_index=round_index,
            is_subpage=False,
        )
        if is_new:
            created.append(source_id)

        for sub in list(result.subpages or []):
            sub_id, sub_is_new = self._upsert_source_from_subpage(
                ctx=ctx,
                sub=sub,
                round_index=round_index,
            )
            if sub_is_new:
                created.append(sub_id)
        return created

    def _upsert_source_from_subpage(
        self,
        *,
        ctx: ResearchStepContext,
        sub: FetchSubpagesResult,
        round_index: int,
    ) -> tuple[int, bool]:
        return self._upsert_source(
            ctx=ctx,
            url=str(sub.url),
            title=str(sub.title),
            abstracts=list(sub.abstracts or []),
            content=str(sub.content or ""),
            round_index=round_index,
            is_subpage=True,
        )

    def _upsert_source(
        self,
        *,
        ctx: ResearchStepContext,
        url: str,
        title: str,
        abstracts: list[str],
        content: str,
        round_index: int,
        is_subpage: bool,
    ) -> tuple[int, bool]:
        existing = ctx.corpus.source_url_to_id.get(url)
        if existing is not None:
            for source in ctx.corpus.sources:
                if source.source_id != existing:
                    continue
                if not source.title and title:
                    source.title = title
                source.abstracts = merge_strings(
                    list(source.abstracts),
                    normalize_strings(abstracts, limit=32),
                    limit=32,
                )
                if not source.content and content:
                    source.content = content
                return existing, False

        source_id = len(ctx.corpus.sources) + 1
        ctx.corpus.sources.append(
            ResearchSource(
                source_id=source_id,
                url=url,
                title=title,
                abstracts=normalize_strings(abstracts, limit=32),
                content=content,
                round_index=round_index,
                is_subpage=bool(is_subpage),
            )
        )
        ctx.corpus.source_url_to_id[url] = source_id
        return source_id, True


__all__ = ["ResearchSearchStep"]
