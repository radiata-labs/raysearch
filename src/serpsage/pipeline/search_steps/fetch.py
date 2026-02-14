from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING
from typing_extensions import override

import anyio

from serpsage.app.request import FetchChunksRequest, FetchRequest
from serpsage.app.response import PageEnrichment
from serpsage.models.pipeline import FetchStepContext, SearchStepContext
from serpsage.pipeline.step import PipelineStep

if TYPE_CHECKING:
    from serpsage.app.response import ResultItem
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.contracts.services import PipelineRunnerBase
    from serpsage.core.runtime import Runtime


class SearchFetchStep(PipelineStep[SearchStepContext]):
    span_name = "step.search_fetch"

    def __init__(
        self,
        *,
        rt: Runtime,
        fetch_runner: PipelineRunnerBase[FetchStepContext],
    ) -> None:
        super().__init__(rt=rt)
        self._fetch_runner = fetch_runner
        self.bind_deps(fetch_runner)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        depth = ctx.request.depth
        span.set_attr("depth", str(depth))
        if depth == "simple":
            return ctx
        if not ctx.results:
            return ctx

        preset = self.settings.search.depth_profiles.get(depth)  # type: ignore[index]
        if preset is None:
            return ctx

        n = len(ctx.results)
        target = int(math.ceil(n * float(preset.pages_ratio)))
        m = max(int(preset.min_pages), min(int(preset.max_pages), target))
        m = min(m, n)
        if m <= 0:
            return ctx

        work = ctx.results[:m]
        query = ctx.request.query
        top_k = int(preset.top_chunks_per_page)
        step_deadline_ts = time.monotonic() + max(0.1, float(preset.step_timeout_s))
        page_timeout_s = float(preset.page_timeout_s)
        max_parallel = min(
            max(1, int(self.settings.fetch.concurrency.global_limit)),
            max(1, m),
        )
        sem = anyio.Semaphore(max_parallel)
        timeout_count = 0
        completed_count = 0
        rendered_count = 0

        async def enrich_one(rank_index: int, r: ResultItem) -> None:
            nonlocal timeout_count, completed_count, rendered_count
            now = time.monotonic()
            if now >= step_deadline_ts:
                r.page = PageEnrichment(
                    chunks=[],
                    markdown="",
                    timing_ms={"total_ms": 0},
                    warnings=["step deadline exceeded"],
                    error="deadline exceeded",
                )
                timeout_count += 1
                return
            async with sem:
                remaining_s = max(0.0, float(step_deadline_ts - time.monotonic()))
                timeout_s = min(float(page_timeout_s), remaining_s)
                if timeout_s <= 0:
                    r.page = PageEnrichment(
                        chunks=[],
                        markdown="",
                        timing_ms={"total_ms": 0},
                        warnings=["step deadline exceeded"],
                        error="deadline exceeded",
                    )
                    timeout_count += 1
                    return

                fetch_ctx = await self._fetch_runner.run(
                    FetchStepContext(
                        settings=self.settings,
                        request=FetchRequest(
                            url=r.url,
                            content=True,
                            profile=ctx.profile_name or None,
                            chunks=FetchChunksRequest(
                                query=query,
                                top_k_chunks=top_k,
                            ),
                            overview=None,
                            params={
                                "timeout_s": timeout_s,
                                "allow_render": bool(
                                    rank_index < int(preset.max_render_pages)
                                ),
                                "rank_index": rank_index,
                            },
                        ),
                    )
                )
                r.page = fetch_ctx.page
                if fetch_ctx.errors and not r.page.error:
                    r.page.error = str(fetch_ctx.errors[0].message)
                completed_count += 1
                if (r.page.fetch_mode or "") == "playwright":
                    rendered_count += 1
                if (r.page.error or "") in {"timeout", "deadline exceeded"}:
                    timeout_count += 1

        async with anyio.create_task_group() as tg:
            for idx, r in enumerate(work):
                tg.start_soon(enrich_one, idx, r)

        span.set_attr("items_considered", int(m))
        span.set_attr("pages_enriched", int(m))
        span.set_attr("pages_completed", int(completed_count))
        span.set_attr("step_budget_s", float(preset.step_timeout_s))
        span.set_attr("page_timeout_s", float(page_timeout_s))
        span.set_attr("pages_timeout", int(timeout_count))
        span.set_attr("pages_rendered", int(rendered_count))
        span.set_attr("deadline_hit", bool(time.monotonic() >= step_deadline_ts))
        return ctx


__all__ = ["SearchFetchStep"]
