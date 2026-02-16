from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING
from typing_extensions import override

import anyio

from serpsage.app.request import FetchAbstractsRequest, FetchRequest
from serpsage.app.response import PageAbstract, PageEnrichment
from serpsage.models.pipeline import (
    FetchStepContext,
    FetchStepOthers,
    SearchStepContext,
)
from serpsage.steps.base import StepBase

if TYPE_CHECKING:
    from serpsage.app.response import ResultItem
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
        top_k = int(preset.top_abstracts_per_page)
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

        async def enrich_one(r: ResultItem) -> None:
            nonlocal timeout_count, completed_count, rendered_count
            now = time.monotonic()
            if now >= step_deadline_ts:
                r.page = PageEnrichment(
                    abstracts=[],
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
                        abstracts=[],
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
                            urls=[r.url],
                            crawl_mode="fallback",
                            crawl_timeout=timeout_s,
                            content=True,
                            abstracts=FetchAbstractsRequest(
                                query=query,
                                top_k_abstracts=top_k,
                            ),
                            overview=None,
                        ),
                        url=r.url,
                        url_index=0,
                        others=FetchStepOthers(
                            crawl_mode="fallback",
                            crawl_timeout_s=timeout_s,
                            max_links=None,
                            max_image_links=None,
                        ),
                    )
                )
                if fetch_ctx.result is not None:
                    abstracts = [
                        PageAbstract(text=text, score=float(score))
                        for text, score in zip(
                            fetch_ctx.result.abstracts,
                            fetch_ctx.result.abstract_scores,
                            strict=False,
                        )
                    ]
                    r.page = PageEnrichment(
                        abstracts=abstracts,
                        markdown=fetch_ctx.result.content,
                    )
                    if fetch_ctx.errors:
                        r.page.error = str(fetch_ctx.errors[0].message)
                else:
                    err_msg = (
                        str(fetch_ctx.errors[0].message)
                        if fetch_ctx.errors
                        else "fetch failed"
                    )
                    r.page = PageEnrichment(
                        abstracts=[],
                        markdown="",
                        timing_ms={"total_ms": 0},
                        warnings=["fetch failed"],
                        error=err_msg,
                    )
                completed_count += 1
                if (
                    fetch_ctx.fetch_result and fetch_ctx.fetch_result.fetch_mode
                ) == "playwright":
                    rendered_count += 1
                if (r.page.error or "") in {"timeout", "deadline exceeded"}:
                    timeout_count += 1

        async with anyio.create_task_group() as tg:
            for r in work:
                tg.start_soon(enrich_one, r)

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
