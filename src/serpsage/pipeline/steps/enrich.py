from __future__ import annotations

import math
import time
from typing import TYPE_CHECKING, cast
from typing_extensions import override

import anyio

from serpsage.app.response import PageEnrichment
from serpsage.pipeline.base import StepBase
from serpsage.pipeline.context import SearchStepContext

if TYPE_CHECKING:
    from serpsage.app.response import ResultItem
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime
    from serpsage.domain.enrich import Enricher
    from serpsage.settings.models import ProfileSettings


class EnrichStep(StepBase):
    span_name = "step.enrich"

    def __init__(self, *, rt: Runtime, enricher: Enricher) -> None:
        super().__init__(rt=rt)
        self._enricher = enricher
        self.bind_deps(enricher)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        depth = ctx.request.depth
        span.set_attr("depth", str(depth))
        if depth == "simple":
            return ctx
        if not self.settings.enrich.enabled:
            return ctx
        if not ctx.results:
            return ctx

        preset = self.settings.enrich.depth_presets.get(depth)  # type: ignore[index]
        if preset is None:
            return ctx

        n = len(ctx.results)
        target = int(math.ceil(n * float(preset.pages_ratio)))
        m = max(int(preset.min_pages), min(int(preset.max_pages), target))
        m = min(m, n)
        if m <= 0:
            return ctx

        work = ctx.results[:m]
        span.set_attr("items_considered", int(m))
        query = ctx.request.query
        profile = cast("ProfileSettings", ctx.profile)
        top_k = int(preset.top_chunks_per_page)
        latency_cfg = self.settings.enrich.latency_budgets.get(depth)  # type: ignore[index]
        step_budget_s = float(getattr(latency_cfg, "step_timeout_s", 2.0) or 2.0)
        page_timeout_s = float(getattr(latency_cfg, "page_timeout_s", 1.6) or 1.6)
        step_deadline_ts = time.monotonic() + max(0.1, step_budget_s)
        max_parallel = min(
            max(1, int(self.settings.enrich.fetch.rate_limit.global_concurrency)),
            max(1, m),
        )
        sem = anyio.Semaphore(max_parallel)
        timeout_count = 0

        async def enrich_one(rank_index: int, r: ResultItem) -> None:
            nonlocal timeout_count
            if time.monotonic() >= step_deadline_ts:
                r.page = PageEnrichment(chunks=[], markdown="", error="deadline exceeded")
                timeout_count += 1
                return
            async with sem:
                r.page = await self._enricher.enrich_one(
                    result=r,
                    query=query,
                    query_tokens=ctx.query_tokens or [],
                    intent_tokens=ctx.intent_tokens or [],
                    profile=profile,
                    top_k=top_k,
                    depth=depth,
                    rank_index=rank_index,
                    step_deadline_ts=step_deadline_ts,
                    page_timeout_s=page_timeout_s,
                )
                if (r.page.error or "") in {"timeout", "deadline exceeded"}:
                    timeout_count += 1

        async with anyio.create_task_group() as tg:
            for idx, r in enumerate(work):
                tg.start_soon(enrich_one, idx, r)
        span.set_attr("pages_enriched", int(m))
        span.set_attr("step_budget_s", float(step_budget_s))
        span.set_attr("page_timeout_s", float(page_timeout_s))
        span.set_attr("pages_timeout", int(timeout_count))
        return ctx


__all__ = ["EnrichStep"]
