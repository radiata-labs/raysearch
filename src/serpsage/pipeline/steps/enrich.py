from __future__ import annotations

import math
from typing import TYPE_CHECKING, cast
from typing_extensions import override

import anyio

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

        async def enrich_one(r: ResultItem) -> None:
            r.page = await self._enricher.enrich_one(
                result=r,
                query=query,
                query_tokens=ctx.query_tokens or [],
                intent_tokens=ctx.intent_tokens or [],
                profile=profile,
                top_k=top_k,
            )

        async with anyio.create_task_group() as tg:
            for r in work:
                tg.start_soon(enrich_one, r)
        span.set_attr("pages_enriched", int(m))
        return ctx


__all__ = ["EnrichStep"]
