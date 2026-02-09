from __future__ import annotations

import math
from typing import TYPE_CHECKING

import anyio

from serpsage.contracts.base import WorkUnit
from serpsage.text.tokenize import tokenize

if TYPE_CHECKING:
    from serpsage.domain.enrich import Enricher
    from serpsage.pipeline.steps import StepContext


class EnrichStep(WorkUnit):
    def __init__(self, *, rt, enricher: Enricher) -> None:  # noqa: ANN001
        super().__init__(rt=rt)
        self._enricher = enricher

    async def run(self, ctx: StepContext) -> StepContext:
        with self.span("step.enrich"):
            depth = ctx.request.depth
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
            query = ctx.request.query
            query_tokens = list(ctx.scratch.get("query_tokens") or tokenize(query))
            profile = ctx.profile or self.settings.get_profile(
                self.settings.pipeline.default_profile
            )
            top_k = int(preset.top_chunks_per_page)

            async def enrich_one(r) -> None:  # noqa: ANN001
                r.page = await self._enricher.enrich_one(
                    result=r,
                    query=query,
                    query_tokens=query_tokens,
                    profile=profile,
                    top_k=top_k,
                )

            async with anyio.create_task_group() as tg:
                for r in work:
                    tg.start_soon(enrich_one, r)
            return ctx


__all__ = ["EnrichStep"]
