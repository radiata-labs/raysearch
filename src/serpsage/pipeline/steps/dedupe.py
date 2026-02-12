from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.pipeline.base import StepBase
from serpsage.pipeline.context import SearchStepContext

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime
    from serpsage.domain.dedupe import Deduper


class DedupeStep(StepBase):
    span_name = "step.dedupe"

    def __init__(self, *, rt: Runtime, deduper: Deduper) -> None:
        super().__init__(rt=rt)
        self._deduper = deduper
        self.bind_deps(deduper)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        span.set_attr("before_count", int(len(ctx.results or [])))
        profile = ctx.profile or self.settings.get_profile(
            self.settings.pipeline.default_profile
        )
        kept, comparisons = self._deduper.dedupe(results=ctx.results, profile=profile)
        ctx.results = kept
        ctx.dedupe_comparisons = int(comparisons)
        span.set_attr("after_count", int(len(ctx.results or [])))
        span.set_attr("comparisons", int(ctx.dedupe_comparisons))
        return ctx


__all__ = ["DedupeStep"]
