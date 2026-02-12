from __future__ import annotations

from typing import TYPE_CHECKING, cast
from typing_extensions import override

from serpsage.pipeline.base import StepBase
from serpsage.pipeline.context import SearchStepContext

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime
    from serpsage.domain.dedupe import Deduper
    from serpsage.settings.models import ProfileSettings


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
        kept, comparisons = self._deduper.dedupe(
            results=ctx.results, profile=cast("ProfileSettings", ctx.profile)
        )
        ctx.results = kept
        span.set_attr("after_count", int(len(ctx.results or [])))
        span.set_attr("comparisons", int(comparisons))
        return ctx


__all__ = ["DedupeStep"]
