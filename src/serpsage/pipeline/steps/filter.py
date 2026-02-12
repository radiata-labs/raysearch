from __future__ import annotations

from typing import TYPE_CHECKING, cast
from typing_extensions import override

from serpsage.pipeline.base import StepBase
from serpsage.pipeline.context import SearchStepContext

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime
    from serpsage.domain.filter import Filterer
    from serpsage.settings.models import ProfileSettings


class FilterStep(StepBase):
    span_name = "step.filter"

    def __init__(self, *, rt: Runtime, filterer: Filterer) -> None:
        super().__init__(rt=rt)
        self._filterer = filterer
        self.bind_deps(filterer)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        span.set_attr("before_count", int(len(ctx.results or [])))
        ctx.results = self._filterer.filter(
            ctx.results,
            query_tokens=ctx.query_tokens or [],
            profile=cast("ProfileSettings", ctx.profile),
        )
        span.set_attr("after_count", int(len(ctx.results or [])))
        return ctx


__all__ = ["FilterStep"]
