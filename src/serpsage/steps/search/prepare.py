from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import SearchStepContext
from serpsage.steps.base import StepBase
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class SearchPrepareStep(StepBase[SearchStepContext]):
    span_name = "step.search_prepare"

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        query = clean_whitespace(ctx.request.query or "")
        depth = ctx.request.depth or "simple"
        max_results = (
            int(ctx.request.max_results)
            if ctx.request.max_results is not None
            else int(self.settings.search.max_results)
        )
        max_results = max(1, int(max_results))
        ctx.request = ctx.request.model_copy(
            update={
                "query": query,
                "depth": depth,
                "max_results": max_results,
            }
        )
        span.set_attr("depth", str(depth))
        span.set_attr("max_results", int(max_results))
        return ctx


__all__ = ["SearchPrepareStep"]
