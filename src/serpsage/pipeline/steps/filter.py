from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.pipeline.base import StepBase
from serpsage.pipeline.context import SearchStepContext

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime
    from serpsage.domain.filter import ResultFilterer


class FilterStep(StepBase):
    span_name = "step.filter"

    def __init__(self, *, rt: Runtime, filterer: ResultFilterer) -> None:
        super().__init__(rt=rt)
        self._filterer = filterer
        self.bind_deps(filterer)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        span.set_attr("before_count", int(len(ctx.results or [])))
        outcome = self._filterer.filter(
            query=ctx.request.query,
            explicit_profile=ctx.request.profile,
            results=ctx.results,
        )
        ctx.profile_name = outcome.profile_name
        ctx.profile = outcome.profile
        ctx.query_tokens = list(outcome.query_tokens)
        ctx.results = outcome.results
        span.set_attr("profile_name", str(ctx.profile_name or ""))
        span.set_attr("after_count", int(len(ctx.results or [])))
        return ctx


__all__ = ["FilterStep"]
