from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.contracts.base import WorkUnit

if TYPE_CHECKING:
    from serpsage.domain.filter import ResultFilterer
    from serpsage.pipeline.steps import StepContext


class FilterStep(WorkUnit):
    def __init__(self, *, rt, filterer: ResultFilterer) -> None:  # noqa: ANN001
        super().__init__(rt=rt)
        self._filterer = filterer

    async def run(self, ctx: StepContext) -> StepContext:
        with self.span("step.filter"):
            outcome = self._filterer.filter(
                query=ctx.request.query,
                explicit_profile=ctx.request.profile,
                results=ctx.results,
            )
            ctx.profile_name = outcome.profile_name
            ctx.profile = outcome.profile
            ctx.scratch["query_tokens"] = outcome.query_tokens
            ctx.results = outcome.results
            return ctx


__all__ = ["FilterStep"]
