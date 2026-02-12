from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.pipeline.base import StepBase
from serpsage.pipeline.context import SearchStepContext

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime
    from serpsage.domain.normalize import Normalizer


class NormalizeStep(StepBase):
    span_name = "step.normalize"

    def __init__(self, *, rt: Runtime, normalizer: Normalizer) -> None:
        super().__init__(rt=rt)
        self._normalizer = normalizer
        self.bind_deps(normalizer)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        span.set_attr("raw_results_count", int(len(ctx.raw_results or [])))
        ctx.results = self._normalizer.normalize_many(ctx.raw_results)
        span.set_attr("results_count", int(len(ctx.results or [])))
        return ctx


__all__ = ["NormalizeStep"]
