from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.contracts.base import WorkUnit

if TYPE_CHECKING:
    from serpsage.domain.normalize import ResultNormalizer
    from serpsage.pipeline.steps import StepContext


class NormalizeStep(WorkUnit):
    def __init__(self, *, rt, normalizer: ResultNormalizer) -> None:  # noqa: ANN001
        super().__init__(rt=rt)
        self._normalizer = normalizer

    async def run(self, ctx: StepContext) -> StepContext:
        with self.span("step.normalize"):
            ctx.results = self._normalizer.normalize_many(ctx.raw_results)
            return ctx


__all__ = ["NormalizeStep"]
