from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.contracts.base import WorkUnit

if TYPE_CHECKING:
    from serpsage.domain.dedupe import ResultDeduper
    from serpsage.pipeline.steps import StepContext


class DedupeStep(WorkUnit):
    def __init__(self, *, rt, deduper: ResultDeduper) -> None:  # noqa: ANN001
        super().__init__(rt=rt)
        self._deduper = deduper

    async def run(self, ctx: StepContext) -> StepContext:
        with self.span("step.dedupe"):
            profile = ctx.profile or self.settings.get_profile(
                self.settings.pipeline.default_profile
            )
            kept, comparisons = self._deduper.dedupe(
                results=ctx.results, profile=profile
            )
            ctx.results = kept
            ctx.scratch["dedupe_comparisons"] = comparisons
            return ctx


__all__ = ["DedupeStep"]
