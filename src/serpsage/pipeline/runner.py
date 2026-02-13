from __future__ import annotations

from typing import TYPE_CHECKING, Generic
from typing_extensions import override

from serpsage.contracts.services import PipelineRunnerBase, PipelineStepBase, TContext

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


class PipelineRunner(PipelineRunnerBase[TContext], Generic[TContext]):
    def __init__(self, *, rt: Runtime, steps: list[PipelineStepBase[TContext]]) -> None:
        super().__init__(rt=rt)
        self._steps = list(steps)
        self.bind_deps(*steps)

    @override
    async def run(self, ctx: TContext) -> TContext:
        for step in self._steps:
            ctx = await step.run(ctx)
        return ctx


__all__ = ["PipelineRunner"]
