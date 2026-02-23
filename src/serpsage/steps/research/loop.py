from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import ResearchStepContext
from serpsage.steps.base import StepBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.steps.base import RunnerBase
    from serpsage.telemetry.base import SpanBase


class ResearchLoopStep(StepBase[ResearchStepContext]):
    span_name = "step.research_loop"

    def __init__(
        self,
        *,
        rt: Runtime,
        round_runner: RunnerBase[ResearchStepContext],
    ) -> None:
        super().__init__(rt=rt)
        self._round_runner = round_runner
        self.bind_deps(round_runner)

    @override
    async def run_inner(
        self, ctx: ResearchStepContext, *, span: SpanBase
    ) -> ResearchStepContext:
        max_rounds = int(ctx.runtime.budget.max_rounds)
        while not ctx.runtime.stop and len(ctx.rounds) < max_rounds:
            before = int(len(ctx.rounds))
            ctx = await self._round_runner.run(ctx)
            if int(len(ctx.rounds)) <= before:
                ctx.runtime.stop = True
                ctx.runtime.stop_reason = "round_stalled"
                break
            if ctx.current_round is None:
                ctx.runtime.stop = True
                ctx.runtime.stop_reason = "round_missing"
                break
            if ctx.current_round.stop:
                break

        span.set_attr("rounds", int(len(ctx.rounds)))
        span.set_attr("search_calls", int(ctx.runtime.search_calls))
        span.set_attr("fetch_calls", int(ctx.runtime.fetch_calls))
        span.set_attr("stop", bool(ctx.runtime.stop))
        span.set_attr("stop_reason", str(ctx.runtime.stop_reason or ""))
        return ctx


__all__ = ["ResearchLoopStep"]

