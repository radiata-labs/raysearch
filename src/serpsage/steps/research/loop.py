from __future__ import annotations

import json
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
            print(
                "[research.loop] round_start",
                json.dumps(
                    {
                        "next_round_index": int(ctx.runtime.round_index) + 1,
                        "search_calls": int(ctx.runtime.search_calls),
                        "fetch_calls": int(ctx.runtime.fetch_calls),
                        "planned_queries": list(ctx.plan.next_queries),
                    },
                    ensure_ascii=False,
                ),
            )
            before = int(len(ctx.rounds))
            ctx = await self._round_runner.run(ctx)
            if ctx.current_round is not None:
                print(
                    "[research.loop] round_end",
                    json.dumps(
                        {
                            "round_index": int(ctx.current_round.round_index),
                            "queries": list(ctx.current_round.queries),
                            "result_count": int(ctx.current_round.result_count),
                            "new_source_ids": list(ctx.current_round.new_source_ids),
                            "confidence": float(ctx.current_round.confidence),
                            "coverage_ratio": float(ctx.current_round.coverage_ratio),
                            "unresolved_conflicts": int(
                                ctx.current_round.unresolved_conflicts
                            ),
                            "critical_gaps": int(ctx.current_round.critical_gaps),
                            "stop": bool(ctx.current_round.stop),
                            "stop_reason": str(ctx.current_round.stop_reason or ""),
                        },
                        ensure_ascii=False,
                    ),
                )
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
        print(
            "[research.loop] finished",
            json.dumps(
                {
                    "rounds": int(len(ctx.rounds)),
                    "search_calls": int(ctx.runtime.search_calls),
                    "fetch_calls": int(ctx.runtime.fetch_calls),
                    "stop": bool(ctx.runtime.stop),
                    "stop_reason": str(ctx.runtime.stop_reason or ""),
                },
                ensure_ascii=False,
            ),
        )
        return ctx


__all__ = ["ResearchLoopStep"]
