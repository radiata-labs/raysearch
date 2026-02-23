from __future__ import annotations

import json
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import ResearchStepContext
from serpsage.steps.base import StepBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class ResearchFinalizeStep(StepBase[ResearchStepContext]):
    span_name = "step.research_finalize"

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: ResearchStepContext, *, span: SpanBase
    ) -> ResearchStepContext:
        span.set_attr("rounds", int(len(ctx.rounds)))
        span.set_attr("search_calls", int(ctx.runtime.search_calls))
        span.set_attr("fetch_calls", int(ctx.runtime.fetch_calls))
        span.set_attr("sources", int(len(ctx.corpus.sources)))
        span.set_attr("stop", bool(ctx.runtime.stop))
        span.set_attr("stop_reason", str(ctx.runtime.stop_reason or ""))
        span.set_attr("has_content", bool(ctx.output.content))
        span.set_attr("has_structured", bool(ctx.output.structured is not None))
        print(
            "[research.finalize]",
            json.dumps(
                {
                    "rounds": int(len(ctx.rounds)),
                    "search_calls": int(ctx.runtime.search_calls),
                    "fetch_calls": int(ctx.runtime.fetch_calls),
                    "sources": int(len(ctx.corpus.sources)),
                    "stop": bool(ctx.runtime.stop),
                    "stop_reason": str(ctx.runtime.stop_reason or ""),
                    "errors": [item.model_dump() for item in ctx.errors],
                    "content": str(ctx.output.content),
                    "structured": ctx.output.structured,
                },
                ensure_ascii=False,
            ),
        )
        return ctx


__all__ = ["ResearchFinalizeStep"]

