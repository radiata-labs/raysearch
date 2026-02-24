from __future__ import annotations

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
        span.set_attr("stop", bool(ctx.runtime.stop))
        span.set_attr("stop_reason", str(ctx.runtime.stop_reason or ""))
        span.set_attr("has_content", bool(ctx.output.content))
        span.set_attr("has_structured", bool(ctx.output.structured is not None))
        print(
            (
                "[research][finalize] "
                f"request_id={ctx.request_id} "
                f"stop={bool(ctx.runtime.stop)} "
                f"stop_reason={str(ctx.runtime.stop_reason or 'n/a')} "
                f"content_chars={int(len(str(ctx.output.content or '')))} "
                f"has_structured={bool(ctx.output.structured is not None)} "
                f"errors={int(len(ctx.errors))}"
            ),
            flush=True,
        )
        return ctx


__all__ = ["ResearchFinalizeStep"]
