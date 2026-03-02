from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import ResearchStepContext
from serpsage.steps.base import StepBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


class ResearchFinalizeStep(StepBase[ResearchStepContext]):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        style_cfg = ctx.settings.research.report_style
        await self.emit_tracking_event(
            event_name="research.finalize.summary",
            request_id=ctx.request_id,
            stage="finalize",
            attrs={
                "stop": bool(ctx.runtime.stop),
                "stop_reason": str(ctx.runtime.stop_reason or "n/a"),
                "content_chars": int(len(str(ctx.output.content or ""))),
                "has_structured": bool(ctx.output.structured is not None),
                "report_style_selected": str(ctx.plan.theme_plan.report_style or ""),
                "report_style_enabled": bool(style_cfg.enabled),
                "report_style_apply_subreport": bool(style_cfg.apply_subreport),
                "report_style_apply_render": bool(style_cfg.apply_render),
            },
        )
        return ctx


__all__ = ["ResearchFinalizeStep"]
