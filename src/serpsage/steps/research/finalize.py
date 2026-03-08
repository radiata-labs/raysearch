from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.steps.research import ResearchStepContext
from serpsage.steps.base import StepBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


class ResearchFinalizeStep(StepBase[ResearchStepContext]):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        mode_depth = ctx.run.limits
        theme_plan = ctx.task
        report_chars = len(ctx.result.content)
        await self.emit_tracking_event(
            event_name="research.finalize.summary",
            request_id=ctx.request_id,
            stage="finalize",
            attrs={
                "stop": ctx.run.stop,
                "stop_reason": ctx.run.stop_reason or "n/a",
                "report_chars": report_chars,
                "has_structured": ctx.result.structured is not None,
                "report_style_selected": theme_plan.style,
                "mode_depth_profile": mode_depth.mode_key,
                "llm_orchestrator_enabled": False,
                "input_language": theme_plan.input_language,
                "output_language": theme_plan.output_language,
                "authority_weight_applied": True,
                "explore_resolved_relative_links": ctx.run.explore_resolved_relative_links,
                "budget_restore_applied": ctx.run.restored_budget_applied,
                "budget_extension_applied": ctx.run.extension_budget_applied,
                "budget_events": len(ctx.run.budget_events),
            },
        )
        return ctx


__all__ = ["ResearchFinalizeStep"]
