from __future__ import annotations

from typing_extensions import override

from raysearch.models.steps.research import ResearchStepContext
from raysearch.steps.base import StepBase


class ResearchFinalizeStep(StepBase[ResearchStepContext]):
    @override
    async def should_run(self, ctx: ResearchStepContext) -> bool:
        """Finalize always runs (last step, emits telemetry)."""
        _ = ctx
        return True

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        mode_depth = ctx.run.limits
        theme_plan = ctx.task
        report_chars = len(ctx.result.content)
        await self.tracker.info(
            name="research.finalize.summary",
            request_id=ctx.request_id,
            step="research.finalize",
            data={
                "success": True,
                "stop": ctx.run.stop,
                "stop_reason": ctx.run.stop_reason or "n/a",
                "report_chars": report_chars,
                "has_structured": ctx.result.structured is not None,
            },
        )
        await self.tracker.debug(
            name="research.finalize.summary.detail",
            request_id=ctx.request_id,
            step="research.finalize",
            data={
                "success": True,
                "report_style_selected": theme_plan.style,
                "mode_depth_profile": mode_depth.mode_key,
                "llm_orchestrator_enabled": False,
                "input_language": theme_plan.input_language,
                "output_language": theme_plan.output_language,
                "authority_weight_applied": True,
                "explore_resolved_relative_links": ctx.run.explore_resolved_relative_links,
                "budget_tier": ctx.run.budget.tier,
                "track_count": len(ctx.run.track_runtimes),
            },
        )
        return ctx


__all__ = ["ResearchFinalizeStep"]
