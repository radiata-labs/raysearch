from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import ResearchStepContext
from serpsage.steps.base import StepBase
from serpsage.steps.research.language import map_provider_language_param

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


class ResearchFinalizeStep(StepBase[ResearchStepContext]):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        style_cfg = ctx.settings.research.report_style
        mode_depth = ctx.runtime.mode_depth
        theme_plan = ctx.plan.theme_plan
        search_language = str(theme_plan.search_language or "")
        provider_params = map_provider_language_param(
            provider_backend=str(ctx.settings.provider.backend),
            search_language=search_language,
        )
        provider_language_param_applied = (
            ctx.runtime.provider_language_param_applied
            or (bool(provider_params) and ctx.runtime.search_calls > 0)
        )
        content_chars = len(str(ctx.output.content or ""))
        mode_key = mode_depth.mode_key
        await self.emit_tracking_event(
            event_name="research.finalize.summary",
            request_id=ctx.request_id,
            stage="finalize",
            attrs={
                "stop": ctx.runtime.stop,
                "stop_reason": str(ctx.runtime.stop_reason or "n/a"),
                "content_chars": content_chars,
                "has_structured": ctx.output.structured is not None,
                "report_style_selected": str(theme_plan.report_style or ""),
                "report_style_enabled": style_cfg.enabled,
                "report_style_apply_subreport": style_cfg.apply_subreport,
                "report_style_apply_render": style_cfg.apply_render,
                "mode_depth_profile": str(mode_depth.mode_key),
                "density_gate_passes_applied": ctx.runtime.density_gate_passes_applied,
                "gap_closure_passes_applied": ctx.runtime.gap_closure_passes_applied,
                "llm_orchestrator_enabled": mode_key != "research-fast",
                "input_language": str(theme_plan.input_language),
                "output_language": str(theme_plan.output_language),
                "search_language": str(search_language),
                "provider_language_param_applied": provider_language_param_applied,
            },
        )
        return ctx


__all__ = ["ResearchFinalizeStep"]
