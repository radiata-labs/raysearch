from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.steps.research import ResearchStepContext
from serpsage.steps.base import StepBase
from serpsage.steps.research.language import map_provider_language_param

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


class ResearchFinalizeStep(StepBase[ResearchStepContext]):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        mode_depth = ctx.runtime.mode_depth
        theme_plan = ctx.plan.theme_plan
        search_language = theme_plan.search_language
        provider_params = map_provider_language_param(
            provider_backend=ctx.settings.provider.backend,
            search_language=search_language,
        )
        provider_language_param_applied = (
            ctx.runtime.provider_language_param_applied
            or (provider_params and ctx.runtime.search_calls > 0)
        )
        content_chars = len(ctx.output.content)
        mode_key = mode_depth.mode_key
        await self.emit_tracking_event(
            event_name="research.finalize.summary",
            request_id=ctx.request_id,
            stage="finalize",
            attrs={
                "stop": ctx.runtime.stop,
                "stop_reason": ctx.runtime.stop_reason or "n/a",
                "content_chars": content_chars,
                "has_structured": ctx.output.structured is not None,
                "report_style_selected": theme_plan.report_style,
                "mode_depth_profile": mode_depth.mode_key,
                "llm_orchestrator_enabled": mode_key != "research-fast",
                "input_language": theme_plan.input_language,
                "output_language": theme_plan.output_language,
                "search_language": search_language,
                "authority_weight_applied": True,
                "explore_resolved_relative_links": (
                    ctx.runtime.explore_resolved_relative_links
                ),
                "provider_language_param_applied": provider_language_param_applied,
            },
        )
        return ctx


__all__ = ["ResearchFinalizeStep"]
