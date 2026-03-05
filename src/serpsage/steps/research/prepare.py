from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import (
    ResearchBudgetState,
    ResearchCorpusState,
    ResearchModeDepthState,
    ResearchOutputState,
    ResearchParallelState,
    ResearchPlanState,
    ResearchRoundWorkState,
    ResearchRuntimeState,
    ResearchStepContext,
)
from serpsage.models.research import ResearchThemePlan
from serpsage.steps.base import StepBase
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import (
        ResearchModeDepthProfileSettings,
        ResearchModeSettings,
    )


class ResearchPrepareStep(StepBase[ResearchStepContext]):
    _DEFAULT_SEARCH_MODE = "research"
    _GLOBAL_BUDGET_MULTIPLIER_BY_MODE: dict[str, float] = {
        "research-fast": 1.5,
        "research": 2.0,
        "research-pro": 2.5,
    }
    _KNOWN_MODES: set[str] = {
        "research-fast",
        "research",
        "research-pro",
    }

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        mode = self._normalize_search_mode(ctx.request.search_mode)
        themes = clean_whitespace(ctx.request.themes or "")
        profile = self._resolve_profile(mode)
        mode_depth = self._resolve_mode_depth_profile(mode)
        ctx.request = ctx.request.model_copy(
            update={"search_mode": mode, "themes": themes}
        )
        ctx.runtime = ResearchRuntimeState(
            mode_depth=ResearchModeDepthState(
                mode_key=mode,  # type: ignore[arg-type]
                max_question_cards_effective=mode_depth.max_question_cards_effective,
                min_rounds_per_track=mode_depth.min_rounds_per_track,
                no_progress_rounds_to_stop_effective=mode_depth.no_progress_rounds_to_stop_effective,
                enable_llm_track_orchestrator=mode_depth.enable_llm_track_orchestrator,
                enable_gap_closure_pass=mode_depth.enable_gap_closure_pass,
                gap_closure_passes=mode_depth.gap_closure_passes,
                enable_density_gate=mode_depth.enable_density_gate,
                density_gate_passes=mode_depth.density_gate_passes,
                render_section_min=mode_depth.render_section_min,
                render_section_max=mode_depth.render_section_max,
                overview_source_topk=mode_depth.overview_source_topk,
                content_source_topk=mode_depth.content_source_topk,
                subreport_source_topk=mode_depth.subreport_source_topk,
                content_source_chars=mode_depth.content_source_chars,
                subreport_overview_chars=mode_depth.subreport_overview_chars,
                subreport_excerpt_chars=mode_depth.subreport_excerpt_chars,
                subreport_total_chars=mode_depth.subreport_total_chars,
                target_length_ratio_vs_current=mode_depth.target_length_ratio_vs_current,
                search_links_main_limit=mode_depth.search_links_main_limit,
                explore_target_pages_per_round=mode_depth.explore_target_pages_per_round,
                explore_links_per_page=mode_depth.explore_links_per_page,
                explore_fetch_round_ratio=mode_depth.explore_fetch_round_ratio,
            ),
            budget=ResearchBudgetState(
                max_rounds=profile.max_rounds,
                max_search_calls=profile.max_search_calls,
                max_fetch_calls=profile.max_fetch_calls,
                max_results_per_search=profile.max_results_per_search,
                max_queries_per_round=profile.max_queries_per_round,
                max_fetch_per_round=profile.max_fetch_per_round,
                stop_confidence=profile.stop_confidence,
                min_coverage_ratio=profile.min_coverage_ratio,
                max_unresolved_conflicts=profile.max_unresolved_conflicts,
            ),
            search_calls=0,
            fetch_calls=0,
            no_progress_rounds=0,
            gap_closure_passes_applied=0,
            density_gate_passes_applied=0,
            target_output_chars=0,
            output_length_ratio_vs_target=0.0,
            stop=False,
            stop_reason="",
            round_index=0,
        )
        global_search_budget = max(
            1,
            math.ceil(
                profile.max_search_calls * self._resolve_global_budget_multiplier(mode)
            ),
        )
        global_fetch_budget = max(
            1,
            math.ceil(
                profile.max_fetch_calls * self._resolve_global_budget_multiplier(mode)
            ),
        )
        ctx.plan = ResearchPlanState(
            theme_plan=ResearchThemePlan(core_question=themes),
            next_queries=[themes],
        )
        ctx.parallel = ResearchParallelState(
            question_cards=[],
            track_results=[],
            global_search_budget=global_search_budget,
            global_fetch_budget=global_fetch_budget,
            global_search_used=0,
            global_fetch_used=0,
        )
        ctx.corpus = ResearchCorpusState()
        ctx.work = ResearchRoundWorkState()
        ctx.rounds = []
        ctx.current_round = None
        ctx.notes = []
        ctx.output = ResearchOutputState(content="", structured=None)
        await self.emit_tracking_event(
            event_name="research.progress",
            request_id=ctx.request_id,
            stage="prepare",
            attrs={
                "message": "research.prepare.initialized",
                "mode": mode,
                "max_rounds": profile.max_rounds,
                "max_search_calls": profile.max_search_calls,
                "mode_depth_profile": str(mode),
                "mode_depth_question_cards": mode_depth.max_question_cards_effective,
                "mode_depth_min_rounds_per_track": mode_depth.min_rounds_per_track,
                "mode_depth_orchestrator_enabled": mode_depth.enable_llm_track_orchestrator,
                "theme": themes,
            },
        )
        await self.emit_tracking_event(
            event_name="research.mode_depth.selected",
            request_id=ctx.request_id,
            stage="prepare",
            attrs={
                "mode_depth_profile": str(mode),
                "llm_orchestrator_enabled": mode_depth.enable_llm_track_orchestrator,
                "gap_closure_passes": mode_depth.gap_closure_passes,
                "density_gate_passes": mode_depth.density_gate_passes,
            },
        )
        return ctx

    def _normalize_search_mode(self, raw_mode: object | None) -> str:
        token = clean_whitespace(str(raw_mode or self._DEFAULT_SEARCH_MODE)).casefold()
        if token in self._KNOWN_MODES:
            return token
        return self._DEFAULT_SEARCH_MODE

    def _resolve_profile(self, mode: str) -> ResearchModeSettings:
        profiles: dict[str, ResearchModeSettings] = {
            "research-fast": self.settings.research.research_fast,
            "research": self.settings.research.research,
            "research-pro": self.settings.research.research_pro,
        }
        return profiles.get(mode, self.settings.research.research)

    def _resolve_mode_depth_profile(
        self, mode: str
    ) -> ResearchModeDepthProfileSettings:
        profiles: dict[str, ResearchModeDepthProfileSettings] = {
            "research-fast": self.settings.research.mode_depth.research_fast,
            "research": self.settings.research.mode_depth.research,
            "research-pro": self.settings.research.mode_depth.research_pro,
        }
        return profiles.get(mode, self.settings.research.mode_depth.research)

    def _resolve_global_budget_multiplier(self, mode: str) -> float:
        token = clean_whitespace(mode).casefold()
        return self._GLOBAL_BUDGET_MULTIPLIER_BY_MODE.get(token, 2.0)


__all__ = ["ResearchPrepareStep"]
