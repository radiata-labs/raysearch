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
    from serpsage.settings.models import ResearchModeSettings


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
        ctx.request = ctx.request.model_copy(
            update={"search_mode": mode, "themes": themes}
        )
        ctx.runtime = ResearchRuntimeState(
            mode_depth=ResearchModeDepthState(
                mode_key=mode,  # type: ignore[arg-type]
                max_question_cards_effective=profile.max_question_cards_effective,
                min_rounds_per_track=profile.min_rounds_per_track,
                no_progress_rounds_to_stop_effective=profile.no_progress_rounds_to_stop_effective,
                gap_closure_passes=profile.gap_closure_passes,
                density_gate_passes=profile.density_gate_passes,
                overview_source_topk=profile.overview_source_topk,
                content_source_topk=profile.content_source_topk,
                content_source_chars=profile.content_source_chars,
                explore_target_pages_per_round=profile.explore_target_pages_per_round,
                explore_links_per_page=profile.explore_links_per_page,
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
                "mode_depth_question_cards": profile.max_question_cards_effective,
                "mode_depth_min_rounds_per_track": profile.min_rounds_per_track,
                "mode_depth_orchestrator_enabled": self._mode_uses_orchestrator(mode),
                "theme": themes,
            },
        )
        await self.emit_tracking_event(
            event_name="research.mode_depth.selected",
            request_id=ctx.request_id,
            stage="prepare",
            attrs={
                "mode_depth_profile": str(mode),
                "llm_orchestrator_enabled": self._mode_uses_orchestrator(mode),
                "gap_closure_passes": profile.gap_closure_passes,
                "density_gate_passes": profile.density_gate_passes,
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

    def _resolve_global_budget_multiplier(self, mode: str) -> float:
        token = clean_whitespace(mode).casefold()
        return self._GLOBAL_BUDGET_MULTIPLIER_BY_MODE.get(token, 2.0)

    def _mode_uses_orchestrator(self, mode: str) -> bool:
        mode_name = clean_whitespace(mode).casefold()
        return mode_name != "research-fast"


__all__ = ["ResearchPrepareStep"]
