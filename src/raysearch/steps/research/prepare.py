from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from raysearch.models.steps.research import (
    GlobalBudget,
    ResearchKnowledge,
    ResearchLimits,
    ResearchResult,
    ResearchRun,
    ResearchStepContext,
    ResearchTask,
)
from raysearch.steps.base import StepBase

if TYPE_CHECKING:
    from raysearch.settings.models import ResearchModeSettings


class ResearchPrepareStep(StepBase[ResearchStepContext]):
    _DEFAULT_SEARCH_MODE = "research"
    _KNOWN_MODES: set[str] = {
        "research-fast",
        "research",
        "research-pro",
    }

    @override
    async def should_run(self, ctx: ResearchStepContext) -> bool:
        """Prepare always runs (first step, initializes context)."""
        _ = ctx
        return True

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        mode = self._normalize_search_mode(ctx.request.search_mode)
        themes = ctx.request.themes.strip()
        profile = self._resolve_profile(mode)
        ctx.request = ctx.request.model_copy(
            update={"search_mode": mode, "themes": themes}
        )
        limits = ResearchLimits(
            mode_key=mode,  # type: ignore[arg-type]
            max_rounds=profile.max_rounds,
            max_search_calls=profile.max_search_calls,
            max_fetch_calls=profile.max_fetch_calls,
            max_results_per_search=profile.max_results_per_search,
            max_queries_per_round=profile.max_queries_per_round,
            max_question_cards_effective=profile.max_question_cards_effective,
            min_rounds_per_track=profile.min_rounds_per_track,
            round_search_budget=profile.round_search_budget,
            round_fetch_budget=profile.round_fetch_budget,
            review_source_window=profile.review_source_window,
            report_source_batch_size=profile.report_source_batch_size,
            report_source_batch_chars=profile.report_source_batch_chars,
            fetch_page_max_chars=profile.fetch_page_max_chars,
            explore_target_pages_per_round=profile.explore_target_pages_per_round,
            explore_links_per_page=profile.explore_links_per_page,
        )
        ctx.task = ResearchTask(
            question=themes,
            style="explainer",
            intent="other",
            complexity="medium",
            input_language="other",
            output_language="other",
            subthemes=[],
            entities=[],
            cards=[],
        )
        ctx.run = ResearchRun(
            mode=mode,  # type: ignore[arg-type]
            limits=limits,
            budget=GlobalBudget(
                total_search=max(1, int(profile.max_search_calls)),
                total_fetch=max(1, int(profile.max_fetch_calls)),
            ),
            stop=False,
            stop_reason="",
            notes=[],
        )
        ctx.knowledge = ResearchKnowledge()
        ctx.result = ResearchResult(content="", structured=None, tracks=[])
        await self.tracker.info(
            name="research.prepare.configured",
            request_id=ctx.request_id,
            step="research.prepare",
            data={
                "success": True,
                "mode": mode,
                "max_rounds": profile.max_rounds,
                "question_cards_limit": profile.max_question_cards_effective,
            },
        )
        await self.tracker.debug(
            name="research.prepare.configured.detail",
            request_id=ctx.request_id,
            step="research.prepare",
            data={
                "success": True,
                "mode": mode,
                "max_rounds": profile.max_rounds,
                "max_search_calls": profile.max_search_calls,
                "max_fetch_calls": profile.max_fetch_calls,
                "question_cards_limit": profile.max_question_cards_effective,
                "min_rounds_per_track": profile.min_rounds_per_track,
                "review_source_window": profile.review_source_window,
                "explore_target_pages_per_round": profile.explore_target_pages_per_round,
                "theme": themes,
            },
        )
        return ctx

    def _normalize_search_mode(self, raw_mode: object | None) -> str:
        token = str(raw_mode or self._DEFAULT_SEARCH_MODE).strip().casefold()
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


__all__ = ["ResearchPrepareStep"]
