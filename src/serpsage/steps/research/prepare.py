from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import (
    ResearchBudgetState,
    ResearchCorpusState,
    ResearchOutputState,
    ResearchPlanState,
    ResearchRoundWorkState,
    ResearchRuntimeState,
    ResearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import ResearchModeSettings
    from serpsage.telemetry.base import SpanBase


class ResearchPrepareStep(StepBase[ResearchStepContext]):
    span_name = "step.research_prepare"

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: ResearchStepContext, *, span: SpanBase
    ) -> ResearchStepContext:
        mode = str(ctx.request.search_mode or "research")
        themes = clean_whitespace(ctx.request.themes or "")
        profile = self._resolve_profile(mode)

        ctx.request = ctx.request.model_copy(update={"search_mode": mode, "themes": themes})
        ctx.runtime = ResearchRuntimeState(
            budget=ResearchBudgetState(
                max_rounds=int(profile.max_rounds),
                max_search_calls=int(profile.max_search_calls),
                max_fetch_calls=int(profile.max_fetch_calls),
                max_results_per_search=int(profile.max_results_per_search),
                max_queries_per_round=int(profile.max_queries_per_round),
                max_fetch_per_round=int(profile.max_fetch_per_round),
                stop_confidence=float(profile.stop_confidence),
                min_coverage_ratio=float(profile.min_coverage_ratio),
                max_unresolved_conflicts=int(profile.max_unresolved_conflicts),
            ),
            search_calls=0,
            fetch_calls=0,
            no_progress_rounds=0,
            stop=False,
            stop_reason="",
            round_index=0,
        )
        ctx.plan = ResearchPlanState(theme_plan={}, next_queries=[themes])
        ctx.corpus = ResearchCorpusState()
        ctx.work = ResearchRoundWorkState()
        ctx.rounds = []
        ctx.current_round = None
        ctx.notes = []
        ctx.output = ResearchOutputState(content="", structured=None)

        span.set_attr("search_mode", mode)
        span.set_attr("themes_chars", int(len(themes)))
        span.set_attr("max_rounds", int(ctx.runtime.budget.max_rounds))
        span.set_attr("max_search_calls", int(ctx.runtime.budget.max_search_calls))
        span.set_attr("max_fetch_calls", int(ctx.runtime.budget.max_fetch_calls))
        return ctx

    def _resolve_profile(self, mode: str) -> ResearchModeSettings:
        if mode == "research-fast":
            return self.settings.research.research_fast
        if mode == "research-pro":
            return self.settings.research.research_pro
        return self.settings.research.research


__all__ = ["ResearchPrepareStep"]

