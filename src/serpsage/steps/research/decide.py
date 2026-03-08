from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.steps.research import (
    ResearchDecideSignalPayload,
    ResearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import build_decide_prompt_messages
from serpsage.steps.research.utils import resolve_research_model

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime


class ResearchDecideStep(StepBase[ResearchStepContext]):
    _LOW_GAIN_THRESHOLD = 0.05

    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        if ctx.current_round is None:
            return ctx
        budget = ctx.runtime.budget
        round_state = ctx.current_round
        overview_review = ctx.work.overview_review
        content_review = ctx.work.content_review
        overview_stop = (
            bool(overview_review.stop) if overview_review is not None else False
        )
        content_stop = (
            bool(content_review.stop) if content_review is not None else False
        )
        model_stop = (
            overview_stop or content_stop or round_state.query_strategy == "stop-ready"
        )
        confidence_ok = round_state.confidence >= budget.stop_confidence
        coverage_ok = round_state.coverage_ratio >= budget.min_coverage_ratio
        gaps_ok = round_state.critical_gaps == 0
        entity_coverage_ok = (
            round_state.entity_coverage_complete
            if ctx.plan.theme_plan.required_entities
            else True
        )
        unresolved_conflict_topics = list(round_state.unresolved_conflict_topics)
        multi_signal_stop = (
            model_stop
            and confidence_ok
            and coverage_ok
            and gaps_ok
            and entity_coverage_ok
            and not unresolved_conflict_topics
        )
        previous_round = ctx.rounds[-1] if ctx.rounds else None
        current_low_gain = round_state.result_count <= 0 or (
            round_state.corpus_score_gain < float(self._LOW_GAIN_THRESHOLD)
        )
        if previous_round is None:
            low_gain_streak = 1 if current_low_gain else 0
            progress = (
                bool(round_state.new_source_ids) or round_state.corpus_score_gain > 0.0
            )
        else:
            low_gain_streak = (
                previous_round.low_gain_streak + 1 if current_low_gain else 0
            )
            progress = bool(round_state.new_source_ids) or (
                round_state.coverage_ratio > previous_round.coverage_ratio
                or round_state.unresolved_conflicts
                < previous_round.unresolved_conflicts
                or round_state.corpus_score_gain > 0.0
            )
        llm_signal = await self._query_decide_signal(ctx=ctx)
        llm_prefers_continue = (
            llm_signal.continue_research or llm_signal.high_yield_remaining
        )
        next_queries = list(llm_signal.next_queries[: budget.max_queries_per_round])
        gap_objectives = [
            f"gap:{item}"
            for item in [
                *(overview_review.critical_gaps if overview_review is not None else []),
                *(content_review.remaining_gaps if content_review is not None else []),
            ]
        ]
        entity_objectives = [
            f"missing_entity:{item}" for item in round_state.missing_entities
        ]
        conflict_objectives = [
            f"conflict:{item}" for item in unresolved_conflict_topics
        ]
        query_objectives = [f"query:{item}" for item in next_queries]
        remaining_objectives = self._merge_preserving_order(
            [
                *gap_objectives,
                *entity_objectives,
                *conflict_objectives,
                *query_objectives,
            ]
        )
        stop_ready = multi_signal_stop and not remaining_objectives
        if stop_ready:
            next_queries = []
            query_objectives = []
            remaining_objectives = self._merge_preserving_order(
                [*gap_objectives, *entity_objectives, *conflict_objectives]
            )
        search_exhausted = ctx.runtime.search_calls >= budget.max_search_calls
        fetch_exhausted = ctx.runtime.fetch_calls >= budget.max_fetch_calls
        can_search_now = (not search_exhausted) and (not fetch_exhausted)
        can_explore_without_search = self._can_continue_with_explore_only(ctx=ctx)
        min_rounds_per_track = max(1, ctx.runtime.mode_depth.min_rounds_per_track)
        must_continue_for_min_rounds = (len(ctx.rounds) + 1) < min_rounds_per_track
        can_execute_next_round = (
            can_search_now and bool(next_queries)
        ) or can_explore_without_search
        stop = False
        stop_reason = ""
        if fetch_exhausted:
            stop = True
            stop_reason = "max_fetch_calls"
        elif search_exhausted and not can_explore_without_search:
            stop = True
            stop_reason = "max_search_calls"
        elif must_continue_for_min_rounds and (
            can_search_now or can_explore_without_search
        ):
            stop = False
        elif stop_ready:
            stop = True
            stop_reason = "stop_ready"
        elif (
            low_gain_streak >= 2
            and not gap_objectives
            and not entity_objectives
            and not conflict_objectives
            and not llm_prefers_continue
        ):
            stop = True
            stop_reason = "low_gain_stalled"
        elif not can_execute_next_round and not progress:
            stop = True
            stop_reason = "no_executable_path"
        round_state.stop_ready = stop_ready
        round_state.remaining_objectives = remaining_objectives
        round_state.low_gain_streak = low_gain_streak
        round_state.unresolved_conflict_topics = unresolved_conflict_topics
        round_state.stop = stop
        round_state.stop_reason = stop_reason
        ctx.plan.next_queries = [] if stop else next_queries
        ctx.rounds.append(round_state)
        if llm_signal.reason:
            ctx.notes.append(f"Decide signal: {llm_signal.reason}")
        await self.emit_tracking_event(
            event_name="research.decide.summary",
            request_id=ctx.request_id,
            stage="decide",
            attrs={
                "llm_prefers_continue": llm_prefers_continue,
                "min_rounds_per_track": min_rounds_per_track,
                "must_continue_for_min_rounds": must_continue_for_min_rounds,
                "stop_ready": stop_ready,
                "can_search_now": can_search_now,
                "can_explore_without_search": can_explore_without_search,
                "can_execute_next_round": can_execute_next_round,
                "next_queries": len(next_queries),
                "remaining_objectives_count": len(remaining_objectives),
                "low_gain_streak": low_gain_streak,
                "stop": stop,
                "stop_reason": str(stop_reason or "n/a"),
            },
        )
        if stop:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = stop_reason
        return ctx

    def _can_continue_with_explore_only(self, *, ctx: ResearchStepContext) -> bool:
        if ctx.current_round is None:
            return False
        if ctx.runtime.budget.max_fetch_calls <= ctx.runtime.fetch_calls:
            return False
        return (
            ctx.plan.last_round_link_candidates_round == ctx.current_round.round_index
            and bool(ctx.plan.last_round_link_candidates)
        )

    async def _query_decide_signal(
        self,
        *,
        ctx: ResearchStepContext,
    ) -> ResearchDecideSignalPayload:
        model = resolve_research_model(
            ctx=ctx,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        try:
            result = await self._llm.create(
                model=model,
                messages=build_decide_prompt_messages(ctx=ctx),
                response_format=ResearchDecideSignalPayload,
                retries=self.settings.research.llm_self_heal_retries,
            )
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="research.decide.error",
                request_id=ctx.request_id,
                stage="decide",
                status="error",
                error_code="research_decide_signal_failed",
                error_type=type(exc).__name__,
                attrs={
                    "model": str(model),
                    "message": str(exc),
                },
            )
            raise
        return result.data

    def _merge_preserving_order(self, values: list[str]) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for item in values:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged


__all__ = ["ResearchDecideStep"]
