from __future__ import annotations

from typing_extensions import override

from serpsage.components.llm.base import LLMClientBase
from serpsage.components.provider.base import SearchProviderBase
from serpsage.components.provider.blend import (
    build_engine_selection_context,
    resolve_engine_selection_routes,
)
from serpsage.dependencies import Depends
from serpsage.models.steps.research import (
    ResearchDecideSignalPayload,
    ResearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import build_decide_prompt_messages
from serpsage.steps.research.schema import build_decide_schema
from serpsage.steps.research.utils import resolve_research_model


class ResearchDecideStep(StepBase[ResearchStepContext]):
    llm: LLMClientBase = Depends()
    provider: SearchProviderBase = Depends()

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        if ctx.run.current is None:
            return ctx
        budget = ctx.run.limits
        round_state = ctx.run.current
        if round_state.pending_search_jobs or round_state.search_fetched_candidates:
            round_state.waiting_for_budget = True
            round_state.waiting_reason = (
                round_state.waiting_reason or "budget_resume_required"
            )
            ctx.run.stop = True
            ctx.run.stop_reason = round_state.waiting_reason
            return ctx
        overview_review = ctx.run.current.overview_review
        content_review = ctx.run.current.content_review
        unresolved_conflict_topics = list(round_state.unresolved_conflict_topics)
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
        round_state.remaining_objectives = self._merge_preserving_order(
            [
                *gap_objectives,
                *entity_objectives,
                *conflict_objectives,
            ]
        )
        llm_signal = await self._query_decide_signal(ctx=ctx)
        llm_prefers_continue = llm_signal.continue_research
        next_queries = list(llm_signal.next_queries[: budget.max_queries_per_round])
        query_objectives = [f"query:{item.query}" for item in next_queries]
        remaining_objectives = self._merge_preserving_order(
            [
                *round_state.remaining_objectives,
                *query_objectives,
            ]
        )
        stop_ready = (not llm_prefers_continue) and not remaining_objectives
        search_exhausted = ctx.run.search_calls >= budget.max_search_calls
        fetch_exhausted = ctx.run.fetch_calls >= budget.max_fetch_calls
        can_search_now = (not search_exhausted) and (not fetch_exhausted)
        can_explore_without_search = self._can_continue_with_explore_only(ctx=ctx)
        min_rounds_per_track = max(1, ctx.run.limits.min_rounds_per_track)
        must_continue_for_min_rounds = (len(ctx.run.history) + 1) < min_rounds_per_track
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
        elif not llm_prefers_continue:
            stop = True
            stop_reason = "model_stop"
        elif not can_execute_next_round:
            stop = True
            stop_reason = "no_executable_path"
        round_state.remaining_objectives = remaining_objectives
        round_state.unresolved_conflict_topics = unresolved_conflict_topics
        round_state.stop = stop
        round_state.stop_reason = stop_reason
        ctx.run.next_queries = [] if stop else next_queries
        round_state.waiting_for_budget = False
        round_state.waiting_reason = ""
        ctx.run.history.append(round_state)
        ctx.run.current = None
        if llm_signal.reason:
            ctx.run.notes.append(f"Decide signal: {llm_signal.reason}")
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
                "stop": stop,
                "stop_reason": str(stop_reason or "n/a"),
            },
        )
        if stop:
            ctx.run.stop = True
            ctx.run.stop_reason = stop_reason
        return ctx

    def _can_continue_with_explore_only(self, *, ctx: ResearchStepContext) -> bool:
        if ctx.run.current is None:
            return False
        if ctx.run.limits.max_fetch_calls <= ctx.run.fetch_calls:
            return False
        return ctx.run.link_candidates_round == ctx.run.current.round_index and bool(
            ctx.run.link_candidates
        )

    async def _query_decide_signal(
        self,
        *,
        ctx: ResearchStepContext,
    ) -> ResearchDecideSignalPayload:
        model = resolve_research_model(
            settings=self.settings,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        routes = resolve_engine_selection_routes(
            settings=self.settings,
            subsystem="research",
            provider=self.provider,
        )
        engine_selection_context = build_engine_selection_context(routes=routes)
        try:
            result = await self.llm.create(
                model=model,
                messages=build_decide_prompt_messages(
                    ctx=ctx,
                    engine_selection_context=engine_selection_context,
                ),
                response_format=ResearchDecideSignalPayload,
                format_override=build_decide_schema(
                    max_queries=ctx.run.limits.max_queries_per_round,
                    select_engines=bool(routes),
                ),
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
