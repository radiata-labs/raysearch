from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from pydantic import Field

from serpsage.core.model_base import MutableModel
from serpsage.models.pipeline import ResearchStepContext
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import build_decide_prompt_messages
from serpsage.steps.research.utils import (
    merge_strings,
    normalize_strings,
    resolve_research_model,
)

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime


class _DecideSignalPayload(MutableModel):
    continue_research: bool = False
    high_yield_remaining: bool = False
    next_queries: list[str] = Field(default_factory=list, max_length=8)
    reason: str = ""


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
        overview_stop = ctx.work.overview_review.stop
        content_stop = ctx.work.content_review.stop
        strategy = str(round_state.query_strategy).strip().casefold()
        model_stop = overview_stop or content_stop or strategy == "stop-ready"
        confidence_ok = round_state.confidence >= budget.stop_confidence
        coverage_ok = round_state.coverage_ratio >= budget.min_coverage_ratio
        gaps_ok = round_state.critical_gaps == 0
        required_entities = normalize_strings(
            ctx.plan.theme_plan.required_entities,
            limit=24,
        )
        unresolved_conflict_topics = normalize_strings(
            round_state.unresolved_conflict_topics,
            limit=16,
        )
        entity_coverage_ok = (
            round_state.entity_coverage_complete if required_entities else True
        )
        multi_signal_stop = (
            model_stop
            and confidence_ok
            and coverage_ok
            and gaps_ok
            and entity_coverage_ok
            and not unresolved_conflict_topics
        )
        corpus_score_gain = round_state.corpus_score_gain
        prev = ctx.rounds[-1] if ctx.rounds else None
        current_low_gain = round_state.result_count <= 0 or float(
            corpus_score_gain
        ) < float(self._LOW_GAIN_THRESHOLD)
        if prev is not None:
            progress = (
                round_state.new_source_ids
                or round_state.coverage_ratio > prev.coverage_ratio
                or round_state.unresolved_conflicts < prev.unresolved_conflicts
                or corpus_score_gain > 0.0
            )
            low_gain_streak = (prev.low_gain_streak + 1) if current_low_gain else 0
        else:
            progress = (len(round_state.new_source_ids) > 0) or corpus_score_gain > 0.0
            low_gain_streak = 1 if current_low_gain else 0
        llm_signal = await self._query_decide_signal(ctx=ctx)
        llm_prefers_continue = llm_signal is not None and (
            llm_signal.continue_research or llm_signal.high_yield_remaining
        )
        raw_next_queries = merge_strings(
            normalize_strings(
                llm_signal.next_queries if llm_signal is not None else [],
                limit=budget.max_queries_per_round,
            ),
            list(ctx.work.next_queries),
            normalize_strings(
                ctx.work.overview_review.next_queries,
                limit=budget.max_queries_per_round,
            ),
            normalize_strings(
                ctx.work.content_review.next_queries,
                limit=budget.max_queries_per_round,
            ),
            limit=budget.max_queries_per_round,
        )
        next_queries = list(raw_next_queries)
        core_question = ctx.plan.theme_plan.core_question
        if required_entities and not entity_coverage_ok:
            next_queries = merge_strings(
                self._build_entity_backfill_queries(
                    core_question=core_question,
                    missing_entities=round_state.missing_entities,
                    limit=budget.max_queries_per_round,
                ),
                next_queries,
                limit=budget.max_queries_per_round,
            )
        search_exhausted = ctx.runtime.search_calls >= budget.max_search_calls
        fetch_exhausted = ctx.runtime.fetch_calls >= budget.max_fetch_calls
        can_search_now = (not search_exhausted) and (not fetch_exhausted)
        can_explore_without_search = self._can_continue_with_explore_only(ctx=ctx)
        min_rounds_per_track = max(1, ctx.runtime.mode_depth.min_rounds_per_track)
        round_count_after_commit = len(ctx.rounds) + 1
        must_continue_for_min_rounds = round_count_after_commit < min_rounds_per_track
        if (
            not multi_signal_stop
            and must_continue_for_min_rounds
            and can_search_now
            and not next_queries
            and core_question
            and not current_low_gain
        ):
            next_queries = [core_question]
        if (
            not multi_signal_stop
            and progress
            and can_search_now
            and not next_queries
            and core_question
            and not current_low_gain
        ):
            next_queries = [core_question]
        gap_objectives = merge_strings(
            normalize_strings(ctx.work.overview_review.critical_gaps, limit=12),
            normalize_strings(ctx.work.content_review.remaining_gaps, limit=12),
            limit=24,
        )
        entity_objectives = self._build_prefixed_objectives(
            prefix="missing_entity",
            values=round_state.missing_entities,
            limit=16,
        )
        conflict_objectives = self._build_prefixed_objectives(
            prefix="conflict",
            values=unresolved_conflict_topics,
            limit=16,
        )
        stop_ready = multi_signal_stop and not (
            gap_objectives or entity_objectives or conflict_objectives
        )
        if stop_ready:
            next_queries = []
        elif (
            not next_queries
            and llm_prefers_continue
            and can_search_now
            and core_question
        ):
            next_queries = [core_question]
        query_objectives = (
            self._build_prefixed_objectives(
                prefix="query",
                values=next_queries,
                limit=budget.max_queries_per_round,
            )
            if (
                not stop_ready
                and (llm_prefers_continue or must_continue_for_min_rounds)
            )
            else []
        )
        remaining_objectives = merge_strings(
            gap_objectives,
            entity_objectives,
            conflict_objectives,
            query_objectives,
            limit=32,
        )
        can_execute_next_round = (
            can_search_now and (len(next_queries) > 0)
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
        elif stop_ready and not remaining_objectives:
            stop = True
            stop_reason = "stop_ready"
        elif (
            low_gain_streak >= 2
            and not gap_objectives
            and not entity_objectives
            and not conflict_objectives
        ):
            stop = True
            stop_reason = "low_gain_stalled"
        elif not can_execute_next_round:
            stop = True
            stop_reason = "no_executable_path"
        round_state.stop_ready = stop_ready
        round_state.remaining_objectives = list(remaining_objectives)
        round_state.low_gain_streak = low_gain_streak
        round_state.unresolved_conflict_topics = list(unresolved_conflict_topics)
        round_state.stop = stop
        round_state.stop_reason = stop_reason
        ctx.plan.next_queries = [] if stop else list(next_queries)
        ctx.rounds.append(round_state)
        if llm_signal is not None:
            reason = llm_signal.reason.strip()
            if reason:
                ctx.notes.append(f"Decide signal: {reason}")
        await self.emit_tracking_event(
            event_name="research.decide.summary",
            request_id=ctx.request_id,
            stage="decide",
            attrs={
                "llm_signal_used": llm_signal is not None,
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
        expected_round = ctx.current_round.round_index
        if ctx.plan.last_round_link_candidates_round != expected_round:
            return False
        return len(ctx.plan.last_round_link_candidates) > 0

    async def _query_decide_signal(
        self, *, ctx: ResearchStepContext
    ) -> _DecideSignalPayload | None:
        if ctx.runtime.mode_depth.mode_key == "research-fast":
            return None
        model = resolve_research_model(
            ctx=ctx,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        try:
            result = await self._llm.create(
                model=model,
                messages=build_decide_prompt_messages(ctx=ctx),
                response_format=_DecideSignalPayload,
                retries=self.settings.research.llm_self_heal_retries,
            )
            return result.data
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
            return None

    def _build_entity_backfill_queries(
        self,
        *,
        core_question: str,
        missing_entities: list[str],
        limit: int,
    ) -> list[str]:
        base = core_question
        queries: list[str] = []
        for item in normalize_strings(missing_entities, limit=16):
            entity = item
            if not entity:
                continue
            if base:
                queries.append(f"{base} {entity}")
            else:
                queries.append(entity)
        if base:
            queries.append(base)
        return merge_strings(queries, [], limit=max(1, limit))

    def _build_prefixed_objectives(
        self,
        *,
        prefix: str,
        values: list[str],
        limit: int,
    ) -> list[str]:
        return [
            f"{prefix}:{item}"
            for item in normalize_strings(values, limit=max(1, limit))
        ]


__all__ = ["ResearchDecideStep"]
