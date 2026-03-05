from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from pydantic import Field

from serpsage.core.model_base import MutableModel
from serpsage.models.pipeline import ResearchStepContext
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import build_decide_signal_messages
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
        conflict_ok = (
            round_state.unresolved_conflicts <= budget.max_unresolved_conflicts
        )
        gaps_ok = round_state.critical_gaps == 0
        required_entities = normalize_strings(
            ctx.plan.theme_plan.required_entities,
            limit=24,
        )
        entity_coverage_ok = (
            round_state.entity_coverage_complete if required_entities else True
        )
        multi_signal_stop = (
            model_stop
            and confidence_ok
            and coverage_ok
            and conflict_ok
            and gaps_ok
            and entity_coverage_ok
        )
        corpus_score_gain = round_state.corpus_score_gain
        if ctx.rounds:
            prev = ctx.rounds[-1]
            progress = (
                round_state.new_source_ids
                or round_state.coverage_ratio > prev.coverage_ratio
                or round_state.unresolved_conflicts < prev.unresolved_conflicts
                or corpus_score_gain > 0.0
            )
        else:
            progress = (len(round_state.new_source_ids) > 0) or corpus_score_gain > 0.0
        if progress:
            ctx.runtime.no_progress_rounds = 0
        else:
            ctx.runtime.no_progress_rounds += 1
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
        core_question = self._resolve_core_question(ctx)
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
        no_progress_threshold = max(
            1,
            ctx.runtime.mode_depth.no_progress_rounds_to_stop_effective,
        )
        min_rounds_per_track = max(1, ctx.runtime.mode_depth.min_rounds_per_track)
        round_count_after_commit = len(ctx.rounds) + 1
        must_continue_for_min_rounds = round_count_after_commit < min_rounds_per_track
        if llm_prefers_continue and not next_queries and core_question:
            next_queries = [core_question]
        allow_auto_seed = (
            not multi_signal_stop
            and ctx.runtime.no_progress_rounds < no_progress_threshold
            and can_search_now
        )
        if allow_auto_seed and not next_queries and core_question:
            next_queries = [core_question]
        if not multi_signal_stop and not next_queries and progress and can_search_now:
            next_queries = [ctx.request.themes]
        if not next_queries and can_search_now and core_question:
            next_queries = [core_question]
        can_execute_next_round = (
            can_search_now and (len(next_queries) > 0)
        ) or can_explore_without_search
        no_progress_stop_ready = ctx.runtime.no_progress_rounds >= no_progress_threshold
        stop_readiness_high = multi_signal_stop or no_progress_stop_ready
        stop = False
        stop_reason = ""
        if (
            must_continue_for_min_rounds
            and not fetch_exhausted
            and (not search_exhausted or can_explore_without_search)
        ):
            stop = False
        elif fetch_exhausted:
            stop = True
            stop_reason = "max_fetch_calls"
        elif search_exhausted and not can_explore_without_search:
            stop = True
            stop_reason = "max_search_calls"
        elif (
            stop_readiness_high
            and (not llm_prefers_continue)
            and (not can_execute_next_round)
        ):
            stop = True
            stop_reason = "stop_ready_no_executable_path"
        round_state.stop = stop
        round_state.stop_reason = stop_reason
        ctx.plan.next_queries = list(next_queries)
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
                "no_progress_threshold": no_progress_threshold,
                "no_progress_stop_ready": no_progress_stop_ready,
                "stop_readiness_high": stop_readiness_high,
                "can_search_now": can_search_now,
                "can_explore_without_search": can_explore_without_search,
                "can_execute_next_round": can_execute_next_round,
                "next_queries": len(next_queries),
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
                messages=self._build_decide_messages(ctx=ctx),
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

    def _build_decide_messages(
        self, *, ctx: ResearchStepContext
    ) -> list[dict[str, str]]:
        round_state = ctx.current_round
        if round_state is None:
            return []
        return build_decide_signal_messages(
            core_question=self._resolve_core_question(ctx),
            mode_depth_profile=str(ctx.runtime.mode_depth.mode_key),
            confidence=round_state.confidence,
            coverage_ratio=round_state.coverage_ratio,
            unresolved_conflicts=round_state.unresolved_conflicts,
            critical_gaps=round_state.critical_gaps,
            missing_entities=list(round_state.missing_entities),
            search_remaining=max(
                0,
                ctx.runtime.budget.max_search_calls - ctx.runtime.search_calls,
            ),
            fetch_remaining=max(
                0,
                ctx.runtime.budget.max_fetch_calls - ctx.runtime.fetch_calls,
            ),
        )

    def _resolve_core_question(self, ctx: ResearchStepContext) -> str:
        return ctx.plan.theme_plan.core_question or ctx.request.themes

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


__all__ = ["ResearchDecideStep"]
