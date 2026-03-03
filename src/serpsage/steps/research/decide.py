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
from serpsage.utils import clean_whitespace

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
        overview_stop = bool(ctx.work.overview_review.stop)
        content_stop = bool(ctx.work.content_review.stop)
        strategy = str(round_state.query_strategy or "").strip().casefold()
        model_stop = bool(overview_stop or content_stop or strategy == "stop-ready")
        confidence_ok = float(round_state.confidence) >= float(budget.stop_confidence)
        coverage_ok = float(round_state.coverage_ratio) >= float(
            budget.min_coverage_ratio
        )
        conflict_ok = int(round_state.unresolved_conflicts) <= int(
            budget.max_unresolved_conflicts
        )
        gaps_ok = int(round_state.critical_gaps) == 0
        required_entities = normalize_strings(
            ctx.plan.theme_plan.required_entities,
            limit=24,
        )
        entity_coverage_ok = (
            bool(round_state.entity_coverage_complete) if required_entities else True
        )
        multi_signal_stop = bool(
            model_stop
            and confidence_ok
            and coverage_ok
            and conflict_ok
            and gaps_ok
            and entity_coverage_ok
        )
        corpus_score_gain = float(round_state.corpus_score_gain)
        if ctx.rounds:
            prev = ctx.rounds[-1]
            progress = bool(
                round_state.new_source_ids
                or float(round_state.coverage_ratio) > float(prev.coverage_ratio)
                or int(round_state.unresolved_conflicts)
                < int(prev.unresolved_conflicts)
                or corpus_score_gain > 0.0
            )
        else:
            progress = bool(round_state.new_source_ids or corpus_score_gain > 0.0)
        if progress:
            ctx.runtime.no_progress_rounds = 0
        else:
            ctx.runtime.no_progress_rounds += 1
        llm_signal = await self._query_decide_signal(ctx=ctx)
        llm_prefers_continue = bool(
            llm_signal is not None
            and (llm_signal.continue_research or llm_signal.high_yield_remaining)
        )
        raw_next_queries = merge_strings(
            normalize_strings(
                llm_signal.next_queries if llm_signal is not None else [],
                limit=int(budget.max_queries_per_round),
            ),
            list(ctx.work.next_queries),
            normalize_strings(
                ctx.work.overview_review.next_queries,
                limit=int(budget.max_queries_per_round),
            ),
            normalize_strings(
                ctx.work.content_review.next_queries,
                limit=int(budget.max_queries_per_round),
            ),
            limit=int(budget.max_queries_per_round),
        )
        next_queries = list(raw_next_queries)
        core_question = clean_whitespace(ctx.plan.core_question or ctx.request.themes)
        if required_entities and not entity_coverage_ok:
            next_queries = merge_strings(
                self._build_entity_backfill_queries(
                    core_question=core_question,
                    missing_entities=round_state.missing_entities,
                    limit=int(budget.max_queries_per_round),
                ),
                next_queries,
                limit=int(budget.max_queries_per_round),
            )
        no_progress_threshold = max(
            1,
            int(ctx.runtime.no_progress_rounds_to_stop_effective),
        )
        min_rounds_per_track = max(1, int(ctx.runtime.mode_depth.min_rounds_per_track))
        round_count_after_commit = int(len(ctx.rounds)) + 1
        must_continue_for_min_rounds = round_count_after_commit < int(
            min_rounds_per_track
        )
        can_explore_without_search = self._can_continue_with_explore_only(ctx=ctx)
        if llm_prefers_continue and not next_queries and core_question:
            next_queries = [core_question]
        allow_auto_seed = (
            not multi_signal_stop
            and int(ctx.runtime.no_progress_rounds) < int(no_progress_threshold)
            and int(ctx.runtime.search_calls) < int(budget.max_search_calls)
            and int(ctx.runtime.fetch_calls) < int(budget.max_fetch_calls)
        )
        if allow_auto_seed and not next_queries and core_question:
            next_queries = [core_question]
        if (
            not multi_signal_stop
            and not next_queries
            and progress
            and int(ctx.runtime.search_calls) < int(budget.max_search_calls)
        ):
            next_queries = [ctx.request.themes]
        stop = False
        stop_reason = ""
        search_exhausted = int(ctx.runtime.search_calls) >= int(budget.max_search_calls)
        fetch_exhausted = int(ctx.runtime.fetch_calls) >= int(budget.max_fetch_calls)
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
        elif multi_signal_stop and not llm_prefers_continue:
            stop = True
            stop_reason = "multi_signal_stop"
        elif int(ctx.runtime.no_progress_rounds) >= int(no_progress_threshold) and (
            not llm_prefers_continue
        ):
            stop = True
            stop_reason = "no_progress"
        elif (
            not next_queries
            and not llm_prefers_continue
            and not can_explore_without_search
        ):
            stop = True
            stop_reason = "no_next_queries"
        round_state.stop = bool(stop)
        round_state.stop_reason = stop_reason
        ctx.plan.next_queries = list(next_queries)
        ctx.rounds.append(round_state)
        if llm_signal is not None:
            reason = clean_whitespace(llm_signal.reason)
            if reason:
                ctx.notes.append(f"Decide signal: {reason}")
        await self.emit_tracking_event(
            event_name="research.decide.summary",
            request_id=ctx.request_id,
            stage="decide",
            attrs={
                "llm_signal_used": bool(llm_signal is not None),
                "llm_prefers_continue": bool(llm_prefers_continue),
                "min_rounds_per_track": int(min_rounds_per_track),
                "must_continue_for_min_rounds": bool(must_continue_for_min_rounds),
                "no_progress_threshold": int(no_progress_threshold),
                "next_queries": int(len(next_queries)),
                "stop": bool(stop),
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
        if int(ctx.runtime.budget.max_fetch_calls) <= int(ctx.runtime.fetch_calls):
            return False
        expected_round = int(ctx.current_round.round_index)
        if int(ctx.plan.last_round_link_candidates_round) != expected_round:
            return False
        return bool(ctx.plan.last_round_link_candidates)

    async def _query_decide_signal(
        self, *, ctx: ResearchStepContext
    ) -> _DecideSignalPayload | None:
        if not bool(ctx.runtime.mode_depth.enable_llm_track_orchestrator):
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
                retries=int(self.settings.research.llm_self_heal_retries),
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
            core_question=ctx.plan.core_question or ctx.request.themes,
            mode_depth_profile=str(ctx.runtime.mode_depth.mode_key),
            confidence=float(round_state.confidence),
            coverage_ratio=float(round_state.coverage_ratio),
            unresolved_conflicts=int(round_state.unresolved_conflicts),
            critical_gaps=int(round_state.critical_gaps),
            missing_entities=list(round_state.missing_entities),
            search_remaining=max(
                0,
                int(ctx.runtime.budget.max_search_calls)
                - int(ctx.runtime.search_calls),
            ),
            fetch_remaining=max(
                0,
                int(ctx.runtime.budget.max_fetch_calls) - int(ctx.runtime.fetch_calls),
            ),
        )

    def _build_entity_backfill_queries(
        self,
        *,
        core_question: str,
        missing_entities: list[str],
        limit: int,
    ) -> list[str]:
        base = clean_whitespace(core_question)
        queries: list[str] = []
        for item in normalize_strings(missing_entities, limit=16):
            entity = clean_whitespace(item)
            if not entity:
                continue
            if base:
                queries.append(f"{base} {entity}")
            else:
                queries.append(entity)
        if base:
            queries.append(base)
        return merge_strings(queries, [], limit=max(1, int(limit)))


__all__ = ["ResearchDecideStep"]
