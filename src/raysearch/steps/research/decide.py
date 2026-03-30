from __future__ import annotations

from typing_extensions import override

from raysearch.components.llm.base import LLMClientBase
from raysearch.components.provider.base import SearchProviderBase
from raysearch.components.provider.blend import (
    build_engine_selection_context,
    resolve_engine_selection_routes,
)
from raysearch.dependencies import Depends
from raysearch.models.steps.research import RoundStepContext
from raysearch.models.steps.research.payloads import (
    ContentReviewPayload,
    ResearchDecideSignalPayload,
)
from raysearch.steps.base import StepBase
from raysearch.steps.research.prompt import build_decide_prompt_messages
from raysearch.steps.research.schema import build_decide_schema
from raysearch.steps.research.utils import resolve_research_model


class ResearchDecideStep(StepBase[RoundStepContext]):
    _LOW_INFORMATION_GAIN_SCORE = 0.35
    _LOW_CONFIDENCE_SCORE = 0.45
    _LOW_COVERAGE_SCORE = 0.50
    _HIGH_UNCERTAINTY_SCORE = 0.60
    _CONTINUE_SUPPORT_SCORE = 0.45
    _MINIMUM_ROUNDS_FOR_CONFIDENT_STOP = 2

    llm: LLMClientBase = Depends()
    provider: SearchProviderBase = Depends()

    @override
    async def run_inner(self, ctx: RoundStepContext) -> RoundStepContext:
        if ctx.run.current is None:
            return ctx

        budget = ctx.run.limits
        round_state = ctx.run.current
        if round_state.needs_resume:
            round_state.waiting_for_budget = True
            round_state.waiting_reason = (
                round_state.waiting_reason or "budget_resume_required"
            )
            return ctx

        unresolved_conflict_topics = self._get_unresolved_conflict_topics(
            content_review=round_state.content_review
        )
        missing_entities = list(round_state.missing_entities)
        critical_gaps = (
            list(round_state.content_review.remaining_gaps)
            if round_state.content_review is not None
            else []
        )
        remaining_objectives = self._merge_preserving_order(
            [
                *[f"gap:{item}" for item in critical_gaps],
                *[f"missing_entity:{item}" for item in missing_entities],
                *[f"conflict:{item}" for item in unresolved_conflict_topics],
            ]
        )

        llm_signal = await self._query_decide_signal(ctx=ctx)
        information_gain_score = float(llm_signal.information_gain_score)
        next_queries = list(llm_signal.next_queries[: budget.max_queries_per_round])
        remaining_objectives = self._merge_preserving_order(
            [*remaining_objectives, *[f"query:{item.query}" for item in next_queries]]
        )

        search_exhausted = ctx.run.allocation.search_remaining <= 0
        fetch_exhausted = ctx.run.allocation.fetch_remaining <= 0
        can_search_now = (not search_exhausted) and (not fetch_exhausted)
        can_explore_without_search = self._can_continue_with_explore_only(ctx=ctx)
        min_rounds_per_track = max(1, ctx.run.limits.min_rounds_per_track)
        completed_rounds = len(ctx.run.history) + 1
        must_continue_for_min_rounds = completed_rounds < min_rounds_per_track
        can_execute_next_round = (
            can_search_now and bool(next_queries)
        ) or can_explore_without_search

        should_stop_for_low_gain = (
            information_gain_score < self._LOW_INFORMATION_GAIN_SCORE
            and not remaining_objectives
        )
        should_continue_for_low_confidence = (
            round_state.confidence < self._LOW_CONFIDENCE_SCORE
            and information_gain_score >= self._CONTINUE_SUPPORT_SCORE
        )
        should_continue_for_low_coverage = (
            round_state.coverage_ratio < self._LOW_COVERAGE_SCORE
            and information_gain_score >= self._CONTINUE_SUPPORT_SCORE
        )
        should_continue_for_high_uncertainty = (
            round_state.uncertainty_score > self._HIGH_UNCERTAINTY_SCORE
            and information_gain_score >= self._CONTINUE_SUPPORT_SCORE
        )
        should_continue_for_conflicts = bool(unresolved_conflict_topics) and (
            information_gain_score >= self._CONTINUE_SUPPORT_SCORE
        )
        insufficient_rounds = completed_rounds < self._MINIMUM_ROUNDS_FOR_CONFIDENT_STOP

        stop_ready = (
            (not llm_signal.continue_research)
            and not remaining_objectives
            and not should_continue_for_low_confidence
            and not should_continue_for_low_coverage
            and not should_continue_for_high_uncertainty
            and not should_continue_for_conflicts
        ) or should_stop_for_low_gain

        if stop_ready and insufficient_rounds and can_execute_next_round:
            stop_ready = False
            ctx.run.notes.append("Minimum rounds requirement preventing early stop.")

        stop = False
        stop_reason = ""
        if fetch_exhausted:
            stop = True
            stop_reason = "max_fetch_calls"
        elif search_exhausted and not can_explore_without_search:
            stop = True
            stop_reason = "max_search_calls"
        elif (
            (
                must_continue_for_min_rounds
                and (can_search_now or can_explore_without_search)
            )
            or (remaining_objectives and can_execute_next_round)
            or (should_continue_for_low_confidence and can_execute_next_round)
            or (should_continue_for_low_coverage and can_execute_next_round)
            or (should_continue_for_high_uncertainty and can_execute_next_round)
            or (should_continue_for_conflicts and can_execute_next_round)
        ):
            stop = False
        elif stop_ready:
            stop = True
            stop_reason = "model_stop"
        elif not can_execute_next_round:
            stop = True
            stop_reason = "no_executable_path"

        round_state.stop = stop
        round_state.stop_reason = stop_reason
        ctx.run.next_queries = [] if stop else next_queries
        round_state.waiting_for_budget = False
        round_state.waiting_reason = ""
        ctx.run.archive_current_round()

        if llm_signal.reason:
            ctx.run.notes.append(f"Decide signal: {llm_signal.reason}")
        ctx.run.notes.append(f"Information gain score: {information_gain_score:.2f}")
        await self.tracker.info(
            name="research.decide.summary",
            request_id=ctx.request_id,
            step="research.decide",
            data={
                "success": True,
                "next_queries": len(next_queries),
                "remaining_objectives_count": len(remaining_objectives),
                "stop": stop,
                "stop_reason": stop_reason or "n/a",
                "information_gain_score": information_gain_score,
                "confidence": round_state.confidence,
                "coverage_ratio": round_state.coverage_ratio,
                "uncertainty_score": round_state.uncertainty_score,
                "unresolved_conflicts": len(unresolved_conflict_topics),
            },
        )
        await self.tracker.debug(
            name="research.decide.summary.detail",
            request_id=ctx.request_id,
            step="research.decide",
            data={
                "success": True,
                "llm_prefers_continue": llm_signal.continue_research,
                "min_rounds_per_track": min_rounds_per_track,
                "must_continue_for_min_rounds": must_continue_for_min_rounds,
                "stop_ready": stop_ready,
                "can_search_now": can_search_now,
                "can_explore_without_search": can_explore_without_search,
                "can_execute_next_round": can_execute_next_round,
                "remaining_objectives": remaining_objectives,
                "next_queries": [item.model_dump(mode="json") for item in next_queries],
                "llm_reason": llm_signal.reason,
                "information_gain_score": information_gain_score,
            },
        )
        if stop:
            ctx.run.stop = True
            ctx.run.stop_reason = stop_reason
        return ctx

    def _can_continue_with_explore_only(self, *, ctx: RoundStepContext) -> bool:
        if ctx.run.current is None:
            return False
        if ctx.run.allocation.fetch_remaining <= 0:
            return False
        return ctx.run.link_candidates_round == ctx.run.current.round_index and bool(
            ctx.run.link_candidates
        )

    async def _query_decide_signal(
        self,
        *,
        ctx: RoundStepContext,
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
            await self.meter.record(
                name="llm.tokens",
                request_id=ctx.request_id,
                model=str(model),
                unit="token",
                tokens={
                    "prompt_tokens": int(result.usage.prompt_tokens),
                    "completion_tokens": int(result.usage.completion_tokens),
                    "total_tokens": int(result.usage.total_tokens),
                },
            )
        except Exception as exc:  # noqa: BLE001
            await self.tracker.error(
                name="research.decide.failed",
                request_id=ctx.request_id,
                step="research.decide",
                error_code="research_decide_signal_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={"model": str(model)},
            )
            raise
        return result.data

    def _get_unresolved_conflict_topics(
        self,
        *,
        content_review: ContentReviewPayload | None,
    ) -> list[str]:
        if content_review is None:
            return []
        return [
            item.topic
            for item in content_review.conflict_resolutions
            if item.status in {"unresolved", "insufficient"}
        ]

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
