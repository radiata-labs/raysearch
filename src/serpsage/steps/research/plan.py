from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.steps.research import (
    PlanOutputPayload,
    ResearchLinkCandidate,
    ResearchRoundState,
    ResearchSearchJob,
    ResearchStepContext,
    RoundAction,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import build_plan_prompt_messages
from serpsage.steps.research.schema import build_plan_schema
from serpsage.steps.research.utils import resolve_research_model

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime


class ResearchPlanStep(StepBase[ResearchStepContext]):
    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        if ctx.runtime.stop:
            return ctx
        budget = ctx.runtime.budget
        if ctx.runtime.round_index >= budget.max_rounds:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "max_rounds"
            return ctx
        remaining_fetch = budget.max_fetch_calls - ctx.runtime.fetch_calls
        if remaining_fetch <= 0:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "max_fetch_calls"
            return ctx
        remaining_search = budget.max_search_calls - ctx.runtime.search_calls
        round_index = ctx.runtime.round_index + 1
        if remaining_search <= 0 and not self._can_attempt_explore(
            ctx=ctx,
            round_index=round_index,
        ):
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "max_search_calls"
            return ctx
        ctx.runtime.round_index = round_index
        ctx.current_round = ResearchRoundState(round_index=round_index)
        ctx.work.search_jobs = []
        ctx.work.round_action = "search"
        ctx.work.explore_target_source_ids = []
        ctx.work.search_fetched_candidates = []
        ctx.work.overview_review = None
        ctx.work.content_review = None
        ctx.work.need_content_source_ids = []
        ctx.work.next_queries = []
        core_question = ctx.plan.theme_plan.core_question or ctx.request.themes
        candidate_queries = self._build_candidate_queries(
            next_queries=ctx.plan.next_queries,
            core_question=core_question,
        )
        model = resolve_research_model(
            ctx=ctx,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        last_round_candidates = self._resolve_last_round_candidates(
            ctx=ctx,
            round_index=round_index,
        )
        try:
            chat_result = await self._llm.create(
                model=model,
                messages=build_plan_prompt_messages(
                    ctx=ctx,
                    candidate_queries=candidate_queries,
                    core_question=core_question,
                    now_utc=now_utc,
                    last_round_candidates=last_round_candidates,
                ),
                response_format=PlanOutputPayload,
                format_override=build_plan_schema(),
                retries=self.settings.research.llm_self_heal_retries,
            )
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="research.plan.error",
                request_id=ctx.request_id,
                stage="plan",
                status="error",
                error_code="research_round_plan_failed",
                error_type=type(exc).__name__,
                attrs={
                    "round_index": round_index,
                    "message": str(exc),
                },
            )
            raise
        payload = chat_result.data
        round_action = self._resolve_round_action(
            payload_round_action=payload.round_action,
            round_index=round_index,
            allow_explore=self._can_attempt_explore(
                ctx=ctx,
                round_index=round_index,
            ),
        )
        explore_target_source_ids = self._resolve_explore_target_source_ids(
            raw_source_ids=payload.explore_target_source_ids,
            candidates=last_round_candidates,
            limit=ctx.runtime.mode_depth.explore_target_pages_per_round,
        )
        if round_action == "explore" and not explore_target_source_ids:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "no_explore_targets"
            ctx.current_round.stop = True
            ctx.current_round.stop_reason = "no_explore_targets"
            ctx.rounds.append(ctx.current_round)
            return ctx
        search_limit = max(
            0,
            min(
                budget.max_queries_per_round,
                budget.max_search_calls - ctx.runtime.search_calls,
            ),
        )
        search_jobs = self._build_search_jobs(
            payload=payload,
            job_limit=search_limit,
        )
        if round_action == "search" and not search_jobs:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "no_queries"
            ctx.current_round.stop = True
            ctx.current_round.stop_reason = "no_queries"
            ctx.rounds.append(ctx.current_round)
            return ctx
        ctx.work.round_action = round_action
        ctx.work.explore_target_source_ids = explore_target_source_ids
        ctx.work.search_jobs = search_jobs
        ctx.current_round.query_strategy = payload.query_strategy
        ctx.current_round.queries = [job.query for job in search_jobs]
        return ctx

    def _build_candidate_queries(
        self,
        *,
        next_queries: list[str],
        core_question: str,
    ) -> list[str]:
        ordered = [*next_queries, core_question]
        seen: set[str] = set()
        result: list[str] = []
        for item in ordered:
            if not item:
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            result.append(item)
        return result

    def _build_search_jobs(
        self,
        *,
        payload: PlanOutputPayload,
        job_limit: int,
    ) -> list[ResearchSearchJob]:
        if job_limit <= 0:
            return []
        return [
            ResearchSearchJob.model_validate(item.model_dump())
            for item in payload.search_jobs[:job_limit]
        ]

    def _can_attempt_explore(
        self, *, ctx: ResearchStepContext, round_index: int
    ) -> bool:
        if round_index <= 1:
            return False
        if ctx.runtime.budget.max_fetch_calls <= ctx.runtime.fetch_calls:
            return False
        return bool(
            self._resolve_last_round_candidates(
                ctx=ctx,
                round_index=round_index,
            )
        )

    def _resolve_last_round_candidates(
        self,
        *,
        ctx: ResearchStepContext,
        round_index: int,
    ) -> list[ResearchLinkCandidate]:
        expected_round = max(0, round_index - 1)
        if ctx.plan.last_round_link_candidates_round != expected_round:
            return []
        return [
            item.model_copy(deep=True) for item in ctx.plan.last_round_link_candidates
        ]

    def _resolve_round_action(
        self,
        *,
        payload_round_action: RoundAction,
        round_index: int,
        allow_explore: bool,
    ) -> RoundAction:
        if round_index <= 1:
            return "search"
        if payload_round_action == "explore" and allow_explore:
            return "explore"
        return "search"

    def _resolve_explore_target_source_ids(
        self,
        *,
        raw_source_ids: list[int],
        candidates: list[ResearchLinkCandidate],
        limit: int,
    ) -> list[int]:
        if not raw_source_ids or not candidates:
            return []
        allowed_source_ids = {item.source_id for item in candidates}
        result: list[int] = []
        for source_id in raw_source_ids:
            if source_id not in allowed_source_ids:
                continue
            result.append(source_id)
            if len(result) >= max(1, limit):
                break
        return result


__all__ = ["ResearchPlanStep"]
