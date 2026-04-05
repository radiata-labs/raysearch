from __future__ import annotations

from datetime import UTC, datetime
from typing_extensions import override

from raysearch.components.llm.base import LLMClientBase
from raysearch.components.provider.base import SearchProviderBase
from raysearch.components.provider.blend import (
    build_engine_selection_context,
    resolve_engine_selection_routes,
)
from raysearch.dependencies import Depends
from raysearch.models.steps.research import (
    ResearchRound,
    RoundStepContext,
)
from raysearch.models.steps.research.payloads import (
    PlanOutputPayload,
    PlanSearchJobPayload,
    RoundAction,
)
from raysearch.models.steps.search import QuerySourceSpec, SearchFetchedCandidate
from raysearch.steps.base import StepBase
from raysearch.steps.research.prompt import build_plan_prompt_messages
from raysearch.steps.research.schema import build_plan_schema
from raysearch.steps.research.utils import resolve_research_model


class ResearchPlanStep(StepBase[RoundStepContext]):
    llm: LLMClientBase = Depends()
    provider: SearchProviderBase = Depends()

    @override
    async def should_run(self, ctx: RoundStepContext) -> bool:
        """Execute unless stopped or waiting for budget resume."""
        if ctx.run.stop:
            return False
        return not (ctx.run.current is not None and (ctx.run.current.pending_search_jobs or ctx.run.current.search_fetched_candidates or ctx.run.current.waiting_for_budget))

    @override
    async def run_inner(self, ctx: RoundStepContext) -> RoundStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        # Pre-condition: should_run() already verified no pending I/O or budget wait
        budget = ctx.run.limits
        if ctx.run.round_index >= budget.max_rounds:
            ctx.run.stop = True
            ctx.run.stop_reason = "max_rounds"
            return ctx
        if ctx.run.allocation.fetch_remaining <= 0:
            ctx.run.stop = True
            ctx.run.stop_reason = "max_fetch_calls"
            return ctx
        round_index = ctx.run.round_index + 1
        if ctx.run.allocation.search_remaining <= 0 and not self._can_attempt_explore(
            ctx=ctx,
            round_index=round_index,
        ):
            ctx.run.stop = True
            ctx.run.stop_reason = "max_search_calls"
            return ctx
        ctx.run.round_index = round_index
        ctx.run.current = ResearchRound(round_index=round_index)
        core_question = ctx.task.question or ctx.request.themes
        candidate_queries = self._build_candidate_queries(
            next_queries=ctx.run.next_queries,
            core_question=core_question,
        )
        routes = resolve_engine_selection_routes(
            settings=self.settings,
            subsystem="research",
            provider=self.provider,
        )
        engine_selection_context = build_engine_selection_context(routes=routes)
        model = resolve_research_model(
            settings=self.settings,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        last_round_candidates = self._resolve_last_round_candidates(
            ctx=ctx,
            round_index=round_index,
        )
        try:
            chat_result = await self.llm.create(
                model=model,
                messages=build_plan_prompt_messages(
                    ctx=ctx,
                    candidate_queries=candidate_queries,
                    core_question=core_question,
                    now_utc=now_utc,
                    last_round_candidates=last_round_candidates,
                    engine_selection_context=engine_selection_context,
                ),
                response_format=PlanOutputPayload,
                format_override=build_plan_schema(select_engines=bool(routes)),
                retries=self.settings.research.llm_self_heal_retries,
            )
            await self.meter.record(
                name="llm.tokens",
                request_id=ctx.request_id,
                model=str(model),
                unit="token",
                tokens={
                    "prompt_tokens": int(chat_result.usage.prompt_tokens),
                    "completion_tokens": int(chat_result.usage.completion_tokens),
                    "total_tokens": int(chat_result.usage.total_tokens),
                },
            )
        except Exception as exc:  # noqa: BLE001
            await self.tracker.error(
                name="research.plan.failed",
                request_id=ctx.request_id,
                step="research.plan",
                error_code="research_round_plan_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "round_index": round_index,
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
            limit=ctx.run.limits.explore_target_pages_per_round,
        )
        if round_action == "explore" and not explore_target_source_ids:
            ctx.run.stop = True
            ctx.run.stop_reason = "no_explore_targets"
            ctx.run.current.stop = True
            ctx.run.current.stop_reason = "no_explore_targets"
            ctx.run.history.append(ctx.run.current)
            return ctx
        search_limit = max(
            0,
            min(
                budget.max_queries_per_round,
                ctx.run.allocation.search_remaining,
            ),
        )
        search_jobs = self._build_search_jobs(
            payload=payload,
            job_limit=search_limit,
        )
        if round_action == "search" and not search_jobs:
            ctx.run.stop = True
            ctx.run.stop_reason = "no_queries"
            ctx.run.current.stop = True
            ctx.run.current.stop_reason = "no_queries"
            ctx.run.history.append(ctx.run.current)
            return ctx
        ctx.run.current.round_action = round_action
        ctx.run.current.explore_target_source_ids = explore_target_source_ids
        ctx.run.current.search_jobs = search_jobs
        ctx.run.current.pending_search_jobs = [
            item.model_copy(deep=True) for item in search_jobs
        ]
        ctx.run.current.query_strategy = payload.query_strategy
        ctx.run.current.queries = [job.query for job in search_jobs]
        return ctx

    def _build_candidate_queries(
        self,
        *,
        next_queries: list[QuerySourceSpec],
        core_question: str,
    ) -> list[QuerySourceSpec]:
        ordered = [*next_queries, QuerySourceSpec(query=core_question)]
        seen: set[tuple[str, tuple[str, ...]]] = set()
        result: list[QuerySourceSpec] = []
        for item in ordered:
            if not item.query:
                continue
            key = (item.query.casefold(), tuple(item.include_sources))
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
    ) -> list[PlanSearchJobPayload]:
        if job_limit <= 0:
            return []
        return [item.model_copy(deep=True) for item in payload.search_jobs[:job_limit]]

    def _can_attempt_explore(self, *, ctx: RoundStepContext, round_index: int) -> bool:
        if round_index <= 1:
            return False
        if ctx.run.allocation.fetch_remaining <= 0:
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
        ctx: RoundStepContext,
        round_index: int,
    ) -> dict[int, SearchFetchedCandidate]:
        expected_round = max(0, round_index - 1)
        if ctx.run.link_candidates_round != expected_round:
            return {}
        return {
            source_id: item.model_copy(deep=True)
            for source_id, item in ctx.run.link_candidates.items()
        }

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
        candidates: dict[int, SearchFetchedCandidate],
        limit: int,
    ) -> list[int]:
        if not raw_source_ids or not candidates:
            return []
        allowed_source_ids = set(candidates)
        result: list[int] = []
        for source_id in raw_source_ids:
            if source_id not in allowed_source_ids:
                continue
            result.append(source_id)
            if len(result) >= max(1, limit):
                break
        return result


__all__ = ["ResearchPlanStep"]
