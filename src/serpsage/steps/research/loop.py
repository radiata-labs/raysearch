from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing_extensions import override

import anyio
from pydantic import Field

from serpsage.core.model_base import MutableModel
from serpsage.models.pipeline import (
    ResearchBudgetState,
    ResearchQuestionCard,
    ResearchRoundState,
    ResearchRuntimeState,
    ResearchStepContext,
    ResearchTrackResult,
)
from serpsage.models.research import TrackInsightCardPayload
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import (
    build_gap_closure_messages as build_gap_closure_prompt_messages,
)
from serpsage.steps.research.prompt import (
    build_track_orchestrator_messages as build_track_orchestrator_prompt_messages,
)
from serpsage.steps.research.utils import (
    merge_strings,
    normalize_strings,
    resolve_research_model,
)
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.steps.base import RunnerBase


@dataclass(slots=True)
class _TrackAllocation:
    search_grant: int
    fetch_grant: int
    max_queries_per_round: int
    bonus: bool = False
    fetch_only: bool = False


@dataclass(slots=True)
class _BudgetReservationState:
    search_reserved: int = 0
    fetch_reserved: int = 0


@dataclass(slots=True)
class _OrchestratorState:
    last_global_search_used: int = -1
    priorities: dict[str, float] = field(default_factory=dict)
    query_width_hints: dict[str, int] = field(default_factory=dict)
    rationale: str = ""
    refresh_interval_search_calls: int = 2


class _TrackOrchestratorPriorityPayload(MutableModel):
    question_id: str
    priority_score: float = Field(default=0.5, ge=0.0, le=1.0)
    query_width_hint: int = Field(default=1, ge=1, le=2)
    reason: str = ""


class _TrackOrchestratorOutputPayload(MutableModel):
    priorities: list[_TrackOrchestratorPriorityPayload] = Field(
        default_factory=list,
        max_length=24,
    )
    rationale: str = ""


class _GapClosureOutputPayload(MutableModel):
    queries: list[str] = Field(default_factory=list, max_length=8)
    objective: str = ""


class ResearchLoopStep(StepBase[ResearchStepContext]):
    _BASELINE_QUERY_WIDTH = 1
    _BONUS_QUERY_WIDTH = 2
    _BONUS_RATIO = 0.30

    def __init__(
        self,
        *,
        rt: Runtime,
        llm: LLMClientBase,
        round_runner: RunnerBase[ResearchStepContext],
        render_step: StepBase[ResearchStepContext],
    ) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self._round_runner = round_runner
        self._render_step = render_step
        self.bind_deps(llm, round_runner, render_step)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        cards = self._resolve_question_cards(ctx)
        ctx.parallel.question_cards = [item.model_copy(deep=True) for item in cards]
        ctx.parallel.track_results = []
        if not cards:
            ctx.runtime.stop = True
            ctx.runtime.stop_reason = "no_question_cards"
            return ctx
        track_map: dict[str, ResearchStepContext] = {
            card.question_id: self._build_track_context(root=ctx, card=card)
            for card in cards
        }
        result_map: dict[str, ResearchTrackResult] = {}
        reservation_state = _BudgetReservationState()
        orchestrator_state = _OrchestratorState()
        budget_lock = anyio.Lock()
        result_map_lock = anyio.Lock()
        orchestrator_lock = anyio.Lock()
        if self._orchestrator_enabled(ctx):
            await self._refresh_orchestrator_if_needed(
                root=ctx,
                track_map=track_map,
                state=orchestrator_state,
                state_lock=orchestrator_lock,
                force=True,
            )
        async with anyio.create_task_group() as tg:
            for card in cards:
                tg.start_soon(
                    self._run_track_worker,
                    ctx,
                    card,
                    track_map[card.question_id],
                    track_map,
                    result_map,
                    reservation_state,
                    orchestrator_state,
                    budget_lock,
                    result_map_lock,
                    orchestrator_lock,
                )
        ctx.parallel.track_results = [
            result_map[card.question_id]
            for card in cards
            if card.question_id in result_map
        ]
        ctx.runtime.stop = True
        if self._global_budget_exhausted(ctx):
            ctx.runtime.stop_reason = "global_budget_exhausted"
        else:
            ctx.runtime.stop_reason = "all_tracks_completed"
        return ctx

    async def _run_track_worker(
        self,
        root: ResearchStepContext,
        card: ResearchQuestionCard,
        track_ctx: ResearchStepContext,
        track_map: dict[str, ResearchStepContext],
        result_map: dict[str, ResearchTrackResult],
        reservation_state: _BudgetReservationState,
        orchestrator_state: _OrchestratorState,
        budget_lock: anyio.Lock,
        result_map_lock: anyio.Lock,
        orchestrator_lock: anyio.Lock,
    ) -> None:
        question_id = card.question_id
        local_ctx = track_ctx
        track_result: ResearchTrackResult | None = None
        try:
            while not local_ctx.runtime.stop:
                alloc = await self._reserve_track_allocation(
                    root=root,
                    track_ctx=local_ctx,
                    card=card,
                    track_map=track_map,
                    reservation_state=reservation_state,
                    orchestrator_state=orchestrator_state,
                    budget_lock=budget_lock,
                    orchestrator_lock=orchestrator_lock,
                )
                if self._allocation_blocked(alloc):
                    local_ctx.runtime.stop = True
                    local_ctx.runtime.stop_reason = "global_budget_exhausted"
                    break
                before_search = int(local_ctx.runtime.search_calls)
                before_fetch = int(local_ctx.runtime.fetch_calls)
                delta_search = 0
                delta_fetch = 0
                try:
                    self._apply_track_round_budget(
                        track_ctx=local_ctx,
                        alloc=alloc,
                        base_budget=root.runtime.budget,
                    )
                    local_ctx = await self._round_runner.run(local_ctx)
                    delta_search = max(
                        0, int(local_ctx.runtime.search_calls) - int(before_search)
                    )
                    delta_fetch = max(
                        0, int(local_ctx.runtime.fetch_calls) - int(before_fetch)
                    )
                except Exception as exc:  # noqa: BLE001
                    await self.emit_tracking_event(
                        event_name="research.loop.error",
                        request_id=root.request_id,
                        stage="track",
                        status="error",
                        error_code="research_loop_track_failed",
                        error_type=type(exc).__name__,
                        attrs={
                            "question_id": question_id,
                            "message": str(exc),
                        },
                    )
                    local_ctx.runtime.stop = True
                    local_ctx.runtime.stop_reason = "track_execution_failed"
                finally:
                    await self._commit_track_usage(
                        root=root,
                        alloc=alloc,
                        delta_search=delta_search,
                        delta_fetch=delta_fetch,
                        reservation_state=reservation_state,
                        budget_lock=budget_lock,
                    )
            local_ctx = await self._run_gap_closure_passes(
                root=root,
                card=card,
                track_ctx=local_ctx,
                track_map=track_map,
                reservation_state=reservation_state,
                orchestrator_state=orchestrator_state,
                budget_lock=budget_lock,
                orchestrator_lock=orchestrator_lock,
            )
            if not local_ctx.runtime.stop:
                local_ctx.runtime.stop = True
                if self._global_budget_exhausted(root):
                    local_ctx.runtime.stop_reason = "global_budget_exhausted"
                else:
                    local_ctx.runtime.stop_reason = (
                        local_ctx.runtime.stop_reason or "all_tracks_completed"
                    )
            track_result = await self._finalize_track(
                card=card,
                track_ctx=local_ctx,
            )
        except Exception as exc:  # noqa: BLE001
            local_ctx.runtime.stop = True
            local_ctx.runtime.stop_reason = (
                local_ctx.runtime.stop_reason or "track_worker_failed"
            )
            await self.emit_tracking_event(
                event_name="research.loop.error",
                request_id=root.request_id,
                stage="worker",
                status="error",
                error_code="research_loop_worker_failed",
                error_type=type(exc).__name__,
                attrs={
                    "question_id": question_id,
                    "message": str(exc),
                },
            )
            track_result = self._build_failed_track_result(
                card=card,
                track_ctx=local_ctx,
            )
        if track_result is None:
            track_result = self._build_failed_track_result(
                card=card,
                track_ctx=local_ctx,
            )
        async with result_map_lock:
            result_map[question_id] = track_result

    async def _reserve_track_allocation(
        self,
        *,
        root: ResearchStepContext,
        track_ctx: ResearchStepContext,
        card: ResearchQuestionCard,
        track_map: dict[str, ResearchStepContext],
        reservation_state: _BudgetReservationState,
        orchestrator_state: _OrchestratorState,
        budget_lock: anyio.Lock,
        orchestrator_lock: anyio.Lock,
    ) -> _TrackAllocation:
        fetch_per_search_floor = max(
            1,
            int(root.runtime.budget.max_fetch_calls)
            // max(1, int(root.runtime.budget.max_search_calls)),
        )
        bonus_fetch = 1
        baseline_width = max(1, int(self._BASELINE_QUERY_WIDTH))
        bonus_width = max(
            baseline_width,
            int(self._BONUS_QUERY_WIDTH),
        )
        await self._refresh_orchestrator_if_needed(
            root=root,
            track_map=track_map,
            state=orchestrator_state,
            state_lock=orchestrator_lock,
        )
        score = self._score_track(track_ctx, card)
        width_hint = 1
        if self._orchestrator_enabled(root):
            score = float(orchestrator_state.priorities.get(card.question_id, score))
            width_hint = max(
                baseline_width,
                int(orchestrator_state.query_width_hints.get(card.question_id, 1)),
            )
        async with budget_lock:
            remaining_search = max(
                0,
                int(root.parallel.global_search_budget)
                - int(root.parallel.global_search_used)
                - int(reservation_state.search_reserved),
            )
            remaining_fetch = max(
                0,
                int(root.parallel.global_fetch_budget)
                - int(root.parallel.global_fetch_used)
                - int(reservation_state.fetch_reserved),
            )
            if remaining_fetch <= 0:
                return _TrackAllocation(
                    search_grant=0,
                    fetch_grant=0,
                    max_queries_per_round=1,
                    bonus=False,
                )
            if remaining_search <= 0:
                if not self._can_allocate_fetch_only_round(track_ctx):
                    return _TrackAllocation(
                        search_grant=0,
                        fetch_grant=0,
                        max_queries_per_round=1,
                        bonus=False,
                    )
                fetch_grant = max(
                    1,
                    min(
                        int(remaining_fetch),
                        int(root.runtime.budget.max_fetch_per_round),
                    ),
                )
                reservation_state.fetch_reserved += int(fetch_grant)
                return _TrackAllocation(
                    search_grant=0,
                    fetch_grant=int(fetch_grant),
                    max_queries_per_round=1,
                    bonus=False,
                    fetch_only=True,
                )
            bonus_ratio = max(0.0, min(1.0, float(self._BONUS_RATIO)))
            bonus_threshold = max(0.0, min(1.0, 1.0 - bonus_ratio))
            bonus_by_score = bool(score >= bonus_threshold)
            bonus_by_width = bool(width_hint >= bonus_width)
            bonus = bool(
                (bonus_by_score or bonus_by_width)
                and remaining_search >= 2
                and remaining_fetch >= bonus_fetch
            )
            search_grant = 2 if bonus else 1
            search_grant = max(0, min(int(search_grant), int(remaining_search)))
            if search_grant <= 0:
                return _TrackAllocation(
                    search_grant=0,
                    fetch_grant=0,
                    max_queries_per_round=1,
                    bonus=False,
                )
            minimum_fetch_for_grant = int(search_grant * fetch_per_search_floor)
            max_fetch_affordable = int(
                remaining_fetch
                - max(0, int(remaining_search) - int(search_grant))
                * int(fetch_per_search_floor)
            )
            if max_fetch_affordable < minimum_fetch_for_grant and search_grant > 1:
                search_grant = 1
                minimum_fetch_for_grant = int(search_grant * fetch_per_search_floor)
                max_fetch_affordable = int(
                    remaining_fetch
                    - max(0, int(remaining_search) - int(search_grant))
                    * int(fetch_per_search_floor)
                )
            if max_fetch_affordable < minimum_fetch_for_grant:
                return _TrackAllocation(
                    search_grant=0,
                    fetch_grant=0,
                    max_queries_per_round=1,
                    bonus=False,
                )
            fetch_target = minimum_fetch_for_grant + (bonus_fetch if bonus else 0)
            fetch_grant = max(
                0,
                min(
                    int(fetch_target),
                    int(max_fetch_affordable),
                    int(remaining_fetch),
                ),
            )
            if fetch_grant <= 0:
                return _TrackAllocation(
                    search_grant=0,
                    fetch_grant=0,
                    max_queries_per_round=1,
                    bonus=False,
                )
            reservation_state.search_reserved += int(search_grant)
            reservation_state.fetch_reserved += int(fetch_grant)
            target_width = baseline_width
            if bonus:
                target_width = max(
                    baseline_width,
                    min(bonus_width, max(baseline_width, int(width_hint))),
                )
            return _TrackAllocation(
                search_grant=int(search_grant),
                fetch_grant=int(fetch_grant),
                max_queries_per_round=max(
                    1,
                    int(
                        min(
                            search_grant,
                            target_width,
                        )
                    ),
                ),
                bonus=bool(bonus),
            )

    async def _refresh_orchestrator_if_needed(
        self,
        *,
        root: ResearchStepContext,
        track_map: dict[str, ResearchStepContext],
        state: _OrchestratorState,
        state_lock: anyio.Lock,
        force: bool = False,
    ) -> None:
        if not self._orchestrator_enabled(root):
            return
        global_search_used = int(root.parallel.global_search_used)
        if (
            not force
            and state.priorities
            and state.last_global_search_used >= 0
            and global_search_used - state.last_global_search_used
            < int(state.refresh_interval_search_calls)
        ):
            return
        async with state_lock:
            global_search_used = int(root.parallel.global_search_used)
            if (
                not force
                and state.priorities
                and state.last_global_search_used >= 0
                and global_search_used - state.last_global_search_used
                < int(state.refresh_interval_search_calls)
            ):
                return
            payload = await self._run_track_orchestrator(
                root=root,
                track_map=track_map,
            )
            if payload is None:
                return
            priorities: dict[str, float] = {}
            width_hints: dict[str, int] = {}
            for item in payload.priorities:
                question_id = clean_whitespace(item.question_id)
                if not question_id:
                    continue
                priorities[question_id] = min(1.0, max(0.0, float(item.priority_score)))
                width_hints[question_id] = max(1, min(2, int(item.query_width_hint)))
            if priorities:
                state.priorities = priorities
                state.query_width_hints = width_hints
                state.rationale = clean_whitespace(payload.rationale)
                state.last_global_search_used = global_search_used
                await self.emit_tracking_event(
                    event_name="research.orchestrator.updated",
                    request_id=root.request_id,
                    stage="loop",
                    attrs={
                        "mode_depth_profile": str(root.runtime.mode_depth.mode_key),
                        "prioritized_tracks": int(len(priorities)),
                        "global_search_used": int(global_search_used),
                    },
                )

    async def _run_track_orchestrator(
        self,
        *,
        root: ResearchStepContext,
        track_map: dict[str, ResearchStepContext],
    ) -> _TrackOrchestratorOutputPayload | None:
        model = resolve_research_model(
            ctx=root,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        try:
            result = await self._llm.create(
                model=model,
                messages=self._build_track_orchestrator_messages(
                    root=root,
                    track_map=track_map,
                ),
                response_format=_TrackOrchestratorOutputPayload,
                retries=int(self.settings.research.llm_self_heal_retries),
            )
            return result.data
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="research.orchestrator.error",
                request_id=root.request_id,
                stage="loop",
                status="error",
                error_code="research_track_orchestrator_failed",
                error_type=type(exc).__name__,
                attrs={
                    "model": str(model),
                    "message": str(exc),
                },
            )
            return None

    def _build_track_orchestrator_messages(
        self,
        *,
        root: ResearchStepContext,
        track_map: dict[str, ResearchStepContext],
    ) -> list[dict[str, str]]:
        budget = root.runtime.budget
        snapshot_markdown = self._build_track_snapshot_markdown(track_map)
        return build_track_orchestrator_prompt_messages(
            mode_depth_profile=str(root.runtime.mode_depth.mode_key),
            core_question=self._resolve_core_question(root),
            search_remaining=max(
                0,
                int(budget.max_search_calls) - int(root.runtime.search_calls),
            ),
            fetch_remaining=max(
                0,
                int(budget.max_fetch_calls) - int(root.runtime.fetch_calls),
            ),
            track_snapshots_markdown=snapshot_markdown,
        )

    async def _run_gap_closure_passes(
        self,
        *,
        root: ResearchStepContext,
        card: ResearchQuestionCard,
        track_ctx: ResearchStepContext,
        track_map: dict[str, ResearchStepContext],
        reservation_state: _BudgetReservationState,
        orchestrator_state: _OrchestratorState,
        budget_lock: anyio.Lock,
        orchestrator_lock: anyio.Lock,
    ) -> ResearchStepContext:
        mode_depth = root.runtime.mode_depth
        if not bool(mode_depth.enable_gap_closure_pass):
            return track_ctx
        pass_cap = max(0, int(mode_depth.gap_closure_passes))
        if pass_cap <= 0:
            return track_ctx
        local_ctx = track_ctx
        for pass_index in range(pass_cap):
            if self._global_budget_exhausted(root):
                break
            planned_queries = await self._plan_gap_closure_queries(
                root=root,
                card=card,
                track_ctx=local_ctx,
                pass_index=pass_index,
            )
            if not planned_queries:
                break
            local_ctx.plan.next_queries = list(planned_queries)
            local_ctx.runtime.stop = False
            local_ctx.runtime.stop_reason = ""
            alloc = await self._reserve_track_allocation(
                root=root,
                track_ctx=local_ctx,
                card=card,
                track_map=track_map,
                reservation_state=reservation_state,
                orchestrator_state=orchestrator_state,
                budget_lock=budget_lock,
                orchestrator_lock=orchestrator_lock,
            )
            if self._allocation_blocked(alloc):
                local_ctx.runtime.stop = True
                local_ctx.runtime.stop_reason = "global_budget_exhausted"
                break
            before_search = int(local_ctx.runtime.search_calls)
            before_fetch = int(local_ctx.runtime.fetch_calls)
            delta_search = 0
            delta_fetch = 0
            try:
                self._apply_track_round_budget(
                    track_ctx=local_ctx,
                    alloc=alloc,
                    base_budget=root.runtime.budget,
                )
                local_ctx = await self._round_runner.run(local_ctx)
                delta_search = max(
                    0, int(local_ctx.runtime.search_calls) - int(before_search)
                )
                delta_fetch = max(
                    0, int(local_ctx.runtime.fetch_calls) - int(before_fetch)
                )
            finally:
                await self._commit_track_usage(
                    root=root,
                    alloc=alloc,
                    delta_search=delta_search,
                    delta_fetch=delta_fetch,
                    reservation_state=reservation_state,
                    budget_lock=budget_lock,
                )
            root.runtime.gap_closure_passes_applied += 1
            await self.emit_tracking_event(
                event_name="research.gap_closure.applied",
                request_id=root.request_id,
                stage="loop",
                attrs={
                    "question_id": str(card.question_id),
                    "pass_index": int(pass_index + 1),
                    "planned_queries": int(len(planned_queries)),
                },
            )
        return local_ctx

    async def _plan_gap_closure_queries(
        self,
        *,
        root: ResearchStepContext,
        card: ResearchQuestionCard,
        track_ctx: ResearchStepContext,
        pass_index: int,
    ) -> list[str]:
        latest = self._latest_round(track_ctx)
        core_question = clean_whitespace(
            self._resolve_core_question(track_ctx, fallback=card.question)
        )
        fallback_queries = self._fallback_gap_queries(
            core_question=core_question,
            missing_entities=(latest.missing_entities if latest is not None else []),
            critical_gaps=(latest.critical_gaps if latest is not None else 0),
            limit=int(track_ctx.runtime.budget.max_queries_per_round),
        )
        model = resolve_research_model(
            ctx=root,
            stage="plan",
            fallback=self.settings.answer.plan.use_model,
        )
        try:
            result = await self._llm.create(
                model=model,
                messages=self._build_gap_closure_messages(
                    root=root,
                    card=card,
                    track_ctx=track_ctx,
                    pass_index=pass_index,
                ),
                response_format=_GapClosureOutputPayload,
                retries=int(self.settings.research.llm_self_heal_retries),
            )
            queries = self._normalize_queries(
                result.data.queries,
                limit=int(track_ctx.runtime.budget.max_queries_per_round),
            )
            if queries:
                return queries
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="research.gap_closure.error",
                request_id=root.request_id,
                stage="loop",
                status="error",
                error_code="research_gap_closure_plan_failed",
                error_type=type(exc).__name__,
                attrs={
                    "question_id": str(card.question_id),
                    "pass_index": int(pass_index + 1),
                    "model": str(model),
                    "message": str(exc),
                },
            )
        return fallback_queries

    def _build_gap_closure_messages(
        self,
        *,
        root: ResearchStepContext,
        card: ResearchQuestionCard,
        track_ctx: ResearchStepContext,
        pass_index: int,
    ) -> list[dict[str, str]]:
        latest = self._latest_round(track_ctx)
        return build_gap_closure_prompt_messages(
            core_question=self._resolve_core_question(
                track_ctx, fallback=card.question
            ),
            question_id=str(card.question_id),
            pass_index=int(pass_index),
            confidence=float(latest.confidence) if latest is not None else 0.0,
            coverage_ratio=float(latest.coverage_ratio) if latest is not None else 0.0,
            unresolved_conflicts=(
                int(latest.unresolved_conflicts) if latest is not None else 0
            ),
            critical_gaps=int(latest.critical_gaps) if latest is not None else 0,
            missing_entities=list(latest.missing_entities)
            if latest is not None
            else [],
            round_notes_markdown=self._build_track_snapshot_markdown(
                {card.question_id: track_ctx}
            ),
        )

    async def _commit_track_usage(
        self,
        *,
        root: ResearchStepContext,
        alloc: _TrackAllocation,
        delta_search: int,
        delta_fetch: int,
        reservation_state: _BudgetReservationState,
        budget_lock: anyio.Lock,
    ) -> None:
        async with budget_lock:
            reservation_state.search_reserved = max(
                0,
                int(reservation_state.search_reserved)
                - int(max(0, alloc.search_grant)),
            )
            reservation_state.fetch_reserved = max(
                0,
                int(reservation_state.fetch_reserved) - int(max(0, alloc.fetch_grant)),
            )
            actual_search = max(
                0, min(int(delta_search), int(max(0, alloc.search_grant)))
            )
            actual_fetch = max(0, min(int(delta_fetch), int(max(0, alloc.fetch_grant))))
            root.parallel.global_search_used += int(actual_search)
            root.parallel.global_fetch_used += int(actual_fetch)
            root.runtime.search_calls += int(actual_search)
            root.runtime.fetch_calls += int(actual_fetch)

    def _resolve_question_cards(
        self, ctx: ResearchStepContext
    ) -> list[ResearchQuestionCard]:
        cap = max(1, int(ctx.runtime.mode_depth.max_question_cards_effective))
        raw_cards = list(ctx.parallel.question_cards)
        if not raw_cards:
            raw_cards = [
                ResearchQuestionCard(
                    question_id="q1",
                    question=self._resolve_core_question(ctx),
                    priority=5,
                    seed_queries=[self._resolve_core_question(ctx)],
                    evidence_focus=[],
                    expected_gain="Fallback single-track research.",
                )
            ]
        out: list[ResearchQuestionCard] = []
        seen_question: set[str] = set()
        seen_id: set[str] = set()
        for item in raw_cards:
            question = clean_whitespace(item.question or "")
            if not question:
                continue
            key = question.casefold()
            if key in seen_question:
                continue
            seen_question.add(key)
            base_id = clean_whitespace(item.question_id or "")
            question_id = base_id or f"q{len(out) + 1}"
            while question_id in seen_id:
                question_id = f"q{len(out) + 1}"
            seen_id.add(question_id)
            priority = max(1, min(5, int(item.priority)))
            seed_queries = [
                token for x in item.seed_queries if (token := clean_whitespace(x))
            ]
            if not seed_queries:
                seed_queries = [question]
            out.append(
                ResearchQuestionCard(
                    question_id=question_id,
                    question=question,
                    priority=priority,
                    seed_queries=seed_queries,
                    evidence_focus=[
                        token
                        for x in item.evidence_focus
                        if (token := clean_whitespace(x))
                    ],
                    expected_gain=clean_whitespace(item.expected_gain or ""),
                )
            )
            if len(out) >= cap:
                break
        return out

    def _build_track_context(
        self,
        *,
        root: ResearchStepContext,
        card: ResearchQuestionCard,
    ) -> ResearchStepContext:
        request = root.request.model_copy(
            update={
                "themes": card.question,
                "json_schema": None,
            }
        )
        budget = root.runtime.budget.model_copy(deep=True)
        track = ResearchStepContext(
            settings=root.settings,
            request=request,
            request_id=f"{root.request_id}:track:{card.question_id}",
        )
        track.runtime = ResearchRuntimeState(
            mode_depth=root.runtime.mode_depth.model_copy(deep=True),
            budget=budget,
            search_calls=0,
            fetch_calls=0,
            no_progress_rounds=0,
            gap_closure_passes_applied=0,
            density_gate_passes_applied=0,
            target_output_chars=0,
            output_length_ratio_vs_target=0.0,
            stop=False,
            stop_reason="",
            round_index=0,
        )
        track.plan = root.plan.model_copy(deep=True)
        track.plan.theme_plan.core_question = card.question
        track.plan.next_queries = list(card.seed_queries or [card.question])
        track.parallel.question_cards = [card.model_copy(deep=True)]
        track.parallel.track_results = []
        track.parallel.global_search_budget = 0
        track.parallel.global_fetch_budget = 0
        track.parallel.global_search_used = 0
        track.parallel.global_fetch_used = 0
        track.notes = [f"Track initialized for question `{card.question_id}`."]
        return track

    def _score_track(
        self, track_ctx: ResearchStepContext, card: ResearchQuestionCard
    ) -> float:
        latest = self._latest_round(track_ctx)
        confidence = 0.0
        gaps = 5
        conflicts = 3
        if latest is not None:
            confidence = min(1.0, max(0.0, float(latest.confidence)))
            gaps = max(0, int(latest.critical_gaps))
            conflicts = max(0, int(latest.unresolved_conflicts))
        gap_norm = min(gaps, 5) / 5
        conflict_norm = min(conflicts, 3) / 3
        priority = max(1, min(5, int(card.priority)))
        return (
            0.35 * (float(priority) / 5.0)
            + 0.35 * (1.0 - confidence)
            + 0.20 * gap_norm
            + 0.10 * conflict_norm
        )

    def _apply_track_round_budget(
        self,
        *,
        track_ctx: ResearchStepContext,
        alloc: _TrackAllocation,
        base_budget: ResearchBudgetState,
    ) -> None:
        search_grant = max(0, int(alloc.search_grant))
        fetch_grant = max(1, int(alloc.fetch_grant))
        budget = track_ctx.runtime.budget
        budget.max_rounds = int(base_budget.max_rounds)
        budget.max_search_calls = int(track_ctx.runtime.search_calls + search_grant)
        budget.max_fetch_calls = int(track_ctx.runtime.fetch_calls + fetch_grant)
        budget.max_results_per_search = int(base_budget.max_results_per_search)
        budget.max_queries_per_round = max(
            1, int(min(search_grant, alloc.max_queries_per_round))
        )
        budget.max_fetch_per_round = int(fetch_grant)
        budget.stop_confidence = float(base_budget.stop_confidence)
        budget.min_coverage_ratio = float(base_budget.min_coverage_ratio)
        budget.max_unresolved_conflicts = int(base_budget.max_unresolved_conflicts)

    async def _finalize_track(
        self,
        *,
        card: ResearchQuestionCard,
        track_ctx: ResearchStepContext,
    ) -> ResearchTrackResult:
        rendered = await self._render_step.run(track_ctx)
        latest = self._latest_round(rendered)
        insight_card = self._coerce_insight_card(rendered.output.structured)
        await self.emit_tracking_event(
            event_name="research.loop.subreport",
            request_id=rendered.request_id,
            stage="subreport",
            attrs={
                "question_id": card.question_id,
                "rounds": int(len(rendered.rounds)),
                "stop_reason": clean_whitespace(rendered.runtime.stop_reason or "")
                or "n/a",
                "subreport_chars": int(len(str(rendered.output.content or ""))),
                "has_insight_card": bool(insight_card is not None),
            },
        )
        return ResearchTrackResult(
            question_id=card.question_id,
            question=card.question,
            stop_reason=clean_whitespace(rendered.runtime.stop_reason or ""),
            rounds=int(len(rendered.rounds)),
            search_calls=int(rendered.runtime.search_calls),
            fetch_calls=int(rendered.runtime.fetch_calls),
            confidence=float(latest.confidence) if latest is not None else 0.0,
            coverage_ratio=float(latest.coverage_ratio) if latest is not None else 0.0,
            unresolved_conflicts=(
                int(latest.unresolved_conflicts) if latest is not None else 0
            ),
            subreport_markdown=str(rendered.output.content or ""),
            track_insight_card=insight_card,
            key_findings=self._extract_key_findings(rendered),
        )

    def _build_failed_track_result(
        self,
        *,
        card: ResearchQuestionCard,
        track_ctx: ResearchStepContext,
    ) -> ResearchTrackResult:
        latest = self._latest_round(track_ctx)
        insight_card = self._coerce_insight_card(track_ctx.output.structured)
        return ResearchTrackResult(
            question_id=card.question_id,
            question=card.question,
            stop_reason=clean_whitespace(track_ctx.runtime.stop_reason or ""),
            rounds=int(len(track_ctx.rounds)),
            search_calls=int(track_ctx.runtime.search_calls),
            fetch_calls=int(track_ctx.runtime.fetch_calls),
            confidence=float(latest.confidence) if latest is not None else 0.0,
            coverage_ratio=float(latest.coverage_ratio) if latest is not None else 0.0,
            unresolved_conflicts=(
                int(latest.unresolved_conflicts) if latest is not None else 0
            ),
            subreport_markdown=str(track_ctx.output.content or ""),
            track_insight_card=insight_card,
            key_findings=self._extract_key_findings(track_ctx),
        )

    def _extract_key_findings(self, track_ctx: ResearchStepContext) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for round_state in track_ctx.rounds[-8:]:
            for token in str(round_state.overview_summary or "").split("|"):
                item = clean_whitespace(token)
                if not item:
                    continue
                key = item.casefold()
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
            for token in str(round_state.content_summary or "").split("|"):
                item = clean_whitespace(token)
                if not item:
                    continue
                key = item.casefold()
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
        for note in track_ctx.notes[-12:]:
            item = clean_whitespace(note)
            if not item:
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= 8:
                break
        return out[:8]

    def _latest_round(
        self, track_ctx: ResearchStepContext
    ) -> ResearchRoundState | None:
        if track_ctx.rounds:
            return track_ctx.rounds[-1]
        return track_ctx.current_round

    def _global_budget_exhausted(self, ctx: ResearchStepContext) -> bool:
        return bool(
            int(ctx.parallel.global_search_used)
            >= int(ctx.parallel.global_search_budget)
            or int(ctx.parallel.global_fetch_used)
            >= int(ctx.parallel.global_fetch_budget)
        )

    def _allocation_blocked(self, alloc: _TrackAllocation) -> bool:
        return bool(
            int(alloc.fetch_grant) <= 0
            or (int(alloc.search_grant) <= 0 and not bool(alloc.fetch_only))
        )

    def _can_allocate_fetch_only_round(self, track_ctx: ResearchStepContext) -> bool:
        if int(track_ctx.runtime.budget.max_fetch_calls) <= int(
            track_ctx.runtime.fetch_calls
        ):
            return False
        next_round_index = int(track_ctx.runtime.round_index) + 1
        if next_round_index <= 1:
            return False
        expected_round = int(next_round_index) - 1
        if int(track_ctx.plan.last_round_link_candidates_round) != int(expected_round):
            return False
        return bool(track_ctx.plan.last_round_link_candidates)

    def _orchestrator_enabled(self, ctx: ResearchStepContext) -> bool:
        mode_depth = ctx.runtime.mode_depth
        return bool(mode_depth.enable_llm_track_orchestrator)

    def _normalize_queries(self, raw: list[str], *, limit: int) -> list[str]:
        return normalize_strings(raw, limit=max(1, int(limit)))

    def _fallback_gap_queries(
        self,
        *,
        core_question: str,
        missing_entities: list[str],
        critical_gaps: int,
        limit: int,
    ) -> list[str]:
        base = clean_whitespace(core_question)
        items = normalize_strings(missing_entities, limit=8)
        candidates: list[str] = []
        for entity in items:
            if base:
                candidates.append(f"{base} {entity}")
            else:
                candidates.append(entity)
        if base and int(critical_gaps) > 0:
            candidates.append(f"{base} constraints edge cases latest")
        if base:
            candidates.append(base)
        return merge_strings(candidates, [], limit=max(1, int(limit)))

    def _build_track_snapshot_markdown(
        self, track_map: dict[str, ResearchStepContext]
    ) -> str:
        lines: list[str] = []
        for question_id, track_ctx in track_map.items():
            latest = self._latest_round(track_ctx)
            lines.extend(
                [
                    f"### {question_id}",
                    f"- question: {self._resolve_core_question(track_ctx)}",
                    f"- rounds: {len(track_ctx.rounds)}",
                    f"- search_calls: {int(track_ctx.runtime.search_calls)}",
                    f"- fetch_calls: {int(track_ctx.runtime.fetch_calls)}",
                    (
                        f"- confidence: {float(latest.confidence):.3f}"
                        if latest is not None
                        else "- confidence: 0.000"
                    ),
                    (
                        f"- coverage_ratio: {float(latest.coverage_ratio):.3f}"
                        if latest is not None
                        else "- coverage_ratio: 0.000"
                    ),
                    (
                        f"- unresolved_conflicts: {int(latest.unresolved_conflicts)}"
                        if latest is not None
                        else "- unresolved_conflicts: 0"
                    ),
                    (
                        f"- critical_gaps: {int(latest.critical_gaps)}"
                        if latest is not None
                        else "- critical_gaps: 0"
                    ),
                ]
            )
        return "\n".join(lines).strip() or "- (none)"

    def _resolve_core_question(
        self, track_ctx: ResearchStepContext, *, fallback: str = ""
    ) -> str:
        question = clean_whitespace(
            track_ctx.plan.theme_plan.core_question
            or fallback
            or track_ctx.request.themes
        )
        return question or clean_whitespace(track_ctx.request.themes)

    def _coerce_insight_card(
        self, raw: object | None
    ) -> TrackInsightCardPayload | None:
        if raw is None:
            return None
        if isinstance(raw, TrackInsightCardPayload):
            return raw.model_copy(deep=True)
        if isinstance(raw, dict):
            try:
                return TrackInsightCardPayload.model_validate(raw)
            except Exception:  # noqa: S112
                return None
        return None


__all__ = ["ResearchLoopStep"]
