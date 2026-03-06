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
from serpsage.steps.research.prompt import build_track_orchestrator_prompt_messages
from serpsage.steps.research.utils import resolve_research_model

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
                    latest = self._latest_round(local_ctx)
                    min_rounds = max(
                        1, local_ctx.runtime.mode_depth.min_rounds_per_track
                    )
                    local_ctx.runtime.stop = True
                    local_ctx.runtime.stop_reason = (
                        "stop_ready"
                        if (
                            latest is not None
                            and latest.stop_ready
                            and not latest.remaining_objectives
                            and len(local_ctx.rounds) >= min_rounds
                        )
                        else "global_budget_exhausted"
                    )
                    break
                before_search = local_ctx.runtime.search_calls
                before_fetch = local_ctx.runtime.fetch_calls
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
                        0, local_ctx.runtime.search_calls - before_search
                    )
                    delta_fetch = max(0, local_ctx.runtime.fetch_calls - before_fetch)
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
            root.runtime.budget.max_fetch_calls
            // max(1, root.runtime.budget.max_search_calls),
        )
        bonus_fetch = 1
        baseline_width = max(1, self._BASELINE_QUERY_WIDTH)
        bonus_width = max(
            baseline_width,
            self._BONUS_QUERY_WIDTH,
        )
        await self._refresh_orchestrator_if_needed(
            root=root,
            track_map=track_map,
            state=orchestrator_state,
            state_lock=orchestrator_lock,
        )
        score = self._score_track(track_ctx, card)
        latest = self._latest_round(track_ctx)
        min_rounds = max(1, track_ctx.runtime.mode_depth.min_rounds_per_track)
        track_stop_ready = (
            latest is not None
            and latest.stop_ready
            and not latest.remaining_objectives
            and len(track_ctx.rounds) >= min_rounds
        )
        track_has_objectives = (
            latest is None or bool(latest.remaining_objectives) or not latest.stop_ready
        )
        width_hint = 1
        orchestrator_enabled = self._orchestrator_enabled(root)
        has_orchestrator_priority = False
        if track_stop_ready:
            return _TrackAllocation(
                search_grant=0,
                fetch_grant=0,
                max_queries_per_round=1,
                bonus=False,
            )
        if orchestrator_enabled:
            if card.question_id in orchestrator_state.priorities:
                score = float(
                    orchestrator_state.priorities.get(card.question_id, score)
                )
                has_orchestrator_priority = True
            width_hint = max(
                baseline_width,
                orchestrator_state.query_width_hints.get(card.question_id, 1),
            )
        async with budget_lock:
            remaining_search = max(
                0,
                root.parallel.global_search_budget
                - root.parallel.global_search_used
                - reservation_state.search_reserved,
            )
            remaining_fetch = max(
                0,
                root.parallel.global_fetch_budget
                - root.parallel.global_fetch_used
                - reservation_state.fetch_reserved,
            )
            if remaining_fetch <= 0:
                return _TrackAllocation(
                    search_grant=0,
                    fetch_grant=0,
                    max_queries_per_round=1,
                    bonus=False,
                )
            if remaining_search <= 0:
                if not track_has_objectives or not self._can_allocate_fetch_only_round(
                    track_ctx
                ):
                    return _TrackAllocation(
                        search_grant=0,
                        fetch_grant=0,
                        max_queries_per_round=1,
                        bonus=False,
                    )
                fetch_grant = max(
                    1,
                    remaining_fetch,
                )
                reservation_state.fetch_reserved += fetch_grant
                return _TrackAllocation(
                    search_grant=0,
                    fetch_grant=fetch_grant,
                    max_queries_per_round=1,
                    bonus=False,
                    fetch_only=True,
                )
            bonus_ratio = max(
                0.0,
                min(
                    1.0,
                    float(0.40 if orchestrator_enabled else self._BONUS_RATIO),
                ),
            )
            bonus_threshold = max(0.0, min(1.0, 1.0 - bonus_ratio))
            bonus_by_score = score >= bonus_threshold or (
                has_orchestrator_priority
                and score >= max(0.50, float(bonus_threshold - 0.10))
            )
            bonus_by_width = width_hint >= bonus_width
            bonus = (
                (bonus_by_score or bonus_by_width)
                and track_has_objectives
                and remaining_search >= 2
                and remaining_fetch >= bonus_fetch
            )
            search_grant = 2 if bonus else 1
            search_grant = max(0, min(search_grant, remaining_search))
            if search_grant <= 0:
                return _TrackAllocation(
                    search_grant=0,
                    fetch_grant=0,
                    max_queries_per_round=1,
                    bonus=False,
                )
            minimum_fetch_for_grant = search_grant * fetch_per_search_floor
            max_fetch_affordable = (
                remaining_fetch
                - max(0, remaining_search - search_grant) * fetch_per_search_floor
            )
            if max_fetch_affordable < minimum_fetch_for_grant and search_grant > 1:
                search_grant = 1
                minimum_fetch_for_grant = search_grant * fetch_per_search_floor
                max_fetch_affordable = (
                    remaining_fetch
                    - max(0, remaining_search - search_grant) * fetch_per_search_floor
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
                    fetch_target,
                    max_fetch_affordable,
                    remaining_fetch,
                ),
            )
            if fetch_grant <= 0:
                return _TrackAllocation(
                    search_grant=0,
                    fetch_grant=0,
                    max_queries_per_round=1,
                    bonus=False,
                )
            reservation_state.search_reserved += search_grant
            reservation_state.fetch_reserved += fetch_grant
            target_width = baseline_width
            if bonus:
                target_width = max(
                    baseline_width,
                    min(bonus_width, max(baseline_width, width_hint)),
                )
            return _TrackAllocation(
                search_grant=search_grant,
                fetch_grant=fetch_grant,
                max_queries_per_round=max(
                    1,
                    min(search_grant, target_width),
                ),
                bonus=bonus,
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
        global_search_used = root.parallel.global_search_used
        if (
            not force
            and state.priorities
            and state.last_global_search_used >= 0
            and global_search_used - state.last_global_search_used
            < state.refresh_interval_search_calls
        ):
            return
        async with state_lock:
            global_search_used = root.parallel.global_search_used
            if (
                not force
                and state.priorities
                and state.last_global_search_used >= 0
                and global_search_used - state.last_global_search_used
                < state.refresh_interval_search_calls
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
                question_id = item.question_id.strip()
                if not question_id:
                    continue
                priorities[question_id] = min(1.0, max(0.0, float(item.priority_score)))
                width_hints[question_id] = max(1, min(2, item.query_width_hint))
            if priorities:
                state.priorities = priorities
                state.query_width_hints = width_hints
                state.rationale = payload.rationale.strip()
                state.last_global_search_used = global_search_used
                await self.emit_tracking_event(
                    event_name="research.orchestrator.updated",
                    request_id=root.request_id,
                    stage="loop",
                    attrs={
                        "mode_depth_profile": root.runtime.mode_depth.mode_key,
                        "prioritized_tracks": len(priorities),
                        "global_search_used": global_search_used,
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
                messages=build_track_orchestrator_prompt_messages(
                    root=root,
                    track_map=track_map,
                ),
                response_format=_TrackOrchestratorOutputPayload,
                retries=self.settings.research.llm_self_heal_retries,
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
                    "model": model,
                    "message": str(exc),
                },
            )
            return None

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
                reservation_state.search_reserved - max(0, alloc.search_grant),
            )
            reservation_state.fetch_reserved = max(
                0,
                reservation_state.fetch_reserved - max(0, alloc.fetch_grant),
            )
            actual_search = max(0, min(delta_search, max(0, alloc.search_grant)))
            actual_fetch = max(0, min(delta_fetch, max(0, alloc.fetch_grant)))
            root.parallel.global_search_used += actual_search
            root.parallel.global_fetch_used += actual_fetch
            root.runtime.search_calls += actual_search
            root.runtime.fetch_calls += actual_fetch

    def _resolve_question_cards(
        self, ctx: ResearchStepContext
    ) -> list[ResearchQuestionCard]:
        cap = max(1, ctx.runtime.mode_depth.max_question_cards_effective)
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
            question = item.question.strip()
            if not question:
                continue
            key = question.casefold()
            if key in seen_question:
                continue
            seen_question.add(key)
            base_id = item.question_id.strip()
            question_id = base_id or f"q{len(out) + 1}"
            while question_id in seen_id:
                question_id = f"q{len(out) + 1}"
            seen_id.add(question_id)
            priority = max(1, min(5, item.priority))
            seed_queries = [token for x in item.seed_queries if (token := x.strip())]
            if not seed_queries:
                seed_queries = [question]
            out.append(
                ResearchQuestionCard(
                    question_id=question_id,
                    question=question,
                    priority=priority,
                    seed_queries=seed_queries,
                    evidence_focus=[
                        token for x in item.evidence_focus if (token := x.strip())
                    ],
                    expected_gain=item.expected_gain.strip(),
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
        objective_norm = 1.0
        low_gain_penalty = 0.0
        if latest is not None:
            if latest.stop_ready and not latest.remaining_objectives:
                return 0.0
            confidence = min(1.0, max(0.0, float(latest.confidence)))
            gaps = max(0, latest.critical_gaps)
            conflicts = max(0, latest.unresolved_conflicts)
            objective_norm = min(len(latest.remaining_objectives), 4) / 4
            low_gain_penalty = 0.10 if latest.low_gain_streak >= 2 else 0.0
        gap_norm = min(gaps, 5) / 5
        conflict_norm = min(conflicts, 3) / 3
        priority = max(1, min(5, card.priority))
        score = (
            0.35 * (float(priority) / 5.0)
            + 0.30 * (1.0 - confidence)
            + 0.20 * gap_norm
            + 0.10 * conflict_norm
            + 0.05 * objective_norm
        )
        return max(0.0, score - low_gain_penalty)

    def _apply_track_round_budget(
        self,
        *,
        track_ctx: ResearchStepContext,
        alloc: _TrackAllocation,
        base_budget: ResearchBudgetState,
    ) -> None:
        search_grant = max(0, alloc.search_grant)
        fetch_grant = max(1, alloc.fetch_grant)
        budget = track_ctx.runtime.budget
        budget.max_rounds = base_budget.max_rounds
        budget.max_search_calls = track_ctx.runtime.search_calls + search_grant
        budget.max_fetch_calls = track_ctx.runtime.fetch_calls + fetch_grant
        budget.max_results_per_search = base_budget.max_results_per_search
        budget.max_queries_per_round = max(
            1, min(search_grant, alloc.max_queries_per_round)
        )
        budget.stop_confidence = base_budget.stop_confidence
        budget.min_coverage_ratio = base_budget.min_coverage_ratio

    async def _finalize_track(
        self,
        *,
        card: ResearchQuestionCard,
        track_ctx: ResearchStepContext,
    ) -> ResearchTrackResult:
        rendered = await self._render_step.run(track_ctx)
        latest = self._latest_round(rendered)
        raw_insight = rendered.output.structured
        insight_card = self._coerce_insight_card(
            raw_insight
            if isinstance(raw_insight, TrackInsightCardPayload | dict)
            else None
        )
        await self.emit_tracking_event(
            event_name="research.loop.subreport",
            request_id=rendered.request_id,
            stage="subreport",
            attrs={
                "question_id": card.question_id,
                "rounds": len(rendered.rounds),
                "stop_reason": rendered.runtime.stop_reason or "n/a",
                "subreport_chars": len(rendered.output.content),
                "has_insight_card": insight_card is not None,
            },
        )
        return ResearchTrackResult(
            question_id=card.question_id,
            question=card.question,
            stop_reason=rendered.runtime.stop_reason,
            rounds=len(rendered.rounds),
            search_calls=rendered.runtime.search_calls,
            fetch_calls=rendered.runtime.fetch_calls,
            confidence=float(latest.confidence) if latest is not None else 0.0,
            coverage_ratio=float(latest.coverage_ratio) if latest is not None else 0.0,
            unresolved_conflicts=(
                latest.unresolved_conflicts if latest is not None else 0
            ),
            subreport_markdown=rendered.output.content,
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
        raw_insight = track_ctx.output.structured
        insight_card = self._coerce_insight_card(
            raw_insight
            if isinstance(raw_insight, TrackInsightCardPayload | dict)
            else None
        )
        return ResearchTrackResult(
            question_id=card.question_id,
            question=card.question,
            stop_reason=track_ctx.runtime.stop_reason,
            rounds=len(track_ctx.rounds),
            search_calls=track_ctx.runtime.search_calls,
            fetch_calls=track_ctx.runtime.fetch_calls,
            confidence=float(latest.confidence) if latest is not None else 0.0,
            coverage_ratio=float(latest.coverage_ratio) if latest is not None else 0.0,
            unresolved_conflicts=(
                latest.unresolved_conflicts if latest is not None else 0
            ),
            subreport_markdown=track_ctx.output.content,
            track_insight_card=insight_card,
            key_findings=self._extract_key_findings(track_ctx),
        )

    def _extract_key_findings(self, track_ctx: ResearchStepContext) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for round_state in track_ctx.rounds[-8:]:
            for token in (round_state.overview_summary).split("|"):
                item = token
                if not item:
                    continue
                key = item.casefold()
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
            for token in (round_state.content_summary).split("|"):
                item = token
                if not item:
                    continue
                key = item.casefold()
                if key in seen:
                    continue
                seen.add(key)
                out.append(item)
        for note in track_ctx.notes[-12:]:
            item = note
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
        return (
            ctx.parallel.global_search_used >= ctx.parallel.global_search_budget
            or ctx.parallel.global_fetch_used >= ctx.parallel.global_fetch_budget
        )

    def _allocation_blocked(self, alloc: _TrackAllocation) -> bool:
        return alloc.fetch_grant <= 0 or (
            alloc.search_grant <= 0 and not alloc.fetch_only
        )

    def _can_allocate_fetch_only_round(self, track_ctx: ResearchStepContext) -> bool:
        if track_ctx.runtime.budget.max_fetch_calls <= track_ctx.runtime.fetch_calls:
            return False
        next_round_index = track_ctx.runtime.round_index + 1
        if next_round_index <= 1:
            return False
        expected_round = next_round_index - 1
        if track_ctx.plan.last_round_link_candidates_round != expected_round:
            return False
        return bool(track_ctx.plan.last_round_link_candidates)

    def _orchestrator_enabled(self, ctx: ResearchStepContext) -> bool:
        return ctx.runtime.mode_depth.mode_key != "research-fast"

    def _resolve_core_question(
        self, track_ctx: ResearchStepContext, *, fallback: str = ""
    ) -> str:
        return (
            track_ctx.plan.theme_plan.core_question
            or fallback
            or track_ctx.request.themes
        )

    def _coerce_insight_card(
        self, raw: TrackInsightCardPayload | dict[str, object] | None
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
