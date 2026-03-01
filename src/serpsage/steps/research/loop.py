from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing_extensions import override

import anyio

from serpsage.models.errors import AppError
from serpsage.models.pipeline import (
    ResearchBudgetState,
    ResearchQuestionCard,
    ResearchRuntimeState,
    ResearchStepContext,
    ResearchTrackResult,
)
from serpsage.steps.base import StepBase
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.steps.base import RunnerBase
    from serpsage.telemetry.base import SpanBase


@dataclass(slots=True)
class _TrackAllocation:
    search_grant: int
    fetch_grant: int
    max_queries_per_round: int
    bonus: bool = False


@dataclass(slots=True)
class _BudgetReservationState:
    search_reserved: int = 0
    fetch_reserved: int = 0


class ResearchLoopStep(StepBase[ResearchStepContext]):
    span_name = "step.research_loop"

    def __init__(
        self,
        *,
        rt: Runtime,
        round_runner: RunnerBase[ResearchStepContext],
        render_step: StepBase[ResearchStepContext],
    ) -> None:
        super().__init__(rt=rt)
        self._round_runner = round_runner
        self._render_step = render_step
        self.bind_deps(round_runner, render_step)

    @override
    async def run_inner(
        self, ctx: ResearchStepContext, *, span: SpanBase
    ) -> ResearchStepContext:
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
        budget_lock = anyio.Lock()
        result_map_lock = anyio.Lock()
        error_lock = anyio.Lock()

        async with anyio.create_task_group() as tg:
            for card in cards:
                tg.start_soon(
                    self._run_track_worker,
                    ctx,
                    card,
                    track_map[card.question_id],
                    result_map,
                    reservation_state,
                    budget_lock,
                    result_map_lock,
                    error_lock,
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

        span.set_attr("tracks_total", int(len(cards)))
        span.set_attr("tracks_finished", int(len(ctx.parallel.track_results)))
        span.set_attr("stop_reason", str(ctx.runtime.stop_reason or ""))
        return ctx

    async def _run_track_worker(
        self,
        root: ResearchStepContext,
        card: ResearchQuestionCard,
        track_ctx: ResearchStepContext,
        result_map: dict[str, ResearchTrackResult],
        reservation_state: _BudgetReservationState,
        budget_lock: anyio.Lock,
        result_map_lock: anyio.Lock,
        error_lock: anyio.Lock,
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
                    reservation_state=reservation_state,
                    budget_lock=budget_lock,
                )
                if alloc.search_grant <= 0 or alloc.fetch_grant <= 0:
                    local_ctx.runtime.stop = True
                    local_ctx.runtime.stop_reason = "global_budget_exhausted"
                    break

                before_search = int(local_ctx.runtime.search_calls)
                before_fetch = int(local_ctx.runtime.fetch_calls)
                before_errors = int(len(local_ctx.errors))
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
                    local_ctx.errors.append(
                        AppError(
                            code="research_loop_track_failed",
                            message=str(exc),
                            details={"question_id": question_id},
                        )
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
                await self._propagate_track_errors(
                    root=root,
                    track_ctx=local_ctx,
                    before_count=before_errors,
                    error_lock=error_lock,
                )

            if not local_ctx.runtime.stop:
                local_ctx.runtime.stop = True
                if self._global_budget_exhausted(root):
                    local_ctx.runtime.stop_reason = "global_budget_exhausted"
                else:
                    local_ctx.runtime.stop_reason = (
                        local_ctx.runtime.stop_reason or "all_tracks_completed"
                    )

            before_finalize_errors = int(len(local_ctx.errors))
            track_result = await self._finalize_track(
                card=card,
                track_ctx=local_ctx,
            )
            await self._propagate_track_errors(
                root=root,
                track_ctx=local_ctx,
                before_count=before_finalize_errors,
                error_lock=error_lock,
            )
        except Exception as exc:  # noqa: BLE001
            local_ctx.runtime.stop = True
            local_ctx.runtime.stop_reason = (
                local_ctx.runtime.stop_reason or "track_worker_failed"
            )
            local_ctx.errors.append(
                AppError(
                    code="research_loop_worker_failed",
                    message=str(exc),
                    details={"question_id": question_id},
                )
            )
            await self._propagate_track_errors(
                root=root,
                track_ctx=local_ctx,
                before_count=max(0, int(len(local_ctx.errors)) - 1),
                error_lock=error_lock,
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
        reservation_state: _BudgetReservationState,
        budget_lock: anyio.Lock,
    ) -> _TrackAllocation:
        fetch_per_search_floor = max(
            1,
            int(root.runtime.budget.max_fetch_calls)
            // max(1, int(root.runtime.budget.max_search_calls)),
        )
        bonus_fetch = 1
        baseline_width = max(
            1, int(self.settings.research.parallel.baseline_query_width)
        )
        bonus_width = max(
            baseline_width,
            int(self.settings.research.parallel.bonus_query_width),
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
            if remaining_search <= 0 or remaining_fetch <= 0:
                return _TrackAllocation(
                    search_grant=0,
                    fetch_grant=0,
                    max_queries_per_round=1,
                    bonus=False,
                )

            score = self._score_track(track_ctx, card)
            bonus_ratio = max(
                0.0,
                min(1.0, float(self.settings.research.parallel.bonus_ratio)),
            )
            bonus_threshold = max(0.0, min(1.0, 1.0 - bonus_ratio))
            bonus = bool(
                score >= bonus_threshold
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
            return _TrackAllocation(
                search_grant=int(search_grant),
                fetch_grant=int(fetch_grant),
                max_queries_per_round=max(
                    1,
                    int(
                        min(
                            search_grant,
                            bonus_width if bonus else baseline_width,
                        )
                    ),
                ),
                bonus=bool(bonus),
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
                int(reservation_state.search_reserved) - int(max(0, alloc.search_grant)),
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

    async def _propagate_track_errors(
        self,
        *,
        root: ResearchStepContext,
        track_ctx: ResearchStepContext,
        before_count: int,
        error_lock: anyio.Lock,
    ) -> None:
        if int(before_count) >= int(len(track_ctx.errors)):
            return
        new_errors = [
            item.model_copy(deep=True)
            for item in track_ctx.errors[int(before_count) :]
        ]
        if not new_errors:
            return
        async with error_lock:
            root.errors.extend(new_errors)

    def _resolve_question_cards(
        self, ctx: ResearchStepContext
    ) -> list[ResearchQuestionCard]:
        cap = max(1, int(self.settings.research.parallel.question_card_cap))
        raw_cards = list(ctx.parallel.question_cards)
        if not raw_cards:
            raw_cards = [
                ResearchQuestionCard(
                    question_id="q1",
                    question=ctx.plan.core_question or ctx.request.themes,
                    priority=5,
                    seed_queries=[ctx.plan.core_question or ctx.request.themes],
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
            budget=budget,
            search_calls=0,
            fetch_calls=0,
            no_progress_rounds=0,
            stop=False,
            stop_reason="",
            round_index=0,
        )
        track.plan = root.plan.model_copy(deep=True)
        track.plan.core_question = card.question
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
        print(
            (
                "[research][loop][subreport] "
                f"request_id={rendered.request_id} "
                f"question_id={card.question_id} "
                f"rounds={int(len(rendered.rounds))} "
                f"stop_reason={clean_whitespace(rendered.runtime.stop_reason or '') or 'n/a'} "
                f"subreport_chars={int(len(str(rendered.output.content or '')))}"
            ),
            flush=True,
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
            key_findings=self._extract_key_findings(rendered),
        )

    def _build_failed_track_result(
        self,
        *,
        card: ResearchQuestionCard,
        track_ctx: ResearchStepContext,
    ) -> ResearchTrackResult:
        latest = self._latest_round(track_ctx)
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
            key_findings=self._extract_key_findings(track_ctx),
        )

    def _extract_key_findings(self, track_ctx: ResearchStepContext) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for round_state in track_ctx.rounds[-8:]:
            for token in str(round_state.abstract_summary or "").split("|"):
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

    def _latest_round(self, track_ctx: ResearchStepContext):
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


__all__ = ["ResearchLoopStep"]
