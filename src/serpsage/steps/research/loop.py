from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import (
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

        card_map = {card.question_id: card for card in cards}
        track_map: dict[str, ResearchStepContext] = {
            card.question_id: self._build_track_context(root=ctx, card=card)
            for card in cards
        }
        result_map: dict[str, ResearchTrackResult] = {}
        finalized: set[str] = set()
        cycle = 0

        while True:
            active_ids = [
                question_id
                for question_id, item in track_map.items()
                if not bool(item.runtime.stop)
            ]
            if not active_ids:
                break

            if self._global_budget_exhausted(ctx):
                await self._force_stop_tracks(
                    root=ctx,
                    card_map=card_map,
                    track_map=track_map,
                    active_ids=active_ids,
                    reason="global_budget_exhausted",
                    finalized=finalized,
                    result_map=result_map,
                )
                break

            cycle += 1
            scores = {
                question_id: self._score_track(
                    track_map[question_id], card_map[question_id]
                )
                for question_id in active_ids
            }
            allocations = self._allocate_grants(
                root=ctx,
                active_ids=active_ids,
                scores=scores,
            )
            runnable_ids = [
                question_id
                for question_id in active_ids
                if allocations[question_id].search_grant > 0
            ]
            print(
                "[research.loop] cycle_plan",
                json.dumps(
                    {
                        "cycle": int(cycle),
                        "active_tracks": active_ids,
                        "scores": scores,
                        "allocations": {
                            key: {
                                "search_grant": int(value.search_grant),
                                "fetch_grant": int(value.fetch_grant),
                                "max_queries_per_round": int(
                                    value.max_queries_per_round
                                ),
                                "bonus": bool(value.bonus),
                            }
                            for key, value in allocations.items()
                        },
                        "global_search_used": int(ctx.parallel.global_search_used),
                        "global_search_budget": int(ctx.parallel.global_search_budget),
                        "global_fetch_used": int(ctx.parallel.global_fetch_used),
                        "global_fetch_budget": int(ctx.parallel.global_fetch_budget),
                    },
                    ensure_ascii=False,
                ),
            )

            if not runnable_ids:
                await self._force_stop_tracks(
                    root=ctx,
                    card_map=card_map,
                    track_map=track_map,
                    active_ids=active_ids,
                    reason="global_budget_exhausted",
                    finalized=finalized,
                    result_map=result_map,
                )
                break

            run_batch: list[ResearchStepContext] = []
            meta: list[tuple[str, int, int]] = []
            for question_id in runnable_ids:
                track_ctx = track_map[question_id]
                alloc = allocations[question_id]
                self._apply_track_round_budget(
                    track_ctx=track_ctx,
                    alloc=alloc,
                    base_budget=ctx.runtime.budget,
                )
                run_batch.append(track_ctx)
                meta.append(
                    (
                        question_id,
                        int(track_ctx.runtime.search_calls),
                        int(track_ctx.runtime.fetch_calls),
                    )
                )

            out_batch = await self._round_runner.run_batch(run_batch)
            for index, out in enumerate(out_batch):
                question_id, before_search, before_fetch = meta[index]
                track_map[question_id] = out
                delta_search = max(
                    0, int(out.runtime.search_calls) - int(before_search)
                )
                delta_fetch = max(0, int(out.runtime.fetch_calls) - int(before_fetch))
                ctx.parallel.global_search_used += int(delta_search)
                ctx.parallel.global_fetch_used += int(delta_fetch)
                ctx.runtime.search_calls += int(delta_search)
                ctx.runtime.fetch_calls += int(delta_fetch)
                if out.errors:
                    ctx.errors.extend(out.errors)
                if out.runtime.stop and question_id not in finalized:
                    result_map[question_id] = await self._finalize_track(
                        card=card_map[question_id],
                        track_ctx=out,
                    )
                    finalized.add(question_id)
                print(
                    "[research.loop] cycle_track_end",
                    json.dumps(
                        {
                            "cycle": int(cycle),
                            "question_id": question_id,
                            "search_delta": int(delta_search),
                            "fetch_delta": int(delta_fetch),
                            "track_search_calls": int(out.runtime.search_calls),
                            "track_fetch_calls": int(out.runtime.fetch_calls),
                            "track_stop": bool(out.runtime.stop),
                            "track_stop_reason": str(out.runtime.stop_reason or ""),
                        },
                        ensure_ascii=False,
                    ),
                )

            if self._global_budget_exhausted(ctx):
                remaining = [
                    question_id
                    for question_id, item in track_map.items()
                    if not bool(item.runtime.stop)
                ]
                await self._force_stop_tracks(
                    root=ctx,
                    card_map=card_map,
                    track_map=track_map,
                    active_ids=remaining,
                    reason="global_budget_exhausted",
                    finalized=finalized,
                    result_map=result_map,
                )
                break

        for card in cards:
            question_id = card.question_id
            track_ctx = track_map.get(question_id)
            if track_ctx is None:
                continue
            if question_id not in finalized:
                if not track_ctx.runtime.stop:
                    track_ctx.runtime.stop = True
                    if self._global_budget_exhausted(ctx):
                        track_ctx.runtime.stop_reason = "global_budget_exhausted"
                    else:
                        track_ctx.runtime.stop_reason = (
                            track_ctx.runtime.stop_reason or "all_tracks_completed"
                        )
                result_map[question_id] = await self._finalize_track(
                    card=card,
                    track_ctx=track_ctx,
                )
                finalized.add(question_id)

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
        span.set_attr("global_search_used", int(ctx.parallel.global_search_used))
        span.set_attr("global_search_budget", int(ctx.parallel.global_search_budget))
        span.set_attr("global_fetch_used", int(ctx.parallel.global_fetch_used))
        span.set_attr("global_fetch_budget", int(ctx.parallel.global_fetch_budget))
        span.set_attr("stop_reason", str(ctx.runtime.stop_reason or ""))
        print(
            "[research.loop] finished",
            json.dumps(
                {
                    "tracks_total": int(len(cards)),
                    "tracks_finished": int(len(ctx.parallel.track_results)),
                    "global_search_used": int(ctx.parallel.global_search_used),
                    "global_search_budget": int(ctx.parallel.global_search_budget),
                    "global_fetch_used": int(ctx.parallel.global_fetch_used),
                    "global_fetch_budget": int(ctx.parallel.global_fetch_budget),
                    "stop_reason": str(ctx.runtime.stop_reason or ""),
                },
                ensure_ascii=False,
            ),
        )
        return ctx

    def _resolve_question_cards(
        self, ctx: ResearchStepContext
    ) -> list[ResearchQuestionCard]:
        cap = max(1, int(self.settings.research.parallel.question_card_cap))
        raw_cards = (
            list(ctx.parallel.question_cards)
            if ctx.parallel.question_cards
            else list(ctx.plan.question_cards)
        )
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
        track.plan.question_cards = [card.model_copy(deep=True)]
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

    def _allocate_grants(
        self,
        *,
        root: ResearchStepContext,
        active_ids: list[str],
        scores: dict[str, float],
    ) -> dict[str, _TrackAllocation]:
        baseline_fetch = max(1, int(root.runtime.budget.max_fetch_per_round))
        bonus_fetch = baseline_fetch + max(1, baseline_fetch // 2)
        baseline_width = max(
            1, int(self.settings.research.parallel.baseline_query_width)
        )
        bonus_width = max(
            baseline_width,
            int(self.settings.research.parallel.bonus_query_width),
        )
        allocations: dict[str, _TrackAllocation] = {
            question_id: _TrackAllocation(
                search_grant=0,
                fetch_grant=baseline_fetch,
                max_queries_per_round=baseline_width,
                bonus=False,
            )
            for question_id in active_ids
        }
        remaining_search = max(
            0,
            int(root.parallel.global_search_budget)
            - int(root.parallel.global_search_used),
        )
        if remaining_search <= 0:
            return allocations

        sorted_ids = sorted(
            active_ids, key=lambda item: scores.get(item, 0.0), reverse=True
        )
        baseline_count = min(remaining_search, len(sorted_ids))
        for question_id in sorted_ids[:baseline_count]:
            allocations[question_id].search_grant = 1
            remaining_search -= 1

        if remaining_search <= 0 or baseline_count <= 0:
            return allocations

        bonus_ratio = float(self.settings.research.parallel.bonus_ratio)
        top_k = max(1, int(math.ceil(float(len(sorted_ids)) * bonus_ratio)))
        bonus_candidates = sorted_ids[:baseline_count][:top_k]
        for question_id in bonus_candidates:
            if remaining_search <= 0:
                break
            alloc = allocations[question_id]
            alloc.search_grant += 1
            alloc.fetch_grant = bonus_fetch
            alloc.max_queries_per_round = bonus_width
            alloc.bonus = True
            remaining_search -= 1
        return allocations

    def _apply_track_round_budget(
        self,
        *,
        track_ctx: ResearchStepContext,
        alloc: _TrackAllocation,
        base_budget,
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

    async def _force_stop_tracks(
        self,
        *,
        root: ResearchStepContext,
        card_map: dict[str, ResearchQuestionCard],
        track_map: dict[str, ResearchStepContext],
        active_ids: list[str],
        reason: str,
        finalized: set[str],
        result_map: dict[str, ResearchTrackResult],
    ) -> None:
        for question_id in active_ids:
            track_ctx = track_map[question_id]
            if not track_ctx.runtime.stop:
                track_ctx.runtime.stop = True
                track_ctx.runtime.stop_reason = reason
            if question_id in finalized:
                continue
            result_map[question_id] = await self._finalize_track(
                card=card_map[question_id],
                track_ctx=track_ctx,
            )
            finalized.add(question_id)
            if track_ctx.errors:
                root.errors.extend(track_ctx.errors)

    async def _finalize_track(
        self,
        *,
        card: ResearchQuestionCard,
        track_ctx: ResearchStepContext,
    ) -> ResearchTrackResult:
        rendered = await self._render_step.run(track_ctx)
        latest = self._latest_round(rendered)
        result = ResearchTrackResult(
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
            errors=[item.model_copy(deep=True) for item in rendered.errors],
        )
        print(
            "[research.loop] track_finalized",
            json.dumps(
                {
                    "question_id": result.question_id,
                    "question": result.question,
                    "stop_reason": result.stop_reason,
                    "rounds": int(result.rounds),
                    "search_calls": int(result.search_calls),
                    "fetch_calls": int(result.fetch_calls),
                    "confidence": float(result.confidence),
                    "coverage_ratio": float(result.coverage_ratio),
                    "unresolved_conflicts": int(result.unresolved_conflicts),
                },
                ensure_ascii=False,
            ),
        )
        return result

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
