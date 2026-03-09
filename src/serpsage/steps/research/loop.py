from __future__ import annotations

import math
from typing_extensions import override

from serpsage.app.tokens import RESEARCH_ROUND_RUNNER, RESEARCH_SUBREPORT_STEP
from serpsage.components.llm.base import LLMClientBase
from serpsage.dependencies import Inject
from serpsage.models.app.response import ResearchResponse
from serpsage.models.steps.research import (
    ResearchBudgetTier,
    ResearchQuestionCard,
    ResearchResult,
    ResearchRound,
    ResearchRun,
    ResearchStepContext,
    ResearchTrackResult,
    ResearchTrackRuntime,
    TrackInsightCardPayload,
)
from serpsage.steps.base import RunnerBase, StepBase


class ResearchLoopStep(StepBase[ResearchStepContext]):
    _EXTENSION_MULTIPLIERS: dict[str, float] = {
        "low": 0.2,
        "medium": 0.4,
        "high": 0.6,
    }

    llm: LLMClientBase = Inject()
    round_runner: RunnerBase[ResearchStepContext] = Inject(RESEARCH_ROUND_RUNNER)
    render_step: StepBase[ResearchStepContext] = Inject(RESEARCH_SUBREPORT_STEP)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        question_cards = self._resolve_question_cards(ctx)
        ctx.result.tracks = []
        if not question_cards:
            ctx.run.stop = True
            ctx.run.stop_reason = "no_question_cards"
            return ctx
        track_map = {
            card.question_id: self._build_track_context(root=ctx, card=card)
            for card in question_cards
        }
        self._initialize_track_budgets(
            root=ctx, cards=question_cards, track_map=track_map
        )
        result_map: dict[str, ResearchTrackResult] = {}
        while True:
            unfinished_cards = [
                card
                for card in question_cards
                if self._runtime_for(ctx, card.question_id).state
                not in {"completed", "stopped"}
            ]
            if not unfinished_cards:
                break
            progress_made = False
            for card in unfinished_cards:
                runtime = self._runtime_for(ctx, card.question_id)
                track_ctx = track_map[card.question_id]
                if runtime.state == "waiting_for_budget":
                    if not self._grant_waiting_budget(
                        root=ctx,
                        track_ctx=track_ctx,
                        runtime=runtime,
                    ):
                        runtime.waiting_rounds += 1
                        track_ctx.run.track_runtime = runtime.model_copy(deep=True)
                        continue
                before_search = track_ctx.run.search_calls
                before_fetch = track_ctx.run.fetch_calls
                track_ctx.run.stop = False
                track_ctx.run.stop_reason = ""
                if track_ctx.run.current is not None:
                    track_ctx.run.current.stop = False
                    track_ctx.run.current.stop_reason = ""
                try:
                    track_ctx = await self.round_runner.run(track_ctx)
                except Exception as exc:  # noqa: BLE001
                    await self.emit_tracking_event(
                        event_name="research.loop.error",
                        request_id=ctx.request_id,
                        stage="track",
                        status="error",
                        error_code="research_loop_track_failed",
                        error_type=type(exc).__name__,
                        attrs={
                            "question_id": card.question_id,
                            "message": str(exc),
                        },
                    )
                    raise
                track_map[card.question_id] = track_ctx
                delta_search = max(0, track_ctx.run.search_calls - before_search)
                delta_fetch = max(0, track_ctx.run.fetch_calls - before_fetch)
                self._record_budget_usage(
                    root=ctx,
                    runtime=runtime,
                    delta_search=delta_search,
                    delta_fetch=delta_fetch,
                )
                track_ctx.run.track_runtime = runtime.model_copy(deep=True)
                progress_made = progress_made or delta_search > 0 or delta_fetch > 0
                runtime.completed_rounds = len(track_ctx.run.history)
                if self._track_is_waiting(track_ctx):
                    runtime.state = "waiting_for_budget"
                    runtime.waiting_reason = track_ctx.run.stop_reason or "budget_wait"
                    runtime.stop_reason = runtime.waiting_reason
                    track_ctx.run.track_runtime = runtime.model_copy(deep=True)
                    continue
                if track_ctx.run.stop:
                    runtime.state = "completed"
                    runtime.stop_reason = track_ctx.run.stop_reason or "completed"
                    track_ctx.run.track_runtime = runtime.model_copy(deep=True)
                    result_map[card.question_id] = await self._finalize_track(
                        card=card,
                        track_ctx=track_ctx,
                        runtime=runtime,
                    )
                    progress_made = True
                    continue
                runtime.state = "active"
                runtime.waiting_reason = ""
                runtime.stop_reason = ""
                track_ctx.run.track_runtime = runtime.model_copy(deep=True)
            if progress_made:
                continue
            if self._apply_budget_escalation(root=ctx, cards=unfinished_cards):
                continue
            for card in unfinished_cards:
                runtime = self._runtime_for(ctx, card.question_id)
                runtime.state = "stopped"
                runtime.stop_reason = runtime.stop_reason or "global_budget_exhausted"
                track_ctx = track_map[card.question_id]
                track_ctx.run.stop = True
                track_ctx.run.stop_reason = runtime.stop_reason
                track_ctx.run.track_runtime = runtime.model_copy(deep=True)
                result_map[card.question_id] = await self._finalize_track(
                    card=card,
                    track_ctx=track_ctx,
                    runtime=runtime,
                )
            break
        ctx.result.tracks = [
            result_map[card.question_id]
            for card in question_cards
            if card.question_id in result_map
        ]
        ctx.run.stop = True
        ctx.run.stop_reason = (
            "all_tracks_completed"
            if all(
                item.stop_reason != "global_budget_exhausted"
                for item in ctx.result.tracks
            )
            else "global_budget_exhausted"
        )
        return ctx

    def _resolve_question_cards(
        self,
        ctx: ResearchStepContext,
    ) -> list[ResearchQuestionCard]:
        return [
            item.model_copy(deep=True)
            for item in ctx.task.cards[
                : max(1, ctx.run.limits.max_question_cards_effective)
            ]
        ]

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
        track = ResearchStepContext(
            settings=root.settings,
            request=request,
            request_id=f"{root.request_id}:track:{card.question_id}",
            response=ResearchResponse(
                request_id=f"{root.request_id}:track:{card.question_id}",
                content="",
                structured=None,
            ),
        )
        track.task = root.task.model_copy(deep=True)
        track.task.question = card.question
        track.task.cards = [card.model_copy(deep=True)]
        track.run = ResearchRun(
            mode=root.run.mode,
            limits=root.run.limits.model_copy(deep=True),
            search_calls=0,
            fetch_calls=0,
            stop=False,
            stop_reason="",
            round_index=0,
            next_queries=list(card.seed_queries),
            link_candidates=[],
            link_candidates_round=0,
            notes=[f"Track initialized for question `{card.question_id}`."],
            current=None,
            history=[],
            global_search_budget=0,
            global_fetch_budget=0,
            global_search_used=0,
            global_fetch_used=0,
            track_runtime=ResearchTrackRuntime(question_id=card.question_id),
        )
        track.knowledge = root.knowledge.model_copy(deep=True)
        track.result = ResearchResult(content="", structured=None, tracks=[])
        return track

    def _initialize_track_budgets(
        self,
        *,
        root: ResearchStepContext,
        cards: list[ResearchQuestionCard],
        track_map: dict[str, ResearchStepContext],
    ) -> None:
        card_count = max(1, len(cards))
        search_shares = self._distribute_evenly(
            total=root.run.global_search_budget,
            count=card_count,
        )
        fetch_shares = self._distribute_evenly(
            total=root.run.global_fetch_budget,
            count=card_count,
        )
        runtimes: dict[str, ResearchTrackRuntime] = {}
        for index, card in enumerate(cards):
            track_ctx = track_map[card.question_id]
            runtime = ResearchTrackRuntime(
                question_id=card.question_id,
                state=(
                    "active"
                    if search_shares[index] > 0 or fetch_shares[index] > 0
                    else "waiting_for_budget"
                ),
                current_budget_tier="base",
                allocated_search_calls=search_shares[index],
                allocated_fetch_calls=fetch_shares[index],
                last_budget_event="baseline_allocated",
            )
            track_ctx.run.track_runtime = runtime.model_copy(deep=True)
            track_ctx.run.limits.max_search_calls = max(0, search_shares[index])
            track_ctx.run.limits.max_fetch_calls = max(0, fetch_shares[index])
            track_ctx.run.limits.max_queries_per_round = max(
                1,
                min(
                    root.run.limits.max_queries_per_round,
                    root.run.limits.round_search_budget,
                    search_shares[index],
                ),
            )
            track_ctx.run.track_runtime = runtime.model_copy(deep=True)
            runtimes[card.question_id] = runtime
        root.run.track_runtimes = runtimes
        root.run.budget_events.append(
            "baseline_allocated:"
            f"search={root.run.global_search_budget};fetch={root.run.global_fetch_budget}"
        )
        root.run.budget_ledger.extension_multiplier = self._extension_multiplier(root)

    def _grant_waiting_budget(
        self,
        *,
        root: ResearchStepContext,
        track_ctx: ResearchStepContext,
        runtime: ResearchTrackRuntime,
    ) -> bool:
        allow_reclaim = any(
            item.state in {"completed", "stopped"}
            for item in root.run.track_runtimes.values()
            if item.question_id != runtime.question_id
        )
        if (
            not allow_reclaim
            and not root.run.restored_budget_applied
            and not root.run.extension_budget_applied
        ):
            return False
        remaining_search = max(
            0, root.run.global_search_budget - root.run.global_search_used
        )
        remaining_fetch = max(
            0, root.run.global_fetch_budget - root.run.global_fetch_used
        )
        search_needed = 0
        fetch_needed = 0
        if (
            track_ctx.run.current is not None
            and track_ctx.run.current.pending_search_jobs
        ):
            search_needed = max(
                1, min(root.run.limits.round_search_budget, remaining_search)
            )
        if (
            track_ctx.run.current is not None
            and track_ctx.run.current.search_fetched_candidates
        ):
            fetch_needed = max(
                1, min(root.run.limits.round_fetch_budget, remaining_fetch)
            )
        if search_needed <= 0 and fetch_needed <= 0:
            return False
        track_ctx.run.limits.max_search_calls += search_needed
        track_ctx.run.limits.max_fetch_calls += fetch_needed
        if search_needed > 0:
            track_ctx.run.limits.max_queries_per_round = max(
                1,
                min(
                    root.run.limits.max_queries_per_round,
                    root.run.limits.round_search_budget,
                    max(1, search_needed),
                ),
            )
        runtime.allocated_search_calls += search_needed
        runtime.allocated_fetch_calls += fetch_needed
        runtime.state = "active"
        runtime.waiting_reason = ""
        runtime.current_budget_tier = self._current_budget_tier(root)
        runtime.last_budget_event = f"resume_allocated:search={search_needed};fetch={fetch_needed};tier={runtime.current_budget_tier}"
        track_ctx.run.track_runtime = runtime.model_copy(deep=True)
        root.run.budget_events.append(
            f"{runtime.question_id}:{runtime.last_budget_event}"
        )
        return True

    def _apply_budget_escalation(
        self,
        *,
        root: ResearchStepContext,
        cards: list[ResearchQuestionCard],
    ) -> bool:
        if not cards:
            return False
        complexity = root.task.complexity
        original_search = root.run.budget_ledger.original_search_budget
        original_fetch = root.run.budget_ledger.original_fetch_budget
        if complexity in {"low", "medium"} and not root.run.restored_budget_applied:
            root.run.restored_budget_applied = True
            root.run.global_search_budget += original_search
            root.run.global_fetch_budget += original_fetch
            root.run.budget_ledger.restore_used = True
            root.run.budget_ledger.restored_budget.search_total = original_search
            root.run.budget_ledger.restored_budget.fetch_total = original_fetch
            root.run.budget_events.append(
                f"restored_budget_applied:search={original_search};fetch={original_fetch}"
            )
            return True
        if (
            root.run.mode in {"research", "research-pro"}
            and not root.run.extension_budget_applied
        ):
            multiplier = self._extension_multiplier(root)
            extension_search = max(1, math.ceil(original_search * multiplier))
            extension_fetch = max(1, math.ceil(original_fetch * multiplier))
            root.run.extension_budget_applied = True
            root.run.global_search_budget += extension_search
            root.run.global_fetch_budget += extension_fetch
            root.run.budget_ledger.extension_used = True
            root.run.budget_ledger.extension_multiplier = multiplier
            root.run.budget_ledger.extension_budget.search_total = extension_search
            root.run.budget_ledger.extension_budget.fetch_total = extension_fetch
            root.run.budget_events.append(
                f"extension_budget_applied:search={extension_search};fetch={extension_fetch};multiplier={multiplier:.2f}"
            )
            return True
        return False

    def _extension_multiplier(self, root: ResearchStepContext) -> float:
        return self._EXTENSION_MULTIPLIERS.get(root.task.complexity, 0.4)

    def _record_budget_usage(
        self,
        *,
        root: ResearchStepContext,
        runtime: ResearchTrackRuntime,
        delta_search: int,
        delta_fetch: int,
    ) -> None:
        root.run.global_search_used += delta_search
        root.run.global_fetch_used += delta_fetch
        root.run.search_calls += delta_search
        root.run.fetch_calls += delta_fetch
        runtime.used_search_calls += delta_search
        runtime.used_fetch_calls += delta_fetch
        tier = runtime.current_budget_tier
        if tier == "restored":
            root.run.budget_ledger.restored_budget.search_used += delta_search
            root.run.budget_ledger.restored_budget.fetch_used += delta_fetch
        elif tier == "extension":
            root.run.budget_ledger.extension_budget.search_used += delta_search
            root.run.budget_ledger.extension_budget.fetch_used += delta_fetch
        else:
            root.run.budget_ledger.base_budget.search_used += delta_search
            root.run.budget_ledger.base_budget.fetch_used += delta_fetch

    def _track_is_waiting(self, track_ctx: ResearchStepContext) -> bool:
        current = track_ctx.run.current
        if current is None:
            return False
        return bool(
            current.waiting_for_budget
            or current.pending_search_jobs
            or current.search_fetched_candidates
        )

    def _current_budget_tier(self, root: ResearchStepContext) -> ResearchBudgetTier:
        if root.run.extension_budget_applied:
            return "extension"
        if root.run.restored_budget_applied:
            return "restored"
        return "base"

    async def _finalize_track(
        self,
        *,
        card: ResearchQuestionCard,
        track_ctx: ResearchStepContext,
        runtime: ResearchTrackRuntime,
    ) -> ResearchTrackResult:
        rendered = await self.render_step.run(track_ctx)
        latest_round = self._latest_round(rendered)
        insight_card = (
            rendered.result.structured
            if isinstance(rendered.result.structured, TrackInsightCardPayload)
            else None
        )
        await self.emit_tracking_event(
            event_name="research.loop.subreport",
            request_id=rendered.request_id,
            stage="subreport",
            attrs={
                "question_id": card.question_id,
                "rounds": len(rendered.run.history),
                "stop_reason": rendered.run.stop_reason or "n/a",
                "subreport_chars": len(rendered.result.content),
                "has_insight_card": insight_card is not None,
            },
        )
        return ResearchTrackResult(
            question_id=card.question_id,
            question=card.question,
            stop_reason=runtime.stop_reason or rendered.run.stop_reason,
            rounds=len(rendered.run.history),
            search_calls=rendered.run.search_calls,
            fetch_calls=rendered.run.fetch_calls,
            confidence=float(latest_round.confidence)
            if latest_round is not None
            else 0.0,
            coverage_ratio=float(latest_round.coverage_ratio)
            if latest_round is not None
            else 0.0,
            unresolved_conflicts=(
                latest_round.unresolved_conflicts if latest_round is not None else 0
            ),
            subreport_markdown=rendered.result.content,
            track_insight_card=insight_card,
            key_findings=self._extract_key_findings(rendered),
            budget_tier=runtime.current_budget_tier,
            waiting_rounds=runtime.waiting_rounds,
        )

    def _extract_key_findings(self, track_ctx: ResearchStepContext) -> list[str]:
        findings: list[str] = []
        seen: set[str] = set()
        for round_state in track_ctx.run.history[-8:]:
            for item in round_state.overview_summary.split("|"):
                token = item.strip()
                if not token:
                    continue
                key = token.casefold()
                if key in seen:
                    continue
                seen.add(key)
                findings.append(token)
            for item in round_state.content_summary.split("|"):
                token = item.strip()
                if not token:
                    continue
                key = token.casefold()
                if key in seen:
                    continue
                seen.add(key)
                findings.append(token)
        for note in track_ctx.run.notes[-12:]:
            key = note.casefold()
            if key in seen:
                continue
            seen.add(key)
            findings.append(note)
            if len(findings) >= 8:
                break
        return findings[:8]

    def _latest_round(
        self,
        track_ctx: ResearchStepContext,
    ) -> ResearchRound | None:
        if track_ctx.run.history:
            return track_ctx.run.history[-1]
        return track_ctx.run.current

    def _runtime_for(
        self,
        root: ResearchStepContext,
        question_id: str,
    ) -> ResearchTrackRuntime:
        runtime = root.run.track_runtimes.get(question_id)
        if runtime is None:
            runtime = ResearchTrackRuntime(question_id=question_id)
            root.run.track_runtimes[question_id] = runtime
        return runtime

    def _distribute_evenly(self, *, total: int, count: int) -> list[int]:
        count = max(1, count)
        base = max(0, total // count)
        remainder = max(0, total % count)
        shares = [base] * count
        for index in range(remainder):
            shares[index % count] += 1
        return shares


__all__ = ["ResearchLoopStep"]
