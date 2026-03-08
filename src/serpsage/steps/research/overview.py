from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.steps.research import (
    OverviewConflictPayload,
    OverviewOutputPayload,
    ResearchSource,
    ResearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import build_overview_prompt_messages
from serpsage.steps.research.schema import build_overview_schema
from serpsage.steps.research.search import (
    pick_sources_by_ids,
    select_context_source_ids,
)
from serpsage.steps.research.utils import resolve_research_model

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime


class ResearchOverviewStep(StepBase[ResearchStepContext]):
    _CONTEXT_NEW_RESULT_TARGET_RATIO = 0.60
    _CONTEXT_MIN_HISTORY_SOURCES = 3

    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        if ctx.runtime.stop or ctx.current_round is None:
            return ctx
        all_sources = list(ctx.corpus.sources)
        if not all_sources:
            ctx.work.overview_review = self._empty_review()
            ctx.work.need_content_source_ids = []
            ctx.current_round.context_source_ids = []
            return ctx
        mode_depth = ctx.runtime.mode_depth
        source_topk = max(1, mode_depth.source_topk)
        new_result_target_ratio, min_history_sources = (
            self._resolve_context_mix_targets(
                ctx=ctx,
                sources=all_sources,
            )
        )
        context_source_ids = select_context_source_ids(
            ctx=ctx,
            round_index=ctx.current_round.round_index,
            topk=source_topk,
            new_result_target_ratio=new_result_target_ratio,
            min_history_sources=min_history_sources,
        )
        ctx.current_round.context_source_ids = list(context_source_ids)
        sources = pick_sources_by_ids(
            sources=all_sources,
            source_ids=context_source_ids,
        )
        if not sources:
            ctx.work.overview_review = self._empty_review()
            ctx.work.need_content_source_ids = []
            return ctx
        model = resolve_research_model(
            ctx=ctx,
            stage="overview",
            fallback=self.settings.answer.generate.use_model,
        )
        try:
            chat_result = await self._llm.create(
                model=model,
                messages=build_overview_prompt_messages(
                    ctx=ctx,
                    sources=sources,
                    now_utc=now_utc,
                ),
                response_format=OverviewOutputPayload,
                format_override=build_overview_schema(
                    max_queries=ctx.runtime.budget.max_queries_per_round
                ),
                retries=self.settings.research.llm_self_heal_retries,
            )
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="research.overview.error",
                request_id=ctx.request_id,
                stage="overview_review",
                status="error",
                error_code="research_overview_review_failed",
                error_type=type(exc).__name__,
                attrs={
                    "round_index": ctx.current_round.round_index,
                    "message": str(exc),
                },
            )
            raise
        payload = chat_result.data
        ctx.work.overview_review = payload
        ctx.work.need_content_source_ids = self._resolve_need_content_source_ids(
            payload=payload,
            context_source_ids=context_source_ids,
        )
        if payload.findings:
            ctx.notes.extend(payload.findings[:3])
            ctx.current_round.overview_summary = " | ".join(payload.findings[:3])
        ctx.current_round.confidence = payload.confidence
        ctx.current_round.query_strategy = payload.next_query_strategy
        if payload.covered_subthemes:
            ctx.corpus.coverage_state.covered_subthemes = self._merge_preserving_order(
                left=ctx.corpus.coverage_state.covered_subthemes,
                right=list(payload.covered_subthemes),
            )
        total = max(1, ctx.corpus.coverage_state.total_subthemes)
        ctx.current_round.coverage_ratio = min(
            1.0,
            len(ctx.corpus.coverage_state.covered_subthemes) / total,
        )
        ctx.current_round.entity_coverage_complete = payload.entity_coverage_complete
        ctx.current_round.missing_entities = list(payload.missing_entities)
        unresolved_topics = self._extract_unresolved_topics(
            payload.conflict_arbitration
        )
        ctx.current_round.unresolved_conflicts = len(unresolved_topics)
        ctx.current_round.unresolved_conflict_topics = unresolved_topics
        ctx.current_round.critical_gaps = len(payload.critical_gaps)
        ctx.work.next_queries = list(
            payload.next_queries[: ctx.runtime.budget.max_queries_per_round]
        )
        await self.emit_tracking_event(
            event_name="research.style.applied",
            request_id=ctx.request_id,
            stage="overview_review",
            attrs={
                "report_style_selected": ctx.plan.theme_plan.report_style,
                "style_applied_stage": "overview",
                "mode_depth_profile": mode_depth.mode_key,
                "source_topk_effective": source_topk,
                "overview_new_result_target_ratio": new_result_target_ratio,
                "overview_min_history_sources": min_history_sources,
            },
        )
        return ctx

    def _empty_review(self) -> OverviewOutputPayload:
        return OverviewOutputPayload(
            findings=[],
            conflict_arbitration=[],
            covered_subthemes=[],
            entity_coverage_complete=False,
            covered_entities=[],
            missing_entities=[],
            critical_gaps=[],
            confidence=0.0,
            need_content_source_ids=[],
            next_query_strategy="coverage",
            next_queries=[],
            stop=False,
        )

    def _resolve_need_content_source_ids(
        self,
        *,
        payload: OverviewOutputPayload,
        context_source_ids: list[int],
    ) -> list[int]:
        if not payload.need_content_source_ids:
            return []
        allowed_source_ids = set(context_source_ids)
        return [
            source_id
            for source_id in payload.need_content_source_ids
            if source_id in allowed_source_ids
        ]

    def _extract_unresolved_topics(
        self,
        conflicts: list[OverviewConflictPayload],
    ) -> list[str]:
        return [item.topic for item in conflicts if item.status == "unresolved"]

    def _merge_preserving_order(
        self,
        *,
        left: list[str],
        right: list[str],
    ) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for item in [*left, *right]:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
        return merged

    def _resolve_context_mix_targets(
        self,
        *,
        ctx: ResearchStepContext,
        sources: list[ResearchSource],
    ) -> tuple[float, int]:
        mode_key = ctx.runtime.mode_depth.mode_key
        if mode_key == "research-fast":
            base_ratio = 0.70
            base_min_history = 1
        elif mode_key == "research-pro":
            base_ratio = 0.55
            base_min_history = 3
        else:
            base_ratio = self._CONTEXT_NEW_RESULT_TARGET_RATIO
            base_min_history = 2
        round_index = ctx.current_round.round_index if ctx.current_round else 0
        new_count = sum(
            1 for item in sources if int(getattr(item, "round_index", 0)) == round_index
        )
        total_count = len(sources)
        history_count = max(0, total_count - new_count)
        if new_count <= 0:
            return 0.0, min(base_min_history, max(1, history_count))
        if history_count <= 0:
            return 1.0, 0
        ratio = base_ratio
        if round_index <= 1:
            ratio = max(ratio, 0.70)
        if new_count <= 2:
            ratio = min(0.85, ratio + 0.15)
        if history_count <= 2:
            ratio = max(0.35, ratio - 0.20)
        min_history = min(max(1, history_count), base_min_history)
        if round_index <= 1:
            min_history = min(min_history, 1)
        return max(0.0, min(1.0, ratio)), max(0, min_history)


__all__ = ["ResearchOverviewStep"]
