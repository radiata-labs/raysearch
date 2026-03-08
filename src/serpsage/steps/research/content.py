from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.steps.research import (
    ContentConflictPayload,
    ContentOutputPayload,
    ResearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import build_content_prompt_messages
from serpsage.steps.research.schema import build_content_schema
from serpsage.steps.research.search import pick_sources_by_ids, sort_source_ids_by_score
from serpsage.steps.research.utils import resolve_research_model

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime


class ResearchContentStep(StepBase[ResearchStepContext]):
    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        if ctx.run.stop or ctx.run.current is None:
            return ctx
        source_topk = max(1, ctx.run.limits.source_topk)
        source_ids = list(
            ctx.run.current.need_content_source_ids
            or ctx.run.current.context_source_ids
        )
        source_ids = sort_source_ids_by_score(
            ctx=ctx,
            source_ids=source_ids,
        )[:source_topk]
        if not source_ids:
            ctx.run.current.content_review = self._empty_review()
            return ctx
        selected_sources = pick_sources_by_ids(
            sources=ctx.knowledge.sources,
            source_ids=source_ids,
        )
        if not selected_sources:
            ctx.run.current.content_review = self._empty_review()
            return ctx
        model = resolve_research_model(
            ctx=ctx,
            stage="content",
            fallback=self.settings.answer.generate.use_model,
        )
        try:
            chat_result = await self._llm.create(
                model=model,
                messages=build_content_prompt_messages(
                    ctx=ctx,
                    selected_sources=selected_sources,
                    source_ids=source_ids,
                    now_utc=now_utc,
                ),
                response_format=ContentOutputPayload,
                format_override=build_content_schema(
                    max_queries=ctx.run.limits.max_queries_per_round
                ),
                retries=self.settings.research.llm_self_heal_retries,
            )
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="research.content.error",
                request_id=ctx.request_id,
                stage="content_review",
                status="error",
                error_code="research_content_review_failed",
                error_type=type(exc).__name__,
                attrs={
                    "round_index": ctx.run.current.round_index,
                    "message": str(exc),
                },
            )
            raise
        payload = chat_result.data
        ctx.run.current.content_review = payload
        if payload.resolved_findings:
            ctx.run.current.content_summary = " | ".join(payload.resolved_findings[:3])
            ctx.run.notes.extend(payload.resolved_findings[:3])
        ctx.run.current.confidence = min(
            1.0,
            max(0.0, ctx.run.current.confidence + payload.confidence_adjustment),
        )
        unresolved_topics = self._merge_conflict_topics(
            baseline=ctx.run.current.unresolved_conflict_topics,
            resolutions=payload.conflict_resolutions,
        )
        ctx.run.current.unresolved_conflicts = len(unresolved_topics)
        ctx.run.current.unresolved_conflict_topics = unresolved_topics
        ctx.run.current.critical_gaps = len(payload.remaining_gaps)
        ctx.run.current.entity_coverage_complete = payload.entity_coverage_complete
        ctx.run.current.missing_entities = list(payload.missing_entities)
        ctx.run.current.next_queries = self._merge_preserving_order(
            left=list(ctx.run.current.next_queries),
            right=list(payload.next_queries),
            limit=ctx.run.limits.max_queries_per_round,
        )
        ctx.run.current.query_strategy = payload.next_query_strategy
        await self.emit_tracking_event(
            event_name="research.style.applied",
            request_id=ctx.request_id,
            stage="content_review",
            attrs={
                "report_style_selected": ctx.task.style,
                "style_applied_stage": "content",
                "mode_depth_profile": ctx.run.limits.mode_key,
                "source_topk_effective": source_topk,
            },
        )
        return ctx

    def _empty_review(self) -> ContentOutputPayload:
        return ContentOutputPayload(
            resolved_findings=[],
            conflict_resolutions=[],
            entity_coverage_complete=False,
            covered_entities=[],
            missing_entities=[],
            remaining_gaps=[],
            confidence_adjustment=0.0,
            next_query_strategy="coverage",
            next_queries=[],
            stop=False,
        )

    def _merge_conflict_topics(
        self,
        *,
        baseline: list[str],
        resolutions: list[ContentConflictPayload],
    ) -> list[str]:
        active: dict[str, str] = {item.casefold(): item for item in baseline}
        for item in resolutions:
            topic_key = item.topic.casefold()
            if item.status in {"resolved", "closed"}:
                active.pop(topic_key, None)
                continue
            active[topic_key] = item.topic
        return list(active.values())

    def _merge_preserving_order(
        self,
        *,
        left: list[str],
        right: list[str],
        limit: int,
    ) -> list[str]:
        seen: set[str] = set()
        merged: list[str] = []
        for item in [*left, *right]:
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= max(1, limit):
                break
        return merged


__all__ = ["ResearchContentStep"]
