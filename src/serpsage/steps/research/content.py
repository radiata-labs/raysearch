from __future__ import annotations

from datetime import UTC, datetime
from typing_extensions import override

from serpsage.components.llm.base import LLMClientBase
from serpsage.components.provider.base import SearchProviderBase
from serpsage.components.provider.blend import (
    build_engine_selection_context,
    resolve_engine_selection_routes,
)
from serpsage.components.rank.base import RankerBase
from serpsage.dependencies import Depends
from serpsage.models.steps.research import (
    ContentConflictPayload,
    ContentOutputPayload,
    ResearchStepContext,
)
from serpsage.models.steps.search import QuerySourceSpec
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import build_content_prompt_messages
from serpsage.steps.research.rank import rerank_research_sources
from serpsage.steps.research.schema import build_content_schema
from serpsage.steps.research.search import pick_sources_by_ids, sort_source_ids_by_score
from serpsage.steps.research.utils import resolve_research_model


class ResearchContentStep(StepBase[ResearchStepContext]):
    llm: LLMClientBase = Depends()
    ranker: RankerBase = Depends()
    provider: SearchProviderBase = Depends()

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        if ctx.run.stop or ctx.run.current is None:
            return ctx
        review_source_window = max(1, ctx.run.limits.review_source_window)
        source_ids = list(
            ctx.run.current.need_content_source_ids
            or ctx.run.current.context_source_ids
        )
        source_ids = sort_source_ids_by_score(
            ctx=ctx,
            source_ids=source_ids,
        )[:review_source_window]
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
        selected_sources = await rerank_research_sources(
            ctx=ctx,
            ranker=self.ranker,
            sources=selected_sources,
            query=ctx.task.question,
        )
        source_ids = [item.source_id for item in selected_sources]
        model = resolve_research_model(
            settings=self.settings,
            stage="content",
            fallback=self.settings.answer.generate.use_model,
        )
        routes = resolve_engine_selection_routes(
            settings=self.settings,
            subsystem="research",
            provider=self.provider,
        )
        engine_selection_context = build_engine_selection_context(routes=routes)
        try:
            chat_result = await self.llm.create(
                model=model,
                messages=build_content_prompt_messages(
                    ctx=ctx,
                    selected_sources=selected_sources,
                    source_ids=source_ids,
                    now_utc=now_utc,
                    engine_selection_context=engine_selection_context,
                ),
                response_format=ContentOutputPayload,
                format_override=build_content_schema(
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
                    "prompt_tokens": int(chat_result.usage.prompt_tokens),
                    "completion_tokens": int(chat_result.usage.completion_tokens),
                    "total_tokens": int(chat_result.usage.total_tokens),
                },
            )
        except Exception as exc:  # noqa: BLE001
            await self.tracker.error(
                name="research.content.failed",
                request_id=ctx.request_id,
                step="research.content",
                error_code="research_content_review_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "round_index": ctx.run.current.round_index,
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
        await self.tracker.info(
            name="research.style.applied",
            request_id=ctx.request_id,
            step="research.content",
            data={
                "success": True,
                "sources_reviewed": len(selected_sources),
                "resolved_findings": len(payload.resolved_findings),
                "remaining_gaps": len(payload.remaining_gaps),
                "confidence": ctx.run.current.confidence,
            },
        )
        await self.tracker.debug(
            name="research.style.applied.detail",
            request_id=ctx.request_id,
            step="research.content",
            data={
                "success": True,
                "report_style_selected": ctx.task.style,
                "style_applied_stage": "content",
                "mode_depth_profile": ctx.run.limits.mode_key,
                "review_source_window_effective": review_source_window,
                "source_ids": source_ids,
                "resolved_findings": payload.resolved_findings,
                "missing_entities": payload.missing_entities,
                "remaining_gaps": payload.remaining_gaps,
                "unresolved_conflict_topics": unresolved_topics,
                "next_queries": [
                    item.model_dump(mode="json")
                    for item in ctx.run.current.next_queries
                ],
                "query_strategy": payload.next_query_strategy,
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
        left: list[QuerySourceSpec],
        right: list[QuerySourceSpec],
        limit: int,
    ) -> list[QuerySourceSpec]:
        seen: set[tuple[str, tuple[str, ...]]] = set()
        merged: list[QuerySourceSpec] = []
        for item in [*left, *right]:
            key = (
                item.query.casefold(),
                tuple(item.include_sources),
            )
            if key in seen:
                continue
            seen.add(key)
            merged.append(item)
            if len(merged) >= max(1, limit):
                break
        return merged


__all__ = ["ResearchContentStep"]
