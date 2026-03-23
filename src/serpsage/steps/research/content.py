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
    ContentReviewPayload,
    RoundStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import build_content_prompt_messages
from serpsage.steps.research.rank import rerank_research_sources
from serpsage.steps.research.schema import build_content_schema
from serpsage.steps.research.search import pick_sources_by_ids, sort_source_ids_by_score
from serpsage.steps.research.utils import resolve_research_model


class ResearchContentStep(StepBase[RoundStepContext]):
    llm: LLMClientBase = Depends()
    ranker: RankerBase = Depends()
    provider: SearchProviderBase = Depends()

    @override
    async def run_inner(self, ctx: RoundStepContext) -> RoundStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        if ctx.run.stop or ctx.run.current is None:
            return ctx
        if not ctx.run.current.is_review_ready:
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
                response_format=ContentReviewPayload,
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

        # Calculate final confidence from overview + content adjustment
        base_confidence = (
            ctx.run.current.overview_review.confidence
            if ctx.run.current.overview_review
            else 0.0
        )
        final_confidence = max(
            -1.0, min(1.0, base_confidence + payload.confidence_adjustment)
        )

        await self.tracker.info(
            name="research.style.applied",
            request_id=ctx.request_id,
            step="research.content",
            data={
                "success": True,
                "sources_reviewed": len(selected_sources),
                "resolved_findings": len(payload.resolved_findings),
                "remaining_gaps": len(payload.remaining_gaps),
                "confidence": final_confidence,
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
                "remaining_gaps": payload.remaining_gaps,
                "conflict_resolutions": [
                    {"topic": r.topic, "status": r.status}
                    for r in payload.conflict_resolutions
                ],
                "confidence_adjustment": payload.confidence_adjustment,
            },
        )
        return ctx

    def _empty_review(self) -> ContentReviewPayload:
        return ContentReviewPayload(
            resolved_findings=[],
            conflict_resolutions=[],
            remaining_gaps=[],
            confidence_adjustment=0.0,
        )


__all__ = ["ResearchContentStep"]
