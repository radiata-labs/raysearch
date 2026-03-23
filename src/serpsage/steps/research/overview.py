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
    OverviewReviewPayload,
    ResearchSource,
    RoundStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import build_overview_prompt_messages
from serpsage.steps.research.rank import rerank_research_sources
from serpsage.steps.research.schema import build_overview_schema
from serpsage.steps.research.search import (
    pick_sources_by_ids,
    select_context_source_ids,
)
from serpsage.steps.research.utils import resolve_research_model


class ResearchOverviewStep(StepBase[RoundStepContext]):
    _CONTEXT_NEW_RESULT_TARGET_RATIO = 0.60
    _CONTEXT_MIN_HISTORY_SOURCES = 3

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
        all_sources = list(ctx.knowledge.sources)
        if not all_sources:
            ctx.run.current.overview_review = self._empty_review()
            ctx.run.current.need_content_source_ids = []
            ctx.run.current.context_source_ids = []
            return ctx
        mode_depth = ctx.run.limits
        review_source_window = max(1, mode_depth.review_source_window)
        new_result_target_ratio, min_history_sources = (
            self._resolve_context_mix_targets(
                ctx=ctx,
                sources=all_sources,
            )
        )
        context_source_ids = select_context_source_ids(
            ctx=ctx,
            round_index=ctx.run.current.round_index,
            topk=review_source_window,
            new_result_target_ratio=new_result_target_ratio,
            min_history_sources=min_history_sources,
        )
        ctx.run.current.context_source_ids = list(context_source_ids)
        sources = pick_sources_by_ids(
            sources=all_sources,
            source_ids=context_source_ids,
        )
        if not sources:
            ctx.run.current.overview_review = self._empty_review()
            ctx.run.current.need_content_source_ids = []
            return ctx
        sources = await rerank_research_sources(
            ctx=ctx,
            ranker=self.ranker,
            sources=sources,
            query=ctx.task.question,
        )
        ctx.run.current.context_source_ids = [item.source_id for item in sources]
        model = resolve_research_model(
            settings=self.settings,
            stage="overview",
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
                messages=build_overview_prompt_messages(
                    ctx=ctx,
                    sources=sources,
                    now_utc=now_utc,
                    engine_selection_context=engine_selection_context,
                ),
                response_format=OverviewReviewPayload,
                format_override=build_overview_schema(
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
                name="research.overview.failed",
                request_id=ctx.request_id,
                step="research.overview",
                error_code="research_overview_review_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "round_index": ctx.run.current.round_index,
                },
            )
            raise
        payload = chat_result.data
        ctx.run.current.overview_review = payload
        ctx.run.current.need_content_source_ids = self._resolve_need_content_source_ids(
            payload=payload,
        )
        if payload.findings:
            ctx.run.notes.extend(payload.findings[:3])
            ctx.run.current.overview_summary = " | ".join(payload.findings[:3])
        if payload.covered_subthemes:
            ctx.knowledge.covered_subthemes = self._merge_preserving_order(
                left=ctx.knowledge.covered_subthemes,
                right=list(payload.covered_subthemes),
            )
        total = max(1, len(ctx.task.subthemes))
        ctx.run.current.coverage_ratio = min(
            1.0,
            len(ctx.knowledge.covered_subthemes) / total,
        )
        await self.tracker.info(
            name="research.style.applied",
            request_id=ctx.request_id,
            step="research.overview",
            data={
                "success": True,
                "sources_reviewed": len(sources),
                "findings": len(payload.findings),
                "confidence": payload.confidence,
                "coverage_ratio": ctx.run.current.coverage_ratio,
            },
        )
        await self.tracker.debug(
            name="research.style.applied.detail",
            request_id=ctx.request_id,
            step="research.overview",
            data={
                "success": True,
                "report_style_selected": ctx.task.style,
                "style_applied_stage": "overview",
                "mode_depth_profile": mode_depth.mode_key,
                "review_source_window_effective": review_source_window,
                "overview_new_result_target_ratio": new_result_target_ratio,
                "overview_min_history_sources": min_history_sources,
                "context_source_ids": ctx.run.current.context_source_ids,
                "need_content_source_ids": ctx.run.current.need_content_source_ids,
                "covered_subthemes": len(payload.covered_subthemes),
                "missing_entities": payload.missing_entities,
                "conflict_topics": len(payload.conflict_topics),
            },
        )
        return ctx

    def _empty_review(self) -> OverviewReviewPayload:
        return OverviewReviewPayload(
            findings=[],
            conflict_topics=[],
            covered_subthemes=[],
            need_content_source_ids=[],
            missing_entities=[],
            confidence=0.0,
        )

    def _resolve_need_content_source_ids(
        self,
        *,
        payload: OverviewReviewPayload,
    ) -> list[int]:
        if not payload.need_content_source_ids:
            return []
        out: list[int] = []
        seen: set[int] = set()
        for source_id in payload.need_content_source_ids:
            if source_id in seen:
                continue
            seen.add(source_id)
            out.append(source_id)
        return out

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
        ctx: RoundStepContext,
        sources: list[ResearchSource],
    ) -> tuple[float, int]:
        mode_key = ctx.run.limits.mode_key
        if mode_key == "research-fast":
            base_ratio = 0.70
            base_min_history = 1
        elif mode_key == "research-pro":
            base_ratio = 0.55
            base_min_history = 3
        else:
            base_ratio = self._CONTEXT_NEW_RESULT_TARGET_RATIO
            base_min_history = 2
        round_index = ctx.run.current.round_index if ctx.run.current else 0
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
