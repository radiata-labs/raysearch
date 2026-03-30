from __future__ import annotations

import math
from datetime import UTC, datetime
from typing_extensions import override

from raysearch.components.llm.base import LLMClientBase
from raysearch.components.provider.base import SearchProviderBase
from raysearch.components.provider.blend import (
    build_engine_selection_context,
    resolve_engine_selection_routes,
)
from raysearch.components.rank.base import RankerBase
from raysearch.dependencies import Depends
from raysearch.models.steps.research import ResearchSource, RoundStepContext
from raysearch.models.steps.research.payloads import (
    ContentReviewPayload,
    OverviewReviewPayload,
)
from raysearch.steps.base import StepBase
from raysearch.steps.research.prompt import (
    build_content_prompt_messages,
    build_overview_prompt_messages,
)
from raysearch.steps.research.schema import (
    build_content_schema,
    build_overview_schema,
)
from raysearch.steps.research.utils import (
    pick_sources_by_ids,
    rerank_research_sources,
    resolve_research_model,
)


class ResearchOverviewStep(StepBase[RoundStepContext]):
    _CONTEXT_NEW_RESULT_TARGET_RATIO = 0.60

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
            self._reset_review_state(ctx=ctx)
            return ctx

        review_source_window = max(1, ctx.run.limits.review_source_window)
        new_result_target_ratio, min_history_sources = (
            self._resolve_context_mix_targets(
                ctx=ctx,
                sources=all_sources,
            )
        )
        context_source_ids = self._select_context_source_ids(
            ctx=ctx,
            round_index=ctx.run.current.round_index,
            topk=review_source_window,
            new_result_target_ratio=new_result_target_ratio,
            min_history_sources=min_history_sources,
        )
        overview_sources = pick_sources_by_ids(
            sources=all_sources,
            source_ids=context_source_ids,
        )
        if not overview_sources:
            self._reset_review_state(ctx=ctx)
            return ctx

        overview_sources = await rerank_research_sources(
            ctx=ctx,
            ranker=self.ranker,
            sources=overview_sources,
            query=ctx.task.question,
        )

        overview_payload = await self._run_overview_review(
            ctx=ctx,
            sources=overview_sources,
            now_utc=now_utc,
        )
        ctx.run.current.overview_review = overview_payload
        self._apply_overview_state(ctx=ctx, payload=overview_payload)

        content_sources = self._select_content_sources(
            ctx=ctx,
            overview_sources=overview_sources,
            payload=overview_payload,
            review_source_window=review_source_window,
        )
        content_payload = (
            await self._run_content_review(
                ctx=ctx,
                selected_sources=content_sources,
                now_utc=now_utc,
            )
            if content_sources
            else self._empty_content_review()
        )
        ctx.run.current.content_review = content_payload
        self._apply_content_state(ctx=ctx, payload=content_payload)
        await self.tracker.info(
            name="research.review.applied",
            request_id=ctx.request_id,
            step="research.overview",
            data={
                "success": True,
                "overview_sources_reviewed": len(overview_sources),
                "content_sources_reviewed": len(content_sources),
                "overview_findings": len(overview_payload.findings),
                "resolved_findings": len(content_payload.resolved_findings),
                "confidence": ctx.run.current.confidence,
                "coverage_ratio": ctx.run.current.coverage_ratio,
                "uncertainty_score": ctx.run.current.uncertainty_score,
            },
        )
        await self.tracker.debug(
            name="research.review.applied.detail",
            request_id=ctx.request_id,
            step="research.overview",
            data={
                "success": True,
                "review_source_window_effective": review_source_window,
                "overview_need_content_source_ids": overview_payload.need_content_source_ids,
                "covered_subthemes": len(overview_payload.covered_subthemes),
                "missing_entities": overview_payload.missing_entities,
                "conflict_topics": len(overview_payload.conflict_topics),
                "remaining_gaps": content_payload.remaining_gaps,
                "confidence_score": content_payload.confidence_score,
                "uncertainty_score": content_payload.uncertainty_score,
            },
        )
        return ctx

    async def _run_overview_review(
        self,
        *,
        ctx: RoundStepContext,
        sources: list[ResearchSource],
        now_utc: datetime,
    ) -> OverviewReviewPayload:
        current = ctx.run.current
        round_index = current.round_index if current is not None else 0
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
                data={"round_index": round_index},
            )
            raise
        return chat_result.data

    async def _run_content_review(
        self,
        *,
        ctx: RoundStepContext,
        selected_sources: list[ResearchSource],
        now_utc: datetime,
    ) -> ContentReviewPayload:
        current = ctx.run.current
        round_index = current.round_index if current is not None else 0
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
        source_ids = [source.source_id for source in selected_sources]
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
                step="research.overview",
                error_code="research_content_review_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "round_index": round_index,
                    "source_ids": source_ids,
                },
            )
            raise
        return chat_result.data

    def _apply_overview_state(
        self,
        *,
        ctx: RoundStepContext,
        payload: OverviewReviewPayload,
    ) -> None:
        assert ctx.run.current is not None
        if payload.findings:
            ctx.run.notes.extend(payload.findings[:3])
            ctx.run.current.overview_summary = " | ".join(payload.findings[:3])
        else:
            ctx.run.current.overview_summary = ""
        if payload.covered_subthemes:
            ctx.knowledge.covered_subthemes = self._merge_preserving_order(
                left=ctx.knowledge.covered_subthemes,
                right=list(payload.covered_subthemes),
            )
        ctx.run.current.coverage_ratio = float(payload.coverage_score)

    def _apply_content_state(
        self,
        *,
        ctx: RoundStepContext,
        payload: ContentReviewPayload,
    ) -> None:
        assert ctx.run.current is not None
        if payload.resolved_findings:
            ctx.run.current.content_summary = " | ".join(payload.resolved_findings[:3])
            ctx.run.notes.extend(payload.resolved_findings[:3])
        else:
            ctx.run.current.content_summary = ""
        ctx.run.current.uncertainty_score = float(payload.uncertainty_score)

    def _reset_review_state(self, *, ctx: RoundStepContext) -> None:
        assert ctx.run.current is not None
        ctx.run.current.overview_review = self._empty_overview_review()
        ctx.run.current.content_review = self._empty_content_review()
        ctx.run.current.overview_summary = ""
        ctx.run.current.content_summary = ""
        ctx.run.current.coverage_ratio = 0.0
        ctx.run.current.uncertainty_score = 0.0

    def _select_content_sources(
        self,
        *,
        ctx: RoundStepContext,
        overview_sources: list[ResearchSource],
        payload: OverviewReviewPayload,
        review_source_window: int,
    ) -> list[ResearchSource]:
        content_window = max(
            1,
            min(review_source_window, ctx.run.limits.report_source_batch_size),
        )
        source_by_id = {source.source_id: source for source in overview_sources}
        selected: list[ResearchSource] = []
        seen: set[int] = set()

        for source_id in payload.need_content_source_ids:
            source = source_by_id.get(source_id)
            if source is None or source_id in seen:
                continue
            if not self._source_has_reviewable_content(source):
                continue
            selected.append(source)
            seen.add(source_id)
            if len(selected) >= content_window:
                return selected

        for source in overview_sources:
            if source.source_id in seen:
                continue
            if not self._source_has_reviewable_content(source):
                continue
            selected.append(source)
            seen.add(source.source_id)
            if len(selected) >= content_window:
                break

        return selected

    def _source_has_reviewable_content(self, source: ResearchSource) -> bool:
        return bool((source.content or "").strip() or (source.overview or "").strip())

    def _empty_overview_review(self) -> OverviewReviewPayload:
        return OverviewReviewPayload(
            findings=[],
            conflict_topics=[],
            covered_subthemes=[],
            need_content_source_ids=[],
            missing_entities=[],
            confidence_score=0.0,
            coverage_score=0.0,
        )

    def _empty_content_review(self) -> ContentReviewPayload:
        return ContentReviewPayload(
            resolved_findings=[],
            conflict_resolutions=[],
            remaining_gaps=[],
            confidence_score=0.0,
            uncertainty_score=0.0,
        )

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

    def _select_context_source_ids(
        self,
        *,
        ctx: RoundStepContext,
        round_index: int,
        topk: int,
        new_result_target_ratio: float,
        min_history_sources: int,
    ) -> list[int]:
        limit = max(1, topk)
        ranked_ids = self._resolve_ranked_source_ids(ctx=ctx)
        if not ranked_ids:
            return []

        source_by_id = {source.source_id: source for source in ctx.knowledge.sources}
        new_ids = [
            source_id
            for source_id in ranked_ids
            if source_by_id[source_id].round_index == round_index
        ]
        history_ids = [
            source_id
            for source_id in ranked_ids
            if source_by_id[source_id].round_index != round_index
        ]

        target_new = min(
            len(new_ids),
            int(math.ceil(limit * float(max(0.0, min(1.0, new_result_target_ratio))))),
        )
        selected: list[int] = []
        selected.extend(new_ids[:target_new])
        selected.extend(history_ids[: max(0, limit - len(selected))])

        if len(selected) < limit:
            for source_id in new_ids[target_new:]:
                if source_id in selected:
                    continue
                selected.append(source_id)
                if len(selected) >= limit:
                    break
        if len(selected) < limit:
            for source_id in history_ids:
                if source_id in selected:
                    continue
                selected.append(source_id)
                if len(selected) >= limit:
                    break

        min_history = max(0, min_history_sources)
        history_needed = min(min_history, len(history_ids))
        history_selected = sum(1 for source_id in selected if source_id in history_ids)
        if history_selected < history_needed:
            for source_id in history_ids:
                if source_id in selected:
                    continue
                selected.append(source_id)
                history_selected += 1
                if history_selected >= history_needed:
                    break

        if len(selected) > limit:
            selected = self._trim_to_limit(
                selected=selected,
                source_by_id=source_by_id,
                round_index=round_index,
                limit=limit,
            )

        rank_index = {source_id: index for index, source_id in enumerate(ranked_ids)}
        deduped: list[int] = []
        seen: set[int] = set()
        for source_id in selected:
            if source_id in seen:
                continue
            seen.add(source_id)
            deduped.append(source_id)
        deduped.sort(key=lambda source_id: rank_index.get(source_id, 10**9))
        return deduped[:limit]

    def _resolve_ranked_source_ids(self, *, ctx: RoundStepContext) -> list[int]:
        source_ids: list[int] = []
        source_by_id = {source.source_id: source for source in ctx.knowledge.sources}
        seen_canonical: set[str] = set()

        for source_id in ctx.knowledge.ranked_source_ids:
            source = source_by_id.get(source_id)
            if source is None:
                continue
            canonical = source.canonical_url or source.url
            if not canonical or canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
            source_ids.append(source_id)
        if source_ids:
            return source_ids

        fallback = sorted(
            ctx.knowledge.sources,
            key=lambda item: (
                float(ctx.knowledge.source_scores.get(item.source_id, 0.0)),
                item.round_index,
                item.source_id,
            ),
            reverse=True,
        )
        out: list[int] = []
        for source in fallback:
            canonical = source.canonical_url or source.url
            if not canonical or canonical in seen_canonical:
                continue
            seen_canonical.add(canonical)
            out.append(source.source_id)
        return out

    def _trim_to_limit(
        self,
        *,
        selected: list[int],
        source_by_id: dict[int, ResearchSource],
        round_index: int,
        limit: int,
    ) -> list[int]:
        out = list(selected)
        while len(out) > limit:
            removed = False
            for idx in range(len(out) - 1, -1, -1):
                source_id = out[idx]
                if source_by_id[source_id].round_index == round_index:
                    out.pop(idx)
                    removed = True
                    break
            if not removed:
                out.pop()
        return out


__all__ = ["ResearchOverviewStep"]
