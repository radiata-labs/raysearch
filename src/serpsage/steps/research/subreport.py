from __future__ import annotations

from datetime import UTC, datetime
from typing_extensions import override

from pydantic import ValidationError

from serpsage.components.llm.base import LLMClientBase
from serpsage.components.rank.base import RankerBase
from serpsage.dependencies import Depends
from serpsage.models.steps.research import (
    ResearchSource,
    ResearchStepContext,
)
from serpsage.models.steps.research.payloads import (
    OverviewReviewPayload,
    SubreportOutputPayload,
    SubreportUpdatePayload,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import (
    build_overview_prompt_messages,
    build_subreport_prompt_messages,
    build_subreport_update_prompt_messages,
)
from serpsage.steps.research.schema import (
    build_overview_schema,
    build_subreport_update_schema,
)
from serpsage.steps.research.utils import (
    rerank_research_sources,
    resolve_research_model,
    source_authority_score,
)


class ResearchSubreportStep(StepBase[ResearchStepContext]):
    llm: LLMClientBase = Depends()
    ranker: RankerBase = Depends()

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        await self._render_subreport(
            ctx=ctx,
            target_language=ctx.task.output_language,
            now_utc=now_utc,
        )
        return ctx

    async def _render_subreport(
        self,
        *,
        ctx: ResearchStepContext,
        target_language: str,
        now_utc: datetime,
    ) -> None:
        model = resolve_research_model(
            settings=self.settings,
            stage="markdown",
            fallback=self.settings.answer.generate.use_model,
        )
        require_insight_card = self._require_insight_card(ctx)
        notes = self._collect_recent_notes(ctx, limit=16)
        source_evidence = self._select_sources_for_render(ctx)
        if not source_evidence:
            ctx.result.structured = None
            ctx.result.content = ""
            return
        source_evidence = await rerank_research_sources(
            ctx=ctx,
            ranker=self.ranker,
            sources=source_evidence,
            query=ctx.task.question,
        )
        chunk_size = max(1, ctx.run.limits.report_source_batch_size)
        chunk_chars = max(1, ctx.run.limits.report_source_batch_chars)
        selected_source_count = len(source_evidence)
        chunks = [
            source_evidence[index : index + chunk_size]
            for index in range(0, len(source_evidence), chunk_size)
        ]
        try:
            initial = await self.llm.create(
                model=model,
                messages=build_subreport_prompt_messages(
                    ctx=ctx,
                    target_language=target_language,
                    now_utc=now_utc,
                    source_evidence=chunks[0],
                    source_evidence_max_chars=chunk_chars,
                    notes=notes,
                    require_insight_card=require_insight_card,
                ),
                response_format=SubreportOutputPayload,
                retries=self.settings.research.llm_self_heal_retries,
            )
            await self.meter.record(
                name="llm.tokens",
                request_id=ctx.request_id,
                model=str(model),
                unit="token",
                tokens={
                    "prompt_tokens": int(initial.usage.prompt_tokens),
                    "completion_tokens": int(initial.usage.completion_tokens),
                    "total_tokens": int(initial.usage.total_tokens),
                },
            )
        except Exception as exc:  # noqa: BLE001
            await self.tracker.error(
                name="research.subreport.failed",
                request_id=ctx.request_id,
                step="research.subreport",
                error_code="research_render_subreport_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "model": model,
                },
            )
            raise
        payload = initial.data
        if require_insight_card and payload.track_insight_card is None:
            raise ValueError("subreport payload must include track_insight_card")
        current_markdown = payload.subreport_markdown
        current_card = payload.track_insight_card
        used_count = len(chunks[0])
        for index, chunk in enumerate(chunks[1:], start=1):
            update = await self._apply_update_chunk(
                ctx=ctx,
                model=model,
                now_utc=now_utc,
                target_language=target_language,
                current_markdown=current_markdown,
                chunk=chunk,
                chunk_chars=chunk_chars,
                notes=notes,
                require_insight_card=require_insight_card,
                update_phase=f"update-{index}",
            )
            used_count += len(chunk)
            if update is None:
                continue
            if update.action == "update" and update.updated_subreport_markdown.strip():
                current_markdown = update.updated_subreport_markdown
                current_card = update.updated_track_insight_card or current_card
            if update.action == "stop_after_update" and index >= 1:
                break
        remaining_sources = source_evidence[used_count:]
        if remaining_sources:
            remaining_sources = await rerank_research_sources(
                ctx=ctx,
                ranker=self.ranker,
                sources=remaining_sources,
                query=ctx.task.question,
            )
            final_notes = list(notes)
            final_overview = await self._build_remaining_overview(
                ctx=ctx,
                now_utc=now_utc,
                remaining_sources=remaining_sources,
            )
            if final_overview.findings:
                final_notes.extend(final_overview.findings[:3])
            update = await self._apply_update_chunk(
                ctx=ctx,
                model=model,
                now_utc=now_utc,
                target_language=target_language,
                current_markdown=current_markdown,
                chunk=remaining_sources,
                chunk_chars=chunk_chars,
                notes=final_notes,
                require_insight_card=require_insight_card,
                update_phase="finalize",
            )
            if update is not None and update.updated_subreport_markdown.strip():
                current_markdown = update.updated_subreport_markdown
                current_card = update.updated_track_insight_card or current_card
        ctx.result.structured = current_card
        ctx.result.content = current_markdown
        self._mark_sources_used(ctx=ctx, sources=source_evidence)
        await self.tracker.info(
            name="research.style.applied",
            request_id=ctx.request_id,
            step="research.subreport",
            data={
                "success": True,
                "require_insight_card": require_insight_card,
                "has_insight_card": current_card is not None,
                "sources_selected": selected_source_count,
            },
        )
        await self.tracker.debug(
            name="research.style.applied.detail",
            request_id=ctx.request_id,
            step="research.subreport",
            data={
                "success": True,
                "report_style_selected": ctx.task.style,
                "source_chunks": len(chunks),
                "source_batch_size": chunk_size,
                "source_batch_chars": chunk_chars,
                "sources_selected": selected_source_count,
                "sources_used": len(source_evidence),
                "recent_notes": len(notes),
            },
        )

    async def _apply_update_chunk(
        self,
        *,
        ctx: ResearchStepContext,
        model: str,
        now_utc: datetime,
        target_language: str,
        current_markdown: str,
        chunk: list[ResearchSource],
        chunk_chars: int,
        notes: list[str],
        require_insight_card: bool,
        update_phase: str,
    ) -> SubreportUpdatePayload | None:
        if not chunk:
            return None
        try:
            result = await self.llm.create(
                model=model,
                messages=build_subreport_update_prompt_messages(
                    ctx=ctx,
                    target_language=target_language,
                    now_utc=now_utc,
                    current_report_markdown=current_markdown,
                    source_evidence=chunk,
                    source_evidence_max_chars=chunk_chars,
                    notes=notes,
                    require_insight_card=require_insight_card,
                    update_phase=update_phase,
                ),
                response_format=SubreportUpdatePayload,
                format_override=build_subreport_update_schema(
                    require_insight_card=require_insight_card
                ),
                retries=self.settings.research.llm_self_heal_retries,
            )
            await self.meter.record(
                name="llm.tokens",
                request_id=ctx.request_id,
                model=str(model),
                unit="token",
                tokens={
                    "prompt_tokens": int(result.usage.prompt_tokens),
                    "completion_tokens": int(result.usage.completion_tokens),
                    "total_tokens": int(result.usage.total_tokens),
                },
            )
        except Exception as exc:  # noqa: BLE001
            if self._is_nonfatal_update_parse_failure(exc):
                await self.tracker.warning(
                    name="research.subreport.update_skipped",
                    request_id=ctx.request_id,
                    step="research.subreport",
                    warning_code="research_subreport_update_parse_failed",
                    warning_message=(
                        "incremental subreport update chunk skipped after malformed "
                        "structured llm output"
                    ),
                    data={
                        "model": model,
                        "update_phase": update_phase,
                        "error_type": type(exc).__name__,
                        "error_message": str(exc),
                    },
                )
                return None
            await self.tracker.error(
                name="research.subreport.failed",
                request_id=ctx.request_id,
                step="research.subreport",
                error_code="research_render_subreport_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "model": model,
                },
            )
            raise
        payload = result.data
        if (
            require_insight_card
            and payload.action == "update"
            and payload.updated_track_insight_card is None
        ):
            raise ValueError("subreport update must include track_insight_card")
        return payload

    def _is_nonfatal_update_parse_failure(self, exc: Exception) -> bool:
        if isinstance(exc, ValidationError):
            return True
        message = str(exc).casefold()
        return any(
            token in message
            for token in (
                "invalid json",
                "json_invalid",
                "structured llm response",
                "must be a json object",
            )
        )

    async def _build_remaining_overview(
        self,
        *,
        ctx: ResearchStepContext,
        now_utc: datetime,
        remaining_sources: list[ResearchSource],
    ) -> OverviewReviewPayload:
        model = resolve_research_model(
            settings=self.settings,
            stage="overview",
            fallback=self.settings.answer.generate.use_model,
        )
        result = await self.llm.create(
            model=model,
            messages=build_overview_prompt_messages(
                ctx=ctx,
                sources=remaining_sources,
                now_utc=now_utc,
            ),
            response_format=OverviewReviewPayload,
            format_override=build_overview_schema(
                max_queries=ctx.run.limits.max_queries_per_round,
                select_engines=False,
            ),
            retries=self.settings.research.llm_self_heal_retries,
        )
        await self.meter.record(
            name="llm.tokens",
            request_id=ctx.request_id,
            model=str(model),
            unit="token",
            tokens={
                "prompt_tokens": int(result.usage.prompt_tokens),
                "completion_tokens": int(result.usage.completion_tokens),
                "total_tokens": int(result.usage.total_tokens),
            },
        )
        return result.data

    def _require_insight_card(self, ctx: ResearchStepContext) -> bool:
        return ctx.run.limits.mode_key != "research-fast"

    def _collect_recent_notes(
        self,
        ctx: ResearchStepContext,
        *,
        limit: int,
    ) -> list[str]:
        # Use track_state.notes for track-level execution notes
        notes_source = ctx.track_state.notes if ctx.track_state else ctx.run.notes
        seen: set[str] = set()
        notes: list[str] = []
        for item in reversed(notes_source):
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            notes.append(item)
            if len(notes) >= max(1, limit):
                break
        notes.reverse()
        return notes

    def _select_sources_for_render(
        self,
        ctx: ResearchStepContext,
    ) -> list[ResearchSource]:
        # loop.py passes a track-local knowledge snapshot for per-track rendering.
        used_source_ids = set(ctx.knowledge.report_used_source_ids)
        selected_sources = [
            item.model_copy(deep=True)
            for item in list(ctx.knowledge.sources)
            if item.source_id not in used_source_ids
        ]
        selected_sources.sort(
            key=lambda item: (
                float(ctx.knowledge.source_scores.get(item.source_id, 0.0)),
                source_authority_score(item),
                item.round_index,
                item.source_id,
            ),
            reverse=True,
        )
        return selected_sources[: max(1, ctx.run.limits.review_source_window)]

    def _mark_sources_used(
        self,
        *,
        ctx: ResearchStepContext,
        sources: list[ResearchSource],
    ) -> None:
        used_ids = list(ctx.knowledge.report_used_source_ids)
        used_set = set(used_ids)
        for source in sources:
            if source.source_id not in used_set:
                used_ids.append(source.source_id)
                used_set.add(source.source_id)
        ctx.knowledge.report_used_source_ids = used_ids
        for index, item in enumerate(ctx.knowledge.sources):
            if item.source_id not in used_set:
                continue
            ctx.knowledge.sources[index] = item.model_copy(
                update={"used_in_report": True},
                deep=True,
            )


__all__ = ["ResearchSubreportStep"]
