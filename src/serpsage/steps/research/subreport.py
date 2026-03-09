from __future__ import annotations

from datetime import UTC, datetime
from typing_extensions import override

from serpsage.components.llm.base import LLMClientBase
from serpsage.components.rank.base import RankerBase
from serpsage.dependencies import Inject
from serpsage.models.steps.research import (
    OverviewOutputPayload,
    ResearchSource,
    ResearchStepContext,
    SubreportOutputPayload,
    SubreportUpdatePayload,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import (
    build_overview_prompt_messages,
    build_subreport_prompt_messages,
    build_subreport_update_prompt_messages,
)
from serpsage.steps.research.rank import rerank_research_sources
from serpsage.steps.research.schema import build_subreport_update_schema
from serpsage.steps.research.search import source_authority_score
from serpsage.steps.research.utils import resolve_research_model


class ResearchSubreportStep(StepBase[ResearchStepContext]):
    llm: LLMClientBase = Inject()
    ranker: RankerBase = Inject()

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
            ctx=ctx,
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
        except Exception as exc:  # noqa: BLE001
            await self._emit_subreport_error(ctx=ctx, model=model, exc=exc)
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
        await self.emit_tracking_event(
            event_name="research.style.applied",
            request_id=ctx.request_id,
            stage="subreport",
            attrs={
                "report_style_selected": ctx.task.style,
                "require_insight_card": require_insight_card,
                "has_insight_card": current_card is not None,
                "source_chunks": len(chunks),
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
        except Exception as exc:  # noqa: BLE001
            await self._emit_subreport_error(ctx=ctx, model=model, exc=exc)
            raise
        payload = result.data
        if (
            require_insight_card
            and payload.action == "update"
            and payload.updated_track_insight_card is None
        ):
            raise ValueError("subreport update must include track_insight_card")
        return payload

    async def _build_remaining_overview(
        self,
        *,
        ctx: ResearchStepContext,
        now_utc: datetime,
        remaining_sources: list[ResearchSource],
    ) -> OverviewOutputPayload:
        return (
            await self.llm.create(
                model=resolve_research_model(
                    ctx=ctx,
                    stage="overview",
                    fallback=self.settings.answer.generate.use_model,
                ),
                messages=build_overview_prompt_messages(
                    ctx=ctx,
                    sources=remaining_sources,
                    now_utc=now_utc,
                ),
                response_format=OverviewOutputPayload,
                retries=self.settings.research.llm_self_heal_retries,
            )
        ).data

    async def _emit_subreport_error(
        self,
        *,
        ctx: ResearchStepContext,
        model: str,
        exc: Exception,
    ) -> None:
        await self.emit_tracking_event(
            event_name="research.subreport.error",
            request_id=ctx.request_id,
            stage="subreport",
            status="error",
            error_code="research_render_subreport_failed",
            error_type=type(exc).__name__,
            attrs={
                "model": model,
                "message": str(exc),
            },
        )

    def _require_insight_card(self, ctx: ResearchStepContext) -> bool:
        return ctx.run.limits.mode_key != "research-fast"

    def _collect_recent_notes(
        self,
        ctx: ResearchStepContext,
        *,
        limit: int,
    ) -> list[str]:
        seen: set[str] = set()
        notes: list[str] = []
        for item in reversed(ctx.run.notes):
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
        selected_sources = [
            item.model_copy(deep=True)
            for item in list(ctx.knowledge.sources)
            if item.source_id not in set(ctx.knowledge.report_used_source_ids)
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
