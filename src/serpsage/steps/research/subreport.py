from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.steps.research import (
    ResearchSource,
    ResearchStepContext,
    SubreportOutputPayload,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import build_subreport_prompt_messages
from serpsage.steps.research.search import (
    pick_sources_by_ids,
    select_context_source_ids,
    source_authority_score,
)
from serpsage.steps.research.utils import resolve_research_model

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime


class ResearchSubreportStep(StepBase[ResearchStepContext]):
    _CONTEXT_NEW_RESULT_TARGET_RATIO = 0.60
    _CONTEXT_MIN_HISTORY_SOURCES = 3

    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        await self._render_subreport(
            ctx=ctx,
            target_language=ctx.plan.theme_plan.output_language,
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
        try:
            result = await self._llm.create(
                model=model,
                messages=build_subreport_prompt_messages(
                    ctx=ctx,
                    target_language=target_language,
                    now_utc=now_utc,
                    source_evidence=self._select_sources_for_render(ctx),
                    source_evidence_max_chars=max(
                        1, ctx.runtime.mode_depth.source_chars
                    ),
                    notes=self._collect_recent_notes(ctx, limit=12),
                    require_insight_card=require_insight_card,
                ),
                response_format=SubreportOutputPayload,
                retries=self.settings.research.llm_self_heal_retries,
            )
        except Exception as exc:  # noqa: BLE001
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
            raise
        payload = result.data
        if require_insight_card and payload.track_insight_card is None:
            raise ValueError("subreport payload must include track_insight_card")
        ctx.output.structured = payload.track_insight_card
        ctx.output.content = payload.subreport_markdown
        await self.emit_tracking_event(
            event_name="research.style.applied",
            request_id=ctx.request_id,
            stage="subreport",
            attrs={
                "report_style_selected": ctx.plan.theme_plan.report_style,
                "require_insight_card": require_insight_card,
                "has_insight_card": payload.track_insight_card is not None,
            },
        )

    def _require_insight_card(self, ctx: ResearchStepContext) -> bool:
        return ctx.runtime.mode_depth.mode_key != "research-fast"

    def _collect_recent_notes(
        self,
        ctx: ResearchStepContext,
        *,
        limit: int,
    ) -> list[str]:
        seen: set[str] = set()
        notes: list[str] = []
        for item in reversed(ctx.notes):
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
        limit = max(1, ctx.runtime.mode_depth.source_topk)
        if ctx.rounds:
            latest_round_index = ctx.rounds[-1].round_index
        elif ctx.current_round is not None:
            latest_round_index = ctx.current_round.round_index
        else:
            latest_round_index = max(
                (item.round_index for item in ctx.corpus.sources), default=0
            )
        selected_ids = select_context_source_ids(
            ctx=ctx,
            round_index=latest_round_index,
            topk=limit,
            new_result_target_ratio=float(self._CONTEXT_NEW_RESULT_TARGET_RATIO),
            min_history_sources=self._CONTEXT_MIN_HISTORY_SOURCES,
        )
        if selected_ids:
            selected_sources = pick_sources_by_ids(
                sources=ctx.corpus.sources,
                source_ids=selected_ids,
            )
            if selected_sources:
                return sorted(
                    selected_sources,
                    key=lambda item: (
                        float(ctx.corpus.source_scores.get(item.source_id, 0.0)),
                        source_authority_score(item),
                        item.round_index,
                        item.source_id,
                    ),
                    reverse=True,
                )
        return []


__all__ = ["ResearchSubreportStep"]
