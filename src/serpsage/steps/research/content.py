from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

from serpsage.models.pipeline import ResearchSource, ResearchStepContext
from serpsage.models.research import (
    ContentConflictPayload,
    ContentOutputPayload,
    ReportStyle,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.context import (
    render_overview_review_markdown,
    render_theme_plan_markdown,
)
from serpsage.steps.research.prompt import (
    build_content_messages as build_content_prompt_messages,
)
from serpsage.steps.research.prompt import (
    resolve_report_style,
)
from serpsage.steps.research.search import (
    pick_sources_by_ids,
    sort_source_ids_by_score,
)
from serpsage.steps.research.utils import (
    merge_strings,
    normalize_entity_coverage,
    normalize_strings,
    resolve_research_model,
)
from serpsage.utils import clean_whitespace

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
        if ctx.runtime.stop or ctx.current_round is None:
            return ctx
        mode_depth = ctx.runtime.mode_depth
        content_topk = max(1, int(mode_depth.content_context_topk_override))
        packet_max_chars = max(1000, int(mode_depth.content_packet_max_chars))
        source_ids = list(ctx.work.need_content_source_ids or [])
        if not source_ids:
            source_ids = list(ctx.current_round.context_source_ids or [])
        source_ids = sort_source_ids_by_score(
            ctx=ctx,
            source_ids=source_ids,
        )[:content_topk]
        if not source_ids:
            ctx.work.content_review = self._empty_review()
            return ctx
        selected_sources = pick_sources_by_ids(
            sources=ctx.corpus.sources,
            source_ids=source_ids,
        )
        if not selected_sources:
            ctx.work.content_review = self._empty_review()
            return ctx
        packet = self._build_content_packet(
            sources=selected_sources,
            source_ids=source_ids,
            max_chars=packet_max_chars,
        )
        model = resolve_research_model(
            ctx=ctx,
            stage="content",
            fallback=self.settings.answer.generate.use_model,
        )
        payload = self._empty_review()
        try:
            chat_result = await self._llm.create(
                model=model,
                messages=self._build_content_messages(
                    ctx=ctx,
                    packet=packet,
                    now_utc=now_utc,
                ),
                response_format=ContentOutputPayload,
                format_override=self._build_content_schema(
                    max_queries=int(ctx.runtime.budget.max_queries_per_round)
                ),
                retries=int(self.settings.research.llm_self_heal_retries),
            )
            payload = chat_result.data
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="research.content.error",
                request_id=ctx.request_id,
                stage="content_review",
                status="error",
                error_code="research_content_review_failed",
                error_type=type(exc).__name__,
                attrs={
                    "round_index": int(ctx.current_round.round_index),
                    "message": str(exc),
                },
            )
            payload = self._empty_review()
        (
            entity_coverage_complete,
            covered_entities,
            missing_entities,
        ) = normalize_entity_coverage(
            covered_entities=payload.covered_entities,
            missing_entities=payload.missing_entities,
            entity_coverage_complete=payload.entity_coverage_complete,
            required_entities=ctx.plan.theme_plan.required_entities,
        )
        payload = payload.model_copy(
            update={
                "entity_coverage_complete": bool(entity_coverage_complete),
                "covered_entities": list(covered_entities),
                "missing_entities": list(missing_entities),
            }
        )
        ctx.work.content_review = payload
        findings = normalize_strings(payload.resolved_findings, limit=8)
        if findings:
            ctx.current_round.content_summary = " | ".join(findings[:3])
            ctx.notes.extend(findings[:3])
        adjustment = self._normalize_adjustment(payload.confidence_adjustment)
        ctx.current_round.confidence = min(
            1.0,
            max(0.0, float(ctx.current_round.confidence) + adjustment),
        )
        unresolved_count = self._count_unresolved(payload.conflict_resolutions)
        ctx.current_round.unresolved_conflicts = min(
            int(ctx.current_round.unresolved_conflicts),
            int(unresolved_count),
        )
        ctx.current_round.critical_gaps = int(
            len(normalize_strings(payload.remaining_gaps, limit=20))
        )
        ctx.current_round.entity_coverage_complete = bool(entity_coverage_complete)
        ctx.current_round.missing_entities = list(missing_entities)
        ctx.work.next_queries = merge_strings(
            list(ctx.work.next_queries),
            normalize_strings(
                payload.next_queries,
                limit=int(ctx.runtime.budget.max_queries_per_round),
            ),
            limit=int(ctx.runtime.budget.max_queries_per_round),
        )
        strategy = clean_whitespace(str(payload.next_query_strategy or ""))
        if strategy:
            ctx.current_round.query_strategy = strategy
        report_style = self._resolve_report_style(ctx)
        await self.emit_tracking_event(
            event_name="research.style.applied",
            request_id=ctx.request_id,
            stage="content_review",
            attrs={
                "report_style_selected": str(report_style),
                "style_applied_stage": "content",
                "mode_depth_profile": str(mode_depth.mode_key),
                "content_context_topk_effective": int(content_topk),
                "content_packet_max_chars_effective": int(packet_max_chars),
            },
        )
        return ctx

    def _build_content_messages(
        self,
        *,
        ctx: ResearchStepContext,
        packet: str,
        now_utc: datetime,
    ) -> list[dict[str, str]]:
        out_lang = self._resolve_output_language(ctx)
        out_lang_name = clean_whitespace(out_lang) or "unspecified"
        core_question = self._resolve_core_question(ctx)
        round_index = ctx.current_round.round_index if ctx.current_round else "unknown"
        report_style = self._resolve_report_style(ctx)
        theme_plan_markdown = render_theme_plan_markdown(ctx.plan.theme_plan)
        overview_review_markdown = render_overview_review_markdown(
            ctx.work.overview_review
        )
        return build_content_prompt_messages(
            theme=ctx.request.themes,
            core_question=core_question,
            report_style=report_style,
            mode_depth_profile=str(ctx.runtime.mode_depth.mode_key),
            round_index=str(round_index),
            current_utc_timestamp=now_utc.isoformat(),
            current_utc_date=now_utc.date().isoformat(),
            required_output_language=out_lang,
            required_output_language_label=out_lang_name,
            theme_plan_markdown=theme_plan_markdown,
            overview_review_markdown=overview_review_markdown,
            required_entities=list(ctx.plan.theme_plan.required_entities),
            source_content_packet=packet,
        )

    def _resolve_core_question(self, ctx: ResearchStepContext) -> str:
        question = clean_whitespace(
            ctx.plan.theme_plan.core_question or ctx.request.themes
        )
        return question or clean_whitespace(ctx.request.themes)

    def _resolve_output_language(self, ctx: ResearchStepContext) -> str:
        token = clean_whitespace(ctx.plan.theme_plan.output_language)
        return token or "en"

    def _resolve_report_style(self, ctx: ResearchStepContext) -> ReportStyle:
        cfg = self.settings.research.report_style
        fallback_style_key = clean_whitespace(str(cfg.fallback_style)).casefold()
        if fallback_style_key not in {"decision", "explainer", "execution"}:
            fallback_style_key = "explainer"
        return resolve_report_style(
            raw_style=ctx.plan.theme_plan.report_style,
            theme=self._resolve_core_question(ctx),
            enabled=bool(cfg.enabled),
            fallback_style=cast("ReportStyle", fallback_style_key),
            strict_style_lock=bool(cfg.strict_style_lock),
        )

    def _build_content_schema(self, *, max_queries: int) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "resolved_findings",
                "conflict_resolutions",
                "entity_coverage_complete",
                "covered_entities",
                "missing_entities",
                "remaining_gaps",
                "confidence_adjustment",
                "next_query_strategy",
                "next_queries",
                "stop",
            ],
            "properties": {
                "resolved_findings": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {"type": "string"},
                },
                "conflict_resolutions": {
                    "type": "array",
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["status"],
                        "properties": {
                            "status": {"type": "string"},
                        },
                    },
                },
                "entity_coverage_complete": {"type": "boolean"},
                "covered_entities": {
                    "type": "array",
                    "maxItems": 24,
                    "items": {"type": "string"},
                },
                "missing_entities": {
                    "type": "array",
                    "maxItems": 24,
                    "items": {"type": "string"},
                },
                "remaining_gaps": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {"type": "string"},
                },
                "confidence_adjustment": {"type": "number"},
                "next_query_strategy": {"type": "string"},
                "next_queries": {
                    "type": "array",
                    "maxItems": max(1, int(max_queries)),
                    "items": {"type": "string"},
                },
                "stop": {"type": "boolean"},
            },
        }

    def _empty_review(self) -> ContentOutputPayload:
        return ContentOutputPayload()

    def _normalize_adjustment(self, raw: object) -> float:
        try:
            value = float(raw)  # type: ignore[arg-type]
        except Exception:  # noqa: S112
            return 0.0
        return min(1.0, max(-1.0, value))

    def _count_unresolved(self, raw: list[ContentConflictPayload]) -> int:
        total = 0
        for item in raw:
            status = clean_whitespace(item.status).casefold()
            if status == "unresolved":
                total += 1
        return total

    def _build_content_packet(
        self,
        *,
        sources: list[ResearchSource],
        source_ids: list[int],
        max_chars: int,
    ) -> str:
        wanted = set(source_ids)
        blocks: list[str] = []
        for source in sorted(sources, key=lambda item: item.source_id):
            if source.source_id not in wanted:
                continue
            content = (
                str(source.content or "")
                .replace("\r\n", "\n")
                .replace("\r", "\n")
                .strip()
            )
            if len(content) > max_chars:
                content = content[:max_chars]
            content_lines = (content or "(empty)").split("\n")
            blocks.append(
                "\n".join(
                    [
                        f"### Source {int(source.source_id)}",
                        f"- URL: {source.url}",
                        f"- Title: {clean_whitespace(source.title)}",
                        "- Content:",
                        "  ```markdown",
                        *[f"  {line}" for line in content_lines],
                        "  ```",
                    ]
                )
            )
        return "\n\n".join(blocks)


__all__ = ["ResearchContentStep"]
