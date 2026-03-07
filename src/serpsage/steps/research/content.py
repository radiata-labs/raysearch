from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.steps.base import StepBase
from serpsage.steps.models import ResearchStepContext
from serpsage.steps.research.payloads import (
    ContentConflictPayload,
    ContentOutputPayload,
)
from serpsage.steps.research.prompt import (
    build_content_prompt_messages,
)
from serpsage.steps.research.schema import build_content_schema
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
        source_topk = max(1, mode_depth.source_topk)
        source_ids = list(ctx.work.need_content_source_ids or [])
        if not source_ids:
            source_ids = list(ctx.current_round.context_source_ids or [])
        source_ids = sort_source_ids_by_score(
            ctx=ctx,
            source_ids=source_ids,
        )[:source_topk]
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
        model = resolve_research_model(
            ctx=ctx,
            stage="content",
            fallback=self.settings.answer.generate.use_model,
        )
        payload = self._empty_review()
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
                    max_queries=ctx.runtime.budget.max_queries_per_round
                ),
                retries=self.settings.research.llm_self_heal_retries,
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
                    "round_index": ctx.current_round.round_index,
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
                "entity_coverage_complete": entity_coverage_complete,
                "covered_entities": list(covered_entities),
                "missing_entities": list(missing_entities),
            }
        )
        ctx.work.content_review = payload
        findings = normalize_strings(payload.resolved_findings, limit=8)
        if findings:
            ctx.current_round.content_summary = " | ".join(findings[:3])
            ctx.notes.extend(findings[:3])
        ctx.current_round.confidence = min(
            1.0,
            max(0.0, ctx.current_round.confidence + payload.confidence_adjustment),
        )
        unresolved_topics = self._merge_conflict_topics(
            baseline=ctx.current_round.unresolved_conflict_topics,
            raw=payload.conflict_resolutions,
        )
        ctx.current_round.unresolved_conflicts = len(unresolved_topics)
        ctx.current_round.unresolved_conflict_topics = list(unresolved_topics)
        ctx.current_round.critical_gaps = len(
            normalize_strings(payload.remaining_gaps, limit=20)
        )
        ctx.current_round.entity_coverage_complete = entity_coverage_complete
        ctx.current_round.missing_entities = list(missing_entities)
        ctx.work.next_queries = merge_strings(
            list(ctx.work.next_queries),
            normalize_strings(
                payload.next_queries,
                limit=ctx.runtime.budget.max_queries_per_round,
            ),
            limit=ctx.runtime.budget.max_queries_per_round,
        )
        strategy = clean_whitespace(payload.next_query_strategy)
        if strategy:
            ctx.current_round.query_strategy = strategy
        report_style = ctx.plan.theme_plan.report_style
        await self.emit_tracking_event(
            event_name="research.style.applied",
            request_id=ctx.request_id,
            stage="content_review",
            attrs={
                "report_style_selected": report_style,
                "style_applied_stage": "content",
                "mode_depth_profile": mode_depth.mode_key,
                "source_topk_effective": source_topk,
            },
        )
        return ctx

    def _empty_review(self) -> ContentOutputPayload:
        return ContentOutputPayload()

    def _merge_conflict_topics(
        self,
        *,
        baseline: list[str],
        raw: list[ContentConflictPayload],
    ) -> list[str]:
        active: dict[str, str] = {
            clean_whitespace(item).casefold(): clean_whitespace(item)
            for item in baseline
            if clean_whitespace(item)
        }
        for item in raw:
            topic = clean_whitespace(item.topic)
            if not topic:
                continue
            status = clean_whitespace(item.status).casefold()
            if status in {"resolved", "closed"}:
                active.pop(topic.casefold(), None)
                continue
            if status not in {"unresolved", "insufficient"}:
                continue
            active[topic.casefold()] = topic
        return list(active.values())


__all__ = ["ResearchContentStep"]
