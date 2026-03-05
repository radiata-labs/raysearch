from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import ResearchSource, ResearchStepContext
from serpsage.models.research import (
    OverviewConflictPayload,
    OverviewOutputPayload,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import (
    build_overview_messages as build_overview_prompt_messages,
)
from serpsage.steps.research.prompt import (
    build_overview_packet,
    render_rounds_markdown,
    render_theme_plan_markdown,
)
from serpsage.steps.research.schema import build_overview_schema
from serpsage.steps.research.search import (
    pick_sources_by_ids,
    select_context_source_ids,
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


class ResearchOverviewStep(StepBase[ResearchStepContext]):
    _CONTEXT_NEW_RESULT_TARGET_RATIO = 0.60
    _CONTEXT_MIN_HISTORY_SOURCES = 3

    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        if ctx.runtime.stop or ctx.current_round is None:
            return ctx
        all_sources = list(ctx.corpus.sources)
        if not all_sources:
            ctx.work.overview_review = self._empty_review()
            ctx.work.need_content_source_ids = []
            ctx.current_round.context_source_ids = []
            return ctx
        mode_depth = ctx.runtime.mode_depth
        overview_topk = max(1, mode_depth.overview_source_topk)
        overview_chars = max(1000, mode_depth.content_source_chars)
        (
            new_result_target_ratio,
            min_history_sources,
        ) = self._resolve_context_mix_targets(ctx=ctx, sources=all_sources)
        context_source_ids = select_context_source_ids(
            ctx=ctx,
            round_index=ctx.current_round.round_index,
            topk=overview_topk,
            new_result_target_ratio=new_result_target_ratio,
            min_history_sources=min_history_sources,
        )
        ctx.current_round.context_source_ids = list(context_source_ids)
        sources = pick_sources_by_ids(
            sources=all_sources,
            source_ids=context_source_ids,
        )
        if not sources:
            ctx.work.overview_review = self._empty_review()
            ctx.work.need_content_source_ids = []
            return ctx
        packet = build_overview_packet(
            sources=sources,
            max_overview_chars=overview_chars,
        )
        model = resolve_research_model(
            ctx=ctx,
            stage="overview",
            fallback=self.settings.answer.generate.use_model,
        )
        payload = self._empty_review()
        try:
            chat_result = await self._llm.create(
                model=model,
                messages=self._build_overview_messages(
                    ctx=ctx,
                    packet=packet,
                    now_utc=now_utc,
                ),
                response_format=OverviewOutputPayload,
                format_override=build_overview_schema(
                    max_queries=ctx.runtime.budget.max_queries_per_round
                ),
                retries=self.settings.research.llm_self_heal_retries,
            )
            payload = chat_result.data
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="research.overview.error",
                request_id=ctx.request_id,
                stage="overview_review",
                status="error",
                error_code="research_overview_review_failed",
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
        ctx.work.overview_review = payload
        need_content_ids = self._normalize_source_ids(
            payload.need_content_source_ids,
            limit=20,
        )
        ctx.work.need_content_source_ids = need_content_ids
        findings = normalize_strings(payload.findings, limit=8)
        if findings:
            ctx.notes.extend(findings[:3])
            ctx.current_round.overview_summary = " | ".join(findings[:3])
        ctx.current_round.confidence = min(1.0, max(-1.0, payload.confidence or 0.0))
        ctx.current_round.query_strategy = clean_whitespace(
            payload.next_query_strategy or ctx.current_round.query_strategy or "mixed"
        )
        covered_subthemes = normalize_strings(payload.covered_subthemes, limit=16)
        if covered_subthemes:
            ctx.corpus.coverage_state.covered_subthemes = merge_strings(
                list(ctx.corpus.coverage_state.covered_subthemes),
                covered_subthemes,
                limit=64,
            )
        total = max(1, ctx.corpus.coverage_state.total_subthemes or 0)
        if total <= 0:
            total = max(1, len(ctx.corpus.coverage_state.covered_subthemes))
        coverage_ratio = min(
            1.0,
            len(ctx.corpus.coverage_state.covered_subthemes) / total,
        )
        ctx.current_round.coverage_ratio = coverage_ratio
        ctx.current_round.entity_coverage_complete = entity_coverage_complete
        ctx.current_round.missing_entities = list(missing_entities)
        unresolved_topics = self._extract_unresolved_topics(
            payload.conflict_arbitration
        )
        ctx.current_round.unresolved_conflicts = len(unresolved_topics)
        ctx.current_round.critical_gaps = len(
            normalize_strings(payload.critical_gaps, limit=20)
        )
        ctx.work.next_queries = merge_strings(
            normalize_strings(
                payload.next_queries,
                limit=ctx.runtime.budget.max_queries_per_round,
            ),
            [],
            limit=ctx.runtime.budget.max_queries_per_round,
        )
        report_style = ctx.plan.theme_plan.report_style
        await self.emit_tracking_event(
            event_name="research.style.applied",
            request_id=ctx.request_id,
            stage="overview_review",
            attrs={
                "report_style_selected": report_style,
                "style_applied_stage": "overview",
                "mode_depth_profile": mode_depth.mode_key,
                "overview_context_topk_effective": overview_topk,
                "overview_source_chars_effective": overview_chars,
                "overview_new_result_target_ratio": new_result_target_ratio,
                "overview_min_history_sources": min_history_sources,
            },
        )
        return ctx

    def _build_overview_messages(
        self,
        *,
        ctx: ResearchStepContext,
        packet: str,
        now_utc: datetime,
    ) -> list[dict[str, str]]:
        out_lang = self._resolve_output_language(ctx)
        out_lang_name = out_lang or "unspecified"
        core_question = self._resolve_core_question(ctx)
        round_index = ctx.current_round.round_index if ctx.current_round else 0
        report_style = ctx.plan.theme_plan.report_style
        theme_plan_markdown = render_theme_plan_markdown(ctx.plan.theme_plan)
        previous_rounds_markdown = render_rounds_markdown(ctx.rounds, limit=3)
        return build_overview_prompt_messages(
            theme=ctx.request.themes,
            core_question=core_question,
            report_style=report_style,
            mode_depth_profile=ctx.runtime.mode_depth.mode_key,
            round_index=round_index,
            current_utc_timestamp=now_utc.isoformat(),
            current_utc_date=now_utc.date().isoformat(),
            required_output_language=out_lang,
            required_output_language_label=out_lang_name,
            theme_plan_markdown=theme_plan_markdown,
            previous_rounds_markdown=previous_rounds_markdown,
            required_entities=list(ctx.plan.theme_plan.required_entities),
            source_overview_packet=packet,
        )

    def _resolve_core_question(self, ctx: ResearchStepContext) -> str:
        return ctx.plan.theme_plan.core_question or ctx.request.themes

    def _resolve_output_language(self, ctx: ResearchStepContext) -> str:
        return ctx.plan.theme_plan.output_language or "en"

    def _empty_review(self) -> OverviewOutputPayload:
        return OverviewOutputPayload()

    def _extract_unresolved_topics(
        self,
        raw: list[OverviewConflictPayload],
    ) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in raw:
            status = clean_whitespace(item.status).casefold()
            if status != "unresolved":
                continue
            topic = clean_whitespace(item.topic)
            if not topic:
                continue
            key = topic.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(topic)
        return out

    def _normalize_source_ids(self, raw: list[int], *, limit: int) -> list[int]:
        out: list[int] = []
        seen: set[int] = set()
        for item in raw:
            try:
                value = int(item)
            except Exception:  # noqa: S112
                continue
            if value <= 0:
                continue
            if value in seen:
                continue
            seen.add(value)
            out.append(value)
            if len(out) >= max(1, limit):
                break
        return out

    def _resolve_context_mix_targets(
        self,
        *,
        ctx: ResearchStepContext,
        sources: list[ResearchSource],
    ) -> tuple[float, int]:
        mode_key = ctx.runtime.mode_depth.mode_key
        if mode_key == "research-fast":
            base_ratio = 0.70
            base_min_history = 1
        elif mode_key == "research-pro":
            base_ratio = 0.55
            base_min_history = 3
        else:
            base_ratio = self._CONTEXT_NEW_RESULT_TARGET_RATIO
            base_min_history = 2
        round_index = ctx.current_round.round_index if ctx.current_round else 0
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
