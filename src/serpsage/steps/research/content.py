from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.models.pipeline import ResearchSource, ResearchStepContext
from serpsage.models.research import (
    ContentConflictPayload,
    ContentOutputPayload,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt_markdown import (
    render_abstract_review_markdown,
    render_theme_plan_markdown,
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

        cfg = ctx.settings.research.corpus

        source_ids = list(ctx.work.need_content_source_ids or [])

        if not source_ids:
            source_ids = list(ctx.current_round.context_source_ids or [])

        source_ids = sort_source_ids_by_score(
            ctx=ctx,
            source_ids=source_ids,
        )[: max(1, int(cfg.content_context_topk))]

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
            max_chars=9000,
        )

        model = resolve_research_model(
            ctx=ctx,
            stage="content",
            fallback=self.settings.answer.generate.use_model,
        )

        payload = self._empty_review()

        try:
            chat_result = await self._llm.chat(
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

        return ctx

    def _build_content_messages(
        self,
        *,
        ctx: ResearchStepContext,
        packet: str,
        now_utc: datetime,
    ) -> list[dict[str, str]]:

        out_lang = ctx.plan.output_language or "en"

        out_lang_name = clean_whitespace(out_lang) or "unspecified"

        core_question = clean_whitespace(ctx.plan.core_question or ctx.request.themes)

        round_index = ctx.current_round.round_index if ctx.current_round else "unknown"

        theme_plan_markdown = render_theme_plan_markdown(ctx.plan.theme_plan)

        abstract_review_markdown = render_abstract_review_markdown(
            ctx.work.abstract_review
        )

        return [
            {
                "role": "system",
                "content": (
                    "Role: Evidence Arbiter (Full-Content Stage).\n"
                    "Mission: Resolve contradictions and raise evidence completeness for ONE core question.\n"
                    "Instruction Priority:\n"
                    "P1) Schema correctness.\n"
                    "P2) Content-grounded arbitration quality.\n"
                    "P3) Language consistency.\n"
                    "Hard Constraints:\n"
                    "1) Use only SOURCE_CONTENT_PACKET.\n"
                    "2) Keep every judgment aligned to CORE_QUESTION; do not branch into a new standalone topic.\n"
                    "3) Mark conflict status conservatively as resolved, unresolved, or insufficient.\n"
                    "4) Prefer direct content evidence over abstract-level assumptions.\n"
                    "5) If uncertainty remains, list concrete remaining gaps.\n"
                    "6) next_queries must remain strictly focused on CORE_QUESTION and non-redundant.\n"
                    "7) For recency-sensitive claims, explicitly account for publication/update-time relevance.\n"
                    "8) Free-text fields must be in the required output language.\n"
                    "9) resolved_findings should be information-dense: include implication, condition, and edge-case when available.\n"
                    "10) required_entities coverage is mandatory when provided: output entity_coverage_complete, covered_entities, missing_entities.\n"
                    "11) Keep required entity strings exact, including version markers (for example qwen3.5, glm4.7).\n"
                    "12) If no valid focused next query exists, return next_queries as an empty array.\n"
                    "13) Return JSON only and match schema exactly.\n"
                    "Allowed Evidence:\n"
                    "- Theme, theme plan, abstract review, selected content packet.\n"
                    "Failure Policy:\n"
                    "- If evidence is insufficient or stale for recency intent, avoid overclaiming and lower confidence.\n"
                    "Quality Checklist:\n"
                    "- Clear arbitration, traceable rationale, realistic confidence adjustment, gap transparency.\n"
                    "- Preserve detail that can later be rendered as tables (fact, evidence, conflict, constraint, gap)."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"THEME:\n{ctx.request.themes}\n\n"
                    f"CORE_QUESTION:\n{core_question}\n\n"
                    f"ROUND_INDEX:\n{round_index}\n\n"
                    "TIME_CONTEXT:\n"
                    f"- current_utc_timestamp={now_utc.isoformat()}\n"
                    f"- current_utc_date={now_utc.date().isoformat()}\n\n"
                    "TEMPORAL_POLICY:\n"
                    "- Resolve relative time expressions against current_utc_date.\n"
                    "- If latest/current intent exists, prefer the most recent trustworthy evidence and flag stale content.\n"
                    "- Any next_queries must directly reduce uncertainty for CORE_QUESTION only.\n\n"
                    "LANGUAGE_POLICY:\n"
                    f"- required_output_language={out_lang} ({out_lang_name})\n"
                    "- Keep all free-text fields in the required output language.\n\n"
                    f"THEME_PLAN_MARKDOWN:\n{theme_plan_markdown}\n\n"
                    f"ABSTRACT_REVIEW_MARKDOWN:\n{abstract_review_markdown}\n\n"
                    "REQUIRED_ENTITIES:\n"
                    f"{ctx.plan.theme_plan.required_entities}\n\n"
                    f"SOURCE_CONTENT_PACKET:\n{packet}\n\n"
                    "Arbitration rubric:\n"
                    "- resolved: one side is sufficiently better supported by evidence.\n"
                    "- unresolved: both sides remain plausible with no decisive tie-break.\n"
                    "- insufficient: current evidence cannot adjudicate the claim.\n\n"
                    "Output depth rubric:\n"
                    "- Prefer specific, decision-useful findings over generic statements.\n"
                    "- Capture trade-offs and boundary conditions when relevant.\n"
                    "- Keep confidence_adjustment calibrated to evidence strength."
                ),
            },
        ]

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
