from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from pydantic import BaseModel, ConfigDict, Field

from serpsage.models.errors import AppError
from serpsage.models.pipeline import ResearchStepContext
from serpsage.steps.base import StepBase
from serpsage.steps.research.utils import (
    build_abstract_packet,
    chat_pydantic,
    merge_strings,
    normalize_strings,
    resolve_research_model,
)
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class _AbstractEvidenceGradePayload(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    source_id: int = Field(ge=1)
    grade: str
    reason: str


class _AbstractConflictPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    topic: str
    status: str
    source_ids: list[int] = Field(default_factory=list, max_length=8)
    decision: str


class _AbstractOutputPayload(BaseModel):
    model_config = ConfigDict(extra="forbid", validate_assignment=True)

    findings: list[str] = Field(default_factory=list, max_length=20)
    evidence_grades: list[_AbstractEvidenceGradePayload] = Field(
        default_factory=list,
        max_length=24,
    )
    conflict_arbitration: list[_AbstractConflictPayload] = Field(
        default_factory=list,
        max_length=16,
    )
    covered_subthemes: list[str] = Field(default_factory=list, max_length=16)
    coverage_delta: float = 0.0
    critical_gaps: list[str] = Field(default_factory=list, max_length=12)
    confidence: float = 0.0
    need_content_source_ids: list[int] = Field(default_factory=list, max_length=20)
    next_query_strategy: str = "coverage"
    next_queries: list[str] = Field(default_factory=list, max_length=8)
    stop: bool = False


class ResearchAbstractStep(StepBase[ResearchStepContext]):
    span_name = "step.research_abstract"

    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(
        self, ctx: ResearchStepContext, *, span: SpanBase
    ) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        if ctx.runtime.stop or ctx.current_round is None:
            return ctx

        sources = list(ctx.corpus.sources)
        if not sources:
            ctx.work.abstract_review = self._empty_review()
            ctx.work.need_content_source_ids = []
            print(
                "[research.abstract]",
                json.dumps(
                    {
                        "round_index": int(ctx.current_round.round_index),
                        "source_count": 0,
                        "abstract_review": ctx.work.abstract_review,
                        "need_content_source_ids": [],
                    },
                    ensure_ascii=False,
                ),
            )
            return ctx

        packet = build_abstract_packet(sources=sources, max_abstracts_per_source=5)
        model = resolve_research_model(
            ctx=ctx,
            stage="abstract",
            fallback=self.settings.answer.generate.use_model,
        )
        payload = self._empty_review()
        try:
            payload = await chat_pydantic(
                llm=self._llm,
                model=model,
                messages=self._build_abstract_messages(
                    ctx=ctx,
                    packet=packet,
                    now_utc=now_utc,
                ),
                schema_model=_AbstractOutputPayload,
                retries=int(self.settings.research.llm_self_heal_retries),
                schema_json=self._build_abstract_schema(
                    max_queries=int(ctx.runtime.budget.max_queries_per_round)
                ),
            )
        except Exception as exc:  # noqa: BLE001
            ctx.errors.append(
                AppError(
                    code="research_abstract_review_failed",
                    message=str(exc),
                    details={"round_index": int(ctx.current_round.round_index)},
                )
            )
            payload = self._empty_review()

        ctx.work.abstract_review = payload.model_dump()
        need_content_ids = self._normalize_source_ids(
            payload.need_content_source_ids,
            limit=20,
        )
        ctx.work.need_content_source_ids = need_content_ids

        findings = normalize_strings(payload.findings, limit=8)
        if findings:
            ctx.notes.extend(findings[:3])
            ctx.current_round.abstract_summary = " | ".join(findings[:3])
        ctx.current_round.confidence = self._normalize_confidence(payload.confidence)
        ctx.current_round.query_strategy = clean_whitespace(
            str(
                payload.next_query_strategy
                or ctx.current_round.query_strategy
                or "mixed"
            )
        )
        covered_subthemes = normalize_strings(payload.covered_subthemes, limit=16)
        if covered_subthemes:
            ctx.corpus.coverage_state.covered_subthemes = merge_strings(
                list(ctx.corpus.coverage_state.covered_subthemes),
                covered_subthemes,
                limit=64,
            )

        total = max(1, int(ctx.corpus.coverage_state.total_subthemes or 0))
        if total <= 0:
            total = max(1, len(ctx.corpus.coverage_state.covered_subthemes))
        coverage_ratio = min(
            1.0,
            float(len(ctx.corpus.coverage_state.covered_subthemes)) / float(total),
        )
        ctx.current_round.coverage_ratio = float(coverage_ratio)

        unresolved_topics = self._extract_unresolved_topics(payload.conflict_arbitration)
        ctx.current_round.unresolved_conflicts = int(len(unresolved_topics))
        ctx.current_round.critical_gaps = int(
            len(normalize_strings(payload.critical_gaps, limit=20))
        )
        ctx.work.next_queries = merge_strings(
            normalize_strings(
                payload.next_queries,
                limit=int(ctx.runtime.budget.max_queries_per_round),
            ),
            [],
            limit=int(ctx.runtime.budget.max_queries_per_round),
        )

        span.set_attr("round_index", int(ctx.current_round.round_index))
        span.set_attr("packet_chars", int(len(packet)))
        span.set_attr("need_content_ids", int(len(need_content_ids)))
        span.set_attr("confidence", float(ctx.current_round.confidence))
        span.set_attr("coverage_ratio", float(ctx.current_round.coverage_ratio))
        span.set_attr("unresolved_conflicts", int(ctx.current_round.unresolved_conflicts))
        print(
            "[research.abstract]",
            json.dumps(
                {
                    "round_index": int(ctx.current_round.round_index),
                    "source_count": int(len(sources)),
                    "packet_chars": int(len(packet)),
                    "need_content_source_ids": need_content_ids,
                    "confidence": float(ctx.current_round.confidence),
                    "coverage_ratio": float(ctx.current_round.coverage_ratio),
                    "unresolved_conflicts": int(ctx.current_round.unresolved_conflicts),
                    "critical_gaps": int(ctx.current_round.critical_gaps),
                    "next_queries": list(ctx.work.next_queries),
                    "abstract_review": payload.model_dump(),
                },
                ensure_ascii=False,
            ),
        )
        return ctx

    def _build_abstract_messages(
        self,
        *,
        ctx: ResearchStepContext,
        packet: str,
        now_utc: datetime,
    ) -> list[dict[str, str]]:
        out_lang = ctx.plan.output_language or "en"
        out_lang_name = clean_whitespace(out_lang) or "unspecified"
        core_question = clean_whitespace(ctx.plan.core_question or ctx.request.themes)
        round_index = ctx.current_round.round_index if ctx.current_round else 0
        return [
            {
                "role": "system",
                "content": (
                    "Role: Evidence Analyst (Abstract-First) and Methodology Instructor.\n"
                    "Mission: Evaluate evidence quality, theme coverage, and uncertainty using abstracts only.\n"
                    "Instruction Priority:\n"
                    "P1) Schema correctness.\n"
                    "P2) Evidence-grounded reasoning and conflict transparency.\n"
                    "P3) Language consistency.\n"
                    "Hard Constraints:\n"
                    "1) Use only SOURCE_ABSTRACT_PACKET.\n"
                    "2) Keep analysis scoped to CORE_QUESTION and its evidence dimensions.\n"
                    "3) Distinguish observations from inferences.\n"
                    "4) Identify unresolved conflicts and critical evidence gaps.\n"
                    "5) Select source IDs for full-content arbitration when claims are high-impact, comparative, contradictory, or recency-sensitive.\n"
                    "6) Evaluate temporal relevance for recency-sensitive claims.\n"
                    "7) Free-text fields must be in the required output language.\n"
                    "8) Return JSON only, exactly matching schema.\n"
                    "9) Abstract evidence is provisional: avoid final certainty when content verification is still needed.\n"
                    "Allowed Evidence:\n"
                    "- Theme, theme plan, round summaries, abstract packet.\n"
                    "Failure Policy:\n"
                    "- If evidence is weak or temporally stale for a recency query, lower confidence and propose targeted next queries.\n"
                    "Quality Checklist:\n"
                    "- Coverage progression, conflict clarity, economical content escalation, calibrated confidence."
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
                    "- If THEME or prior plans indicate latest/current intent, judge whether each abstract is fresh enough.\n"
                    "- Treat relative time words (today/this month/this year/recent/latest) against current_utc_date.\n"
                    "- For stale evidence under recency intent, request follow-up queries in next_queries.\n\n"
                    "LANGUAGE_POLICY:\n"
                    f"- required_output_language={out_lang} ({out_lang_name})\n"
                    "- Keep all free-text fields in the required output language.\n\n"
                    f"THEME_PLAN:\n{ctx.plan.theme_plan}\n\n"
                    f"PREVIOUS_ROUNDS:\n{[r.model_dump() for r in ctx.rounds[-3:]]}\n\n"
                    f"SOURCE_ABSTRACT_PACKET:\n{packet}\n\n"
                    "Grading rubric guidance:\n"
                    "- Grade A: direct, specific, and internally coherent evidence.\n"
                    "- Grade B: relevant but partial or weakly specific evidence.\n"
                    "- Grade C: low specificity, low trustworthiness, or high ambiguity.\n\n"
                    "Escalation rubric for need_content_source_ids:\n"
                    "- Include IDs for sources tied to key conclusions, major conflicts, or model-selection decisions.\n"
                    "- Include IDs when abstract wording is vague but potentially important.\n"
                    "- Include IDs when evidence freshness is uncertain for latest/current requests."
                ),
            },
        ]

    def _build_abstract_schema(self, *, max_queries: int) -> dict[str, Any]:
        return {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "findings",
                "evidence_grades",
                "conflict_arbitration",
                "covered_subthemes",
                "coverage_delta",
                "critical_gaps",
                "confidence",
                "need_content_source_ids",
                "next_query_strategy",
                "next_queries",
                "stop",
            ],
            "properties": {
                "findings": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {"type": "string"},
                },
                "evidence_grades": {
                    "type": "array",
                    "maxItems": 24,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["source_id", "grade", "reason"],
                        "properties": {
                            "source_id": {"type": "integer", "minimum": 1},
                            "grade": {"type": "string"},
                            "reason": {"type": "string"},
                        },
                    },
                },
                "conflict_arbitration": {
                    "type": "array",
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["topic", "status", "source_ids", "decision"],
                        "properties": {
                            "topic": {"type": "string"},
                            "status": {"type": "string"},
                            "source_ids": {
                                "type": "array",
                                "maxItems": 8,
                                "items": {"type": "integer", "minimum": 1},
                            },
                            "decision": {"type": "string"},
                        },
                    },
                },
                "covered_subthemes": {
                    "type": "array",
                    "maxItems": 16,
                    "items": {"type": "string"},
                },
                "coverage_delta": {"type": "number"},
                "critical_gaps": {
                    "type": "array",
                    "maxItems": 12,
                    "items": {"type": "string"},
                },
                "confidence": {"type": "number"},
                "need_content_source_ids": {
                    "type": "array",
                    "maxItems": 20,
                    "items": {"type": "integer", "minimum": 1},
                },
                "next_query_strategy": {"type": "string"},
                "next_queries": {
                    "type": "array",
                    "maxItems": max(1, int(max_queries)),
                    "items": {"type": "string"},
                },
                "stop": {"type": "boolean"},
            },
        }

    def _empty_review(self) -> _AbstractOutputPayload:
        return _AbstractOutputPayload()

    def _normalize_confidence(self, raw: object) -> float:
        try:
            if isinstance(raw, int | float | str):
                value = float(raw)
            else:
                return 0.0
        except Exception:  # noqa: S112
            return 0.0
        return min(1.0, max(0.0, value))

    def _extract_unresolved_topics(
        self,
        raw: list[_AbstractConflictPayload],
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
            if len(out) >= max(1, int(limit)):
                break
        return out


__all__ = ["ResearchAbstractStep"]
