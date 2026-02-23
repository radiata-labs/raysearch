from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.models.errors import AppError
from serpsage.models.pipeline import ResearchSource, ResearchStepContext
from serpsage.steps.base import StepBase
from serpsage.steps.research.utils import (
    chat_json,
    merge_strings,
    normalize_strings,
    resolve_research_model,
)
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class ResearchContentStep(StepBase[ResearchStepContext]):
    span_name = "step.research_content"

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

        source_ids = list(ctx.work.need_content_source_ids or [])
        if not source_ids:
            ctx.work.content_review = self._empty_review()
            print(
                "[research.content]",
                json.dumps(
                    {
                        "round_index": int(ctx.current_round.round_index),
                        "selected_source_ids": [],
                        "content_review": ctx.work.content_review,
                    },
                    ensure_ascii=False,
                ),
            )
            span.set_attr("round_index", int(ctx.current_round.round_index))
            span.set_attr("content_source_ids", 0)
            return ctx

        packet = self._build_content_packet(
            sources=ctx.corpus.sources,
            source_ids=source_ids,
            max_chars=9000,
        )
        schema: dict[str, Any] = {
            "type": "object",
            "additionalProperties": False,
            "required": [
                "resolved_findings",
                "conflict_resolutions",
                "remaining_gaps",
                "confidence_adjustment",
                "next_query_strategy",
                "next_queries",
                "stop",
            ],
            "properties": {
                "resolved_findings": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 20,
                },
                "conflict_resolutions": {
                    "type": "array",
                    "maxItems": 16,
                    "items": {
                        "type": "object",
                        "additionalProperties": False,
                        "required": ["topic", "status", "source_ids", "reason"],
                        "properties": {
                            "topic": {"type": "string"},
                            "status": {
                                "type": "string",
                                "enum": ["resolved", "unresolved", "insufficient"],
                            },
                            "source_ids": {
                                "type": "array",
                                "items": {"type": "integer", "minimum": 1},
                                "maxItems": 8,
                            },
                            "reason": {"type": "string"},
                        },
                    },
                },
                "remaining_gaps": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": 12,
                },
                "confidence_adjustment": {
                    "type": "number",
                    "minimum": -1.0,
                    "maximum": 1.0,
                },
                "next_query_strategy": {
                    "type": "string",
                    "enum": ["coverage", "deepen", "verify", "refresh", "stop-ready"],
                },
                "next_queries": {
                    "type": "array",
                    "items": {"type": "string"},
                    "maxItems": int(ctx.runtime.budget.max_queries_per_round),
                },
                "stop": {"type": "boolean"},
            },
        }
        model = resolve_research_model(
            ctx=ctx,
            stage="content",
            fallback=self.settings.answer.generate.use_model,
        )
        payload: dict[str, Any]
        try:
            payload = await chat_json(
                llm=self._llm,
                model=model,
                messages=self._build_content_messages(
                    ctx=ctx,
                    packet=packet,
                    now_utc=now_utc,
                ),
                schema=schema,
                retries=int(self.settings.research.llm_self_heal_retries),
            )
        except Exception as exc:  # noqa: BLE001
            ctx.errors.append(
                AppError(
                    code="research_content_review_failed",
                    message=str(exc),
                    details={"round_index": int(ctx.current_round.round_index)},
                )
            )
            payload = self._empty_review()

        ctx.work.content_review = payload
        findings = normalize_strings(payload.get("resolved_findings"), limit=8)
        if findings:
            ctx.current_round.content_summary = " | ".join(findings[:3])
            ctx.notes.extend(findings[:3])
        adjustment = self._normalize_adjustment(payload.get("confidence_adjustment"))
        ctx.current_round.confidence = min(
            1.0,
            max(0.0, float(ctx.current_round.confidence) + adjustment),
        )
        unresolved_count = self._count_unresolved(payload.get("conflict_resolutions"))
        ctx.current_round.unresolved_conflicts = min(
            int(ctx.current_round.unresolved_conflicts),
            int(unresolved_count),
        )
        ctx.corpus.conflict_state.unresolved_count = int(unresolved_count)
        ctx.current_round.critical_gaps = int(
            len(normalize_strings(payload.get("remaining_gaps"), limit=20))
        )
        ctx.work.next_queries = merge_strings(
            list(ctx.work.next_queries),
            normalize_strings(
                payload.get("next_queries"),
                limit=int(ctx.runtime.budget.max_queries_per_round),
            ),
            limit=int(ctx.runtime.budget.max_queries_per_round),
        )
        strategy = clean_whitespace(str(payload.get("next_query_strategy") or ""))
        if strategy:
            ctx.current_round.query_strategy = strategy

        span.set_attr("round_index", int(ctx.current_round.round_index))
        span.set_attr("content_source_ids", int(len(source_ids)))
        span.set_attr("packet_chars", int(len(packet)))
        span.set_attr("confidence", float(ctx.current_round.confidence))
        span.set_attr("unresolved_conflicts", int(ctx.current_round.unresolved_conflicts))
        print(
            "[research.content]",
            json.dumps(
                {
                    "round_index": int(ctx.current_round.round_index),
                    "selected_source_ids": source_ids,
                    "packet_chars": int(len(packet)),
                    "confidence": float(ctx.current_round.confidence),
                    "unresolved_conflicts": int(ctx.current_round.unresolved_conflicts),
                    "critical_gaps": int(ctx.current_round.critical_gaps),
                    "next_queries": list(ctx.work.next_queries),
                    "content_review": payload,
                },
                ensure_ascii=False,
            ),
        )
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
        round_index = ctx.current_round.round_index if ctx.current_round else "unknown"
        return [
            {
                "role": "system",
                "content": (
                    "Role: Evidence Arbiter (Full-Content Stage) and Critical Reasoning Instructor.\n"
                    "Mission: Resolve contradictions and upgrade reliability using selected full-page content.\n"
                    "Instruction Priority:\n"
                    "P1) Schema correctness.\n"
                    "P2) Conflict arbitration quality.\n"
                    "P3) Language consistency.\n"
                    "Hard Constraints:\n"
                    "1) Use only SOURCE_CONTENT_PACKET.\n"
                    "2) Provide explicit conflict decisions with reasons.\n"
                    "3) Prefer content-backed conclusions over abstract-level assumptions.\n"
                    "4) If uncertainty remains, list concrete remaining gaps.\n"
                    "5) Keep next queries targeted and non-redundant.\n"
                    "6) For recency-sensitive claims, explicitly account for publication/update time relevance.\n"
                    "7) Free-text fields must be in the required output language.\n"
                    "8) resolved_findings should be information-dense: include decision implication and constraint when available.\n"
                    "9) Return JSON only and match schema exactly.\n"
                    "Allowed Evidence:\n"
                    "- Theme, theme plan, abstract review, selected content packet.\n"
                    "Failure Policy:\n"
                    "- If evidence is insufficient or stale for recency intent, avoid overclaiming and lower confidence.\n"
                    "Quality Checklist:\n"
                    "- Clear arbitration, traceable rationale, realistic confidence adjustment, gap transparency."
                ),
            },
            {
                "role": "user",
                "content": (
                    f"THEME:\n{ctx.request.themes}\n\n"
                    f"ROUND_INDEX:\n{round_index}\n\n"
                    "TIME_CONTEXT:\n"
                    f"- current_utc_timestamp={now_utc.isoformat()}\n"
                    f"- current_utc_date={now_utc.date().isoformat()}\n\n"
                    "TEMPORAL_POLICY:\n"
                    "- Resolve relative time expressions against current_utc_date.\n"
                    "- If latest/current intent exists, prefer the most recent trustworthy evidence and flag stale content.\n\n"
                    "LANGUAGE_POLICY:\n"
                    f"- required_output_language={out_lang} ({out_lang_name})\n"
                    "- Keep all free-text fields in the required output language.\n\n"
                    f"THEME_PLAN:\n{ctx.plan.theme_plan}\n\n"
                    f"ABSTRACT_REVIEW:\n{ctx.work.abstract_review}\n\n"
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

    def _empty_review(self) -> dict[str, object]:
        return {
            "resolved_findings": [],
            "conflict_resolutions": [],
            "remaining_gaps": [],
            "confidence_adjustment": 0.0,
            "next_query_strategy": "coverage",
            "next_queries": [],
            "stop": False,
        }

    def _normalize_adjustment(self, raw: object) -> float:
        try:
            value = float(raw)  # type: ignore[arg-type]
        except Exception:  # noqa: S112
            return 0.0
        return min(1.0, max(-1.0, value))

    def _count_unresolved(self, raw: object) -> int:
        if not isinstance(raw, list):
            return 0
        total = 0
        for item in raw:
            if not isinstance(item, dict):
                continue
            status = clean_whitespace(str(item.get("status") or "")).casefold()
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
            content = clean_whitespace(str(source.content or ""))
            if len(content) > max_chars:
                content = content[:max_chars]
            blocks.append(
                "\n".join(
                    [
                        f"[citation:{source.source_id}]",
                        f"url={source.url}",
                        f"title={clean_whitespace(source.title)}",
                        "content:",
                        content or "(empty)",
                    ]
                )
            )
        return "\n\n".join(blocks)


__all__ = ["ResearchContentStep"]
