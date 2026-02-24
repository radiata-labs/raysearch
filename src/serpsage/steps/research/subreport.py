from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.errors import AppError
from serpsage.models.pipeline import ResearchSource, ResearchStepContext
from serpsage.steps.base import StepBase
from serpsage.steps.research.utils import resolve_research_model
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class ResearchSubreportStep(StepBase[ResearchStepContext]):
    span_name = "step.research_render_subreport"
    _MAX_SOURCES_FOR_CONTEXT = 12
    _MAX_ABSTRACTS_PER_SOURCE = 3
    _MAX_CONTENT_EXCERPT_CHARS = 2200
    _MAX_TOTAL_CONTENT_CHARS = 22000

    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(
        self, ctx: ResearchStepContext, *, span: SpanBase
    ) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        target_language = self._resolve_target_language(ctx)
        await self._render_subreport(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
        )
        span.set_attr("mode", "subreport")
        span.set_attr("has_structured", False)
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
        messages = self._build_subreport_messages(
            ctx,
            target_language=target_language,
            now_utc=now_utc,
        )
        raw_text = ""
        try:
            result = await self._llm.chat(model=model, messages=messages, schema=None)
            raw_text = str(result.text or "")
        except Exception as exc:  # noqa: BLE001
            ctx.errors.append(
                AppError(
                    code="research_render_subreport_failed",
                    message=str(exc),
                    details={},
                )
            )
        if not raw_text.strip():
            raw_text = self._build_subreport_fallback(ctx)
        ctx.output.structured = None
        ctx.output.content = self._normalize_markdown(raw_text)

    def _build_subreport_messages(
        self,
        ctx: ResearchStepContext,
        *,
        target_language: str,
        now_utc: datetime,
    ) -> list[dict[str, str]]:
        target_language_name = clean_whitespace(target_language) or "unspecified"
        core_question = clean_whitespace(ctx.plan.core_question or ctx.request.themes)
        context_packet = self._build_subreport_context_packet(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
            core_question=core_question,
        )
        return [
            {
                "role": "system",
                "content": (
                    "Role: Single-Question Evidence Archive Instructor.\n"
                    "Mission: Produce one detailed subreport for CORE_QUESTION only.\n"
                    "You are writing an evidence archive for one core question.\n"
                    "Instruction Priority:\n"
                    "P1) Single-question focus.\n"
                    "P2) Evidence completeness and detail.\n"
                    "P3) Language consistency.\n"
                    "Step-by-step method:\n"
                    "1) Extract the direct answer status for CORE_QUESTION (confirmed, partial, or unresolved).\n"
                    "2) Organize evidence by source support, conflicts, and gaps.\n"
                    "3) Explain implications and constraints without drifting away from CORE_QUESTION.\n"
                    "4) Close with targeted next checks that reduce the largest uncertainty.\n"
                    "Hard Constraints:\n"
                    "1) Focus on CORE_QUESTION only.\n"
                    "2) Use SUBREPORT_CONTEXT_PACKET as the primary evidence context.\n"
                    "3) Prioritize evidence completeness and traceable detail over polished summarization.\n"
                    "4) Do not use citation tokens.\n"
                    "5) Resolve relative time words against CURRENT_UTC_DATE.\n"
                    "6) Keep free-text output in TARGET_OUTPUT_LANGUAGE.\n"
                    "7) Do not generate decorative/placeholder tables.\n"
                    "8) Do not force rigid templates when they reduce clarity.\n"
                    "Output Contract:\n"
                    "- Provide: evidence-backed findings, conflict status, remaining gaps, and targeted next checks.\n"
                    "- Keep narrative sections for causal explanation and constraints.\n"
                    "Table Policy:\n"
                    "1) Table is optional, not required.\n"
                    "2) Use a table only when it improves clarity over prose.\n"
                    "3) Use table only when it materially improves evidence mapping clarity.\n"
                    "4) For sparse or non-comparable evidence, prefer prose/bullets.\n"
                    "Final Output Format:\n"
                    "- Output markdown only.\n"
                    "No citation tokens. Markdown only.\n"
                    "Self-checklist:\n"
                    "- Did I stay strictly on CORE_QUESTION?\n"
                    "- Did I preserve evidence detail and uncertainty boundaries?\n"
                    "- Did I avoid forcing table output?"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"TARGET_OUTPUT_LANGUAGE_LABEL:\n{target_language} ({target_language_name})\n\n"
                    f"CURRENT_UTC_DATE:\n{now_utc.date().isoformat()}\n\n"
                    f"SUBREPORT_CONTEXT_PACKET_JSON:\n{context_packet}"
                ),
            },
        ]

    def _build_subreport_fallback(self, ctx: ResearchStepContext) -> str:
        core_question = clean_whitespace(ctx.plan.core_question or ctx.request.themes)
        sources = self._select_sources_for_render(ctx, max_sources=10)
        findings = [
            token for item in ctx.notes[-8:] if (token := clean_whitespace(item))
        ]
        if not findings:
            findings = ["Evidence was collected and reviewed for this core question."]
        round_state = ctx.rounds[-1] if ctx.rounds else None
        conflict_line = "- No decisive conflict captured yet; targeted verification is still needed."
        if round_state and round_state.unresolved_conflicts <= 0:
            conflict_line = "- Major conflict cluster appears resolved in the latest round evidence."
        gap_line = "- Remaining gaps still affect confidence bounds."
        if round_state and round_state.critical_gaps <= 0:
            gap_line = "- No major gap remains in the latest round summary."
        sections = [
            f"## Subreport: {core_question}",
            "This subreport keeps detail completeness first and summarizes evidence grounded in this track.",
            "### Key findings",
            "\n".join(f"- {item}" for item in findings[:8]),
        ]
        evidence_lines = [
            (
                f"- source_id={int(source.source_id)}; "
                f"title={clean_whitespace(source.title or 'n/a')}; "
                f"url={source.url}; "
                f"round={int(source.round_index)}"
            )
            for source in sources
        ]
        if not evidence_lines:
            evidence_lines = ["- No source snapshot is available."]
        sections.extend(["### Evidence snapshot", "\n".join(evidence_lines)])
        sections.extend(
            [
                "### Conflicts and gaps",
                f"{conflict_line}\n{gap_line}",
                "### Next focused checks",
                "- Continue with queries that directly reduce remaining uncertainty around the core question.",
            ]
        )
        return "\n\n".join(sections)

    def _build_subreport_context_packet(
        self,
        *,
        ctx: ResearchStepContext,
        target_language: str,
        now_utc: datetime,
        core_question: str,
    ) -> str:
        selected_sources = self._select_sources_for_render(
            ctx,
            max_sources=self._MAX_SOURCES_FOR_CONTEXT,
        )
        payload = {
            "theme": str(ctx.request.themes),
            "core_question": str(core_question),
            "target_output_language": str(target_language),
            "time_context": {
                "utc_timestamp": now_utc.isoformat(),
                "utc_date": now_utc.date().isoformat(),
            },
            "theme_plan": ctx.plan.theme_plan.model_dump(),
            "round_trajectory": self._build_round_trajectory_packet(ctx),
            "source_evidence": self._build_source_evidence_packet(selected_sources),
            "notes": self._collect_recent_notes(ctx, limit=12),
            "subreport_objective": (
                "Build an evidence archive for one core question with complete, "
                "traceable detail and explicit uncertainty boundaries."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _build_round_trajectory_packet(
        self,
        ctx: ResearchStepContext,
    ) -> list[dict[str, object]]:
        rounds = ctx.rounds[-8:]
        return [
            {
                "round_index": int(round_state.round_index),
                "query_strategy": clean_whitespace(round_state.query_strategy or "n/a"),
                "queries": [clean_whitespace(item) for item in round_state.queries[:8]],
                "result_count": int(round_state.result_count),
                "confidence": float(round_state.confidence),
                "coverage_ratio": float(round_state.coverage_ratio),
                "unresolved_conflicts": int(round_state.unresolved_conflicts),
                "critical_gaps": int(round_state.critical_gaps),
                "stop": bool(round_state.stop),
                "stop_reason": clean_whitespace(round_state.stop_reason or "n/a"),
            }
            for round_state in rounds
        ]

    def _build_source_evidence_packet(
        self,
        sources: list[ResearchSource],
    ) -> list[dict[str, object]]:
        out: list[dict[str, object]] = []
        total_chars = 0
        for source in sources:
            content_excerpt = self._normalize_block_text(str(source.content or ""))
            if content_excerpt:
                content_excerpt = content_excerpt[: self._MAX_CONTENT_EXCERPT_CHARS]
            projected = total_chars + len(content_excerpt)
            if projected > self._MAX_TOTAL_CONTENT_CHARS:
                break
            total_chars = projected
            out.append(
                {
                    "source_id": int(source.source_id),
                    "url": str(source.url),
                    "title": clean_whitespace(source.title or ""),
                    "round_index": int(source.round_index),
                    "is_subpage": bool(source.is_subpage),
                    "abstracts": [
                        token
                        for item in source.abstracts[: self._MAX_ABSTRACTS_PER_SOURCE]
                        if (token := clean_whitespace(item))
                    ],
                    "content_excerpt": content_excerpt,
                }
            )
        return out

    def _collect_recent_notes(
        self,
        ctx: ResearchStepContext,
        *,
        limit: int,
    ) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in reversed(ctx.notes):
            item = clean_whitespace(raw)
            if not item:
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= max(1, int(limit)):
                break
        out.reverse()
        return out

    def _resolve_target_language(self, ctx: ResearchStepContext) -> str:
        token = clean_whitespace(
            str(ctx.plan.output_language or ctx.plan.input_language or "")
        )
        if token:
            return token
        return "same as user input language"

    def _select_sources_for_render(
        self,
        ctx: ResearchStepContext,
        *,
        max_sources: int,
    ) -> list[ResearchSource]:
        with_content = [
            (item, self._normalize_block_text(str(item.content or "")))
            for item in ctx.corpus.sources
        ]
        with_content = [(item, content) for item, content in with_content if content]
        with_content.sort(
            key=lambda pair: (
                int(pair[0].round_index),
                len(pair[1]),
                int(pair[0].source_id),
            ),
            reverse=True,
        )
        if with_content:
            return [item for item, _ in with_content[:max_sources]]
        fallback = sorted(
            ctx.corpus.sources,
            key=lambda item: (int(item.round_index), int(item.source_id)),
            reverse=True,
        )
        return list(fallback[:max_sources])

    def _normalize_block_text(self, text: str) -> str:
        return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()

    def _normalize_markdown(self, text: str) -> str:
        content = str(text or "").replace("\r\n", "\n").replace("\r", "\n")
        lines: list[str] = []
        blank_count = 0
        for raw in content.split("\n"):
            line = raw.rstrip()
            if not line:
                blank_count += 1
                if blank_count > 2:
                    continue
                lines.append("")
                continue
            blank_count = 0
            lines.append(line)
        return "\n".join(lines).strip()


__all__ = ["ResearchSubreportStep"]
