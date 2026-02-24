from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.models.errors import AppError
from serpsage.models.pipeline import ResearchStepContext, ResearchTrackResult
from serpsage.steps.base import StepBase
from serpsage.steps.research.utils import resolve_research_model
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class ResearchRenderStep(StepBase[ResearchStepContext]):
    span_name = "step.research_render_final"
    _MAX_SUBREPORT_EXCERPT_CHARS = 1600

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
        schema = (
            dict(ctx.request.json_schema)
            if isinstance(ctx.request.json_schema, dict)
            else None
        )
        if schema is not None:
            await self._render_final_structured(
                ctx=ctx,
                schema=schema,
                target_language=target_language,
                now_utc=now_utc,
            )
            span.set_attr("mode", "final_structured")
            span.set_attr("target_language", target_language)
            span.set_attr("content_chars", int(len(ctx.output.content)))
            span.set_attr("has_structured", bool(ctx.output.structured is not None))
            return ctx

        await self._render_final_markdown(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
        )
        span.set_attr("mode", "final_markdown")
        span.set_attr("target_language", target_language)
        span.set_attr("content_chars", int(len(ctx.output.content)))
        span.set_attr("has_structured", False)
        return ctx

    async def _render_final_markdown(
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
        messages = self._build_final_markdown_messages(
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
                    code="research_render_markdown_failed",
                    message=str(exc),
                    details={},
                )
            )
        if not raw_text.strip():
            raw_text = self._build_final_markdown_fallback(ctx)
        ctx.output.structured = None
        ctx.output.content = self._normalize_markdown(raw_text)
        print(
            "[research.render.final_markdown]",
            json.dumps(
                {
                    "target_language": target_language,
                    "track_count": int(len(ctx.parallel.track_results)),
                    "content": ctx.output.content,
                },
                ensure_ascii=False,
            ),
        )

    async def _render_final_structured(
        self,
        *,
        ctx: ResearchStepContext,
        schema: dict[str, Any],
        target_language: str,
        now_utc: datetime,
    ) -> None:
        model = resolve_research_model(
            ctx=ctx,
            stage="synthesize",
            fallback=self.settings.answer.generate.use_model,
        )
        messages = self._build_final_structured_messages(
            ctx,
            target_language=target_language,
            now_utc=now_utc,
        )
        try:
            result = await self._llm.chat(model=model, messages=messages, schema=schema)
            payload = (
                result.data
                if result.data is not None
                else self._try_parse_json_value(result.text)
            )
            if not isinstance(payload, dict):
                raise TypeError("structured output must be a JSON object")
            ctx.output.structured = payload
            ctx.output.content = json.dumps(payload, ensure_ascii=False, indent=2)
            print(
                "[research.render.final_structured]",
                json.dumps(
                    {
                        "target_language": target_language,
                        "track_count": int(len(ctx.parallel.track_results)),
                        "structured": payload,
                    },
                    ensure_ascii=False,
                ),
            )
            return
        except Exception as exc:  # noqa: BLE001
            ctx.errors.append(
                AppError(
                    code="research_render_structured_failed",
                    message=str(exc),
                    details={},
                )
            )
        fallback: dict[str, object] = {}
        ctx.output.structured = fallback
        ctx.output.content = json.dumps(fallback, ensure_ascii=False, indent=2)
        print(
            "[research.render.final_structured]",
            json.dumps(
                {
                    "target_language": target_language,
                    "track_count": int(len(ctx.parallel.track_results)),
                    "structured": fallback,
                },
                ensure_ascii=False,
            ),
        )

    def _build_final_markdown_messages(
        self,
        ctx: ResearchStepContext,
        *,
        target_language: str,
        now_utc: datetime,
    ) -> list[dict[str, str]]:
        target_language_name = clean_whitespace(target_language) or "unspecified"
        context_packet = self._build_final_context_packet(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
        )
        return [
            {
                "role": "system",
                "content": (
                    "Role: Theme-Level Research Synthesis Instructor.\n"
                    "Mission: Produce one final report for THEME from multi-track evidence.\n"
                    "Instruction Priority:\n"
                    "P1) Theme-level decision quality.\n"
                    "P2) Cross-track synthesis completeness.\n"
                    "P3) Language consistency and readability.\n"
                    "Step-by-step method:\n"
                    "1) Answer the core theme question directly in a concise opening.\n"
                    "2) Synthesize across tracks: consensus, conflicts, uncertainty boundaries, and implications.\n"
                    "3) Provide integrated conclusions and bounded recommendations.\n"
                    "4) Avoid per-track mini-report repetition.\n"
                    "Hard Constraints:\n"
                    "1) You are writing one theme-level final report, not per-track mini reports.\n"
                    "2) Use FINAL_CONTEXT_PACKET as the primary evidence context.\n"
                    "3) Keep report centered on THEME and FINAL_CONTEXT_PACKET.render_objective.\n"
                    "4) Do not include citation tokens.\n"
                    "5) Resolve relative time words against CURRENT_UTC_DATE.\n"
                    "6) Output markdown only.\n"
                    "7) No citation tokens. Markdown only.\n"
                    "Table Policy:\n"
                    "1) Table is optional, not required.\n"
                    "2) Use a table only when it improves clarity over prose.\n"
                    "3) Do not create decorative or placeholder tables.\n"
                    "4) Use a table only when comparison fields are stable and directly comparable.\n"
                    "5) If data is sparse or non-comparable, use prose/bullets instead of forcing a table.\n"
                    "Output Contract:\n"
                    "- Include: direct answer, integrated evidence synthesis, uncertainty boundaries, and implications.\n"
                    "- Keep high information density without rigid formatting.\n"
                    "Self-checklist:\n"
                    "- Did I synthesize across tracks instead of listing them one by one?\n"
                    "- Did I keep conclusions bounded by uncertainty?\n"
                    "- Did I avoid forcing table output?"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"TARGET_OUTPUT_LANGUAGE_LABEL:\n{target_language} ({target_language_name})\n\n"
                    f"CURRENT_UTC_DATE:\n{now_utc.date().isoformat()}\n\n"
                    f"FINAL_CONTEXT_PACKET_JSON:\n{context_packet}"
                ),
            },
        ]

    def _build_final_structured_messages(
        self,
        ctx: ResearchStepContext,
        *,
        target_language: str,
        now_utc: datetime,
    ) -> list[dict[str, str]]:
        target_language_name = clean_whitespace(target_language) or "unspecified"
        context_packet = self._build_final_context_packet(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
        )
        return [
            {
                "role": "system",
                "content": (
                    "Role: Structured Research Synthesizer.\n"
                    "Mission: Build one JSON object for THEME from FINAL_CONTEXT_PACKET.\n"
                    "Instruction Priority:\n"
                    "P1) Schema validity.\n"
                    "P2) Theme-aligned synthesis quality.\n"
                    "P3) Language consistency.\n"
                    "Step-by-step method:\n"
                    "1) Read FINAL_CONTEXT_PACKET and identify the core theme answer.\n"
                    "2) Merge cross-track evidence into schema-required fields.\n"
                    "3) Preserve uncertainty and unresolved conflicts explicitly.\n"
                    "Hard Constraints:\n"
                    "1) Output must strictly validate provided schema.\n"
                    "2) Use FINAL_CONTEXT_PACKET as the main evidence basis.\n"
                    "3) Do not include markdown or citation tokens.\n"
                    "4) Keep free-text values in TARGET_OUTPUT_LANGUAGE.\n"
                    "5) Preserve uncertainty when evidence is incomplete.\n"
                    "6) Resolve relative time words against CURRENT_UTC_DATE.\n"
                    "Self-checklist:\n"
                    "- Are all required schema fields present and valid?\n"
                    "- Are claims grounded in FINAL_CONTEXT_PACKET evidence?\n"
                    "- Are uncertainty boundaries preserved where needed?"
                ),
            },
            {
                "role": "user",
                "content": (
                    f"TARGET_OUTPUT_LANGUAGE_LABEL:\n{target_language} ({target_language_name})\n\n"
                    f"CURRENT_UTC_DATE:\n{now_utc.date().isoformat()}\n\n"
                    f"FINAL_CONTEXT_PACKET_JSON:\n{context_packet}"
                ),
            },
        ]

    def _build_final_markdown_fallback(self, ctx: ResearchStepContext) -> str:
        findings = self._collect_final_findings(ctx.parallel.track_results)
        if not findings:
            findings = ["No stable cross-track finding was extracted."]
        sections = [
            f"## Final Report: {ctx.request.themes}",
            "This report consolidates all sub-question tracks into one theme-focused synthesis.",
            "### Consolidated findings",
            "\n".join(f"- {item}" for item in findings[:8]),
            "### Track overview",
            "\n".join(
                [
                    "- Track-level comparison is summarized in prose to avoid forced table formatting.",
                    *[
                        (
                            f"- {item.question}: "
                            f"rounds={int(item.rounds)}, "
                            f"confidence={float(item.confidence):.2f}, "
                            f"coverage={float(item.coverage_ratio):.2f}, "
                            f"stop_reason={item.stop_reason or 'n/a'}"
                        )
                        for item in ctx.parallel.track_results
                    ],
                ]
            ),
        ]
        sections.extend(
            [
                "### Remaining risks and next actions",
                "- Unresolved conflicts can still affect final certainty; target disputed claims with dedicated verification queries.",
                "### Final synthesis",
                "Current evidence supports a coherent but bounded conclusion, and confidence should be interpreted with explicit uncertainty boundaries.",
            ]
        )
        return "\n\n".join(sections)

    def _build_final_context_packet(
        self,
        *,
        ctx: ResearchStepContext,
        target_language: str,
        now_utc: datetime,
    ) -> str:
        payload = {
            "theme": str(ctx.request.themes),
            "target_output_language": str(target_language),
            "time_context": {
                "utc_timestamp": now_utc.isoformat(),
                "utc_date": now_utc.date().isoformat(),
            },
            "theme_plan": dict(ctx.plan.theme_plan),
            "question_cards": [
                item.model_dump() for item in ctx.parallel.question_cards
            ],
            "track_results": self._build_track_result_packet(
                ctx.parallel.track_results
            ),
            "render_objective": (
                "Produce one theme-focused final synthesis with consensus, conflicts, "
                "uncertainty boundaries, and implications."
            ),
        }
        return json.dumps(payload, ensure_ascii=False, indent=2)

    def _resolve_target_language(self, ctx: ResearchStepContext) -> str:
        token = clean_whitespace(
            str(ctx.plan.output_language or ctx.plan.input_language or "")
        )
        if token:
            return token
        return "same as user input language"

    def _build_track_result_packet(
        self, track_results: list[ResearchTrackResult]
    ) -> list[dict[str, object]]:
        return [
            {
                "question_id": item.question_id,
                "question": item.question,
                "stop_reason": item.stop_reason,
                "rounds": int(item.rounds),
                "search_calls": int(item.search_calls),
                "fetch_calls": int(item.fetch_calls),
                "confidence": float(item.confidence),
                "coverage_ratio": float(item.coverage_ratio),
                "unresolved_conflicts": int(item.unresolved_conflicts),
                "key_findings": list(item.key_findings),
                "subreport_excerpt": self._normalize_block_text(
                    str(item.subreport_markdown or "")
                )[: self._MAX_SUBREPORT_EXCERPT_CHARS],
            }
            for item in track_results
        ]

    def _collect_final_findings(
        self, track_results: list[ResearchTrackResult]
    ) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in track_results:
            for finding in item.key_findings:
                text = clean_whitespace(finding)
                if not text:
                    continue
                key = text.casefold()
                if key in seen:
                    continue
                seen.add(key)
                out.append(text)
        return out

    def _try_parse_json_value(self, text: str) -> object:
        raw = str(text or "")
        if not raw:
            return {}
        try:
            return json.loads(raw)
        except json.JSONDecodeError:
            start = raw.find("{")
            end = raw.rfind("}")
            if 0 <= start < end:
                return json.loads(raw[start : end + 1])
            raise

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


__all__ = ["ResearchRenderStep"]
