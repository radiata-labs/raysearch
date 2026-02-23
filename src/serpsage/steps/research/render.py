from __future__ import annotations

import json
import re
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.models.errors import AppError
from serpsage.models.pipeline import ResearchStepContext
from serpsage.steps.base import StepBase
from serpsage.steps.research.utils import (
    build_abstract_packet,
    resolve_research_model,
)
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase

_CITATION_RE = re.compile(r"\[\s*citation\s*:\s*([^\]]+?)\s*\]", re.IGNORECASE)


class ResearchRenderStep(StepBase[ResearchStepContext]):
    span_name = "step.research_render"

    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(
        self, ctx: ResearchStepContext, *, span: SpanBase
    ) -> ResearchStepContext:
        target_language = self._resolve_target_language(ctx)
        schema = (
            dict(ctx.request.json_schema)
            if isinstance(ctx.request.json_schema, dict)
            else None
        )
        if schema is not None:
            await self._render_structured(
                ctx=ctx,
                schema=schema,
                target_language=target_language,
            )
            span.set_attr("mode", "structured")
            span.set_attr("target_language", target_language)
            span.set_attr("content_chars", int(len(ctx.output.content)))
            span.set_attr("has_structured", bool(ctx.output.structured is not None))
            return ctx

        await self._render_markdown(ctx=ctx, target_language=target_language)
        span.set_attr("mode", "markdown")
        span.set_attr("target_language", target_language)
        span.set_attr("content_chars", int(len(ctx.output.content)))
        span.set_attr("has_structured", False)
        return ctx

    async def _render_structured(
        self,
        *,
        ctx: ResearchStepContext,
        schema: dict[str, Any],
        target_language: str,
    ) -> None:
        model = resolve_research_model(
            ctx=ctx,
            stage="synthesize",
            fallback=self.settings.answer.generate.use_model,
        )
        messages = self._build_structured_messages(
            ctx, target_language=target_language
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
            cleaned, removed = self._strip_citation_markers(payload)
            if removed > 0:
                ctx.errors.append(
                    AppError(
                        code="research_structured_citation_removed",
                        message="structured output must not contain citation markers",
                        details={"removed_count": int(removed)},
                    )
                )
            ctx.output.structured = cleaned
            ctx.output.content = json.dumps(cleaned, ensure_ascii=False, indent=2)
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

    async def _render_markdown(
        self,
        *,
        ctx: ResearchStepContext,
        target_language: str,
    ) -> None:
        model = resolve_research_model(
            ctx=ctx,
            stage="markdown",
            fallback=self.settings.answer.generate.use_model,
        )
        messages = self._build_markdown_messages(ctx, target_language=target_language)
        raw_text = ""
        try:
            result = await self._llm.chat(model=model, messages=messages, schema=None)
            raw_text = str(result.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            ctx.errors.append(
                AppError(
                    code="research_render_markdown_failed",
                    message=str(exc),
                    details={},
                )
            )
        if not raw_text:
            raw_text = self._build_markdown_fallback(ctx)

        index_to_url = {
            int(source.source_id): str(source.url) for source in ctx.corpus.sources
        }
        rewritten, invalid = self._replace_numeric_citations_with_urls(
            raw_text,
            index_to_url=index_to_url,
        )
        for idx in invalid:
            ctx.errors.append(
                AppError(
                    code="research_invalid_citation",
                    message=f"invalid citation index: {idx}",
                    details={"index": int(idx)},
                )
            )
        ctx.output.structured = None
        ctx.output.content = self._normalize_markdown(rewritten)

    def _build_structured_messages(
        self,
        ctx: ResearchStepContext,
        *,
        target_language: str,
    ) -> list[dict[str, str]]:
        target_language_name = clean_whitespace(target_language) or "unspecified"
        return [
            {
                "role": "system",
                "content": (
                    "Role: Senior Research Synthesizer.\n"
                    "Mission: Convert multi-round evidence into a strict JSON object matching schema.\n"
                    "Instruction Priority:\n"
                    "P1) Schema validity.\n"
                    "P2) Factual precision.\n"
                    "P3) Target language consistency.\n"
                    "Hard Constraints:\n"
                    "1) Output must validate the provided schema.\n"
                    "2) Do not include citation markers.\n"
                    "3) Do not include markdown, comments, or additional keys.\n"
                    "4) Separate verified facts from uncertainty in textual fields.\n"
                    "5) Write all natural-language string values in TARGET_OUTPUT_LANGUAGE.\n"
                    "6) Keep URLs, IDs, numbers, and entity names unchanged where appropriate.\n"
                    "Allowed Evidence:\n"
                    "- Theme plan, round notes, source abstracts.\n"
                    "Failure Policy:\n"
                    "- If evidence is insufficient for a field, use conservative neutral values.\n"
                    "Quality Checklist:\n"
                    "- Internal consistency, factual precision, no overclaiming."
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"THEME:\n{ctx.request.themes}",
                        f"TARGET_OUTPUT_LANGUAGE:\n{target_language} ({target_language_name})",
                        f"THEME_PLAN:\n{ctx.plan.theme_plan}",
                        f"ROUND_NOTES:\n{self._build_round_notes(ctx)}",
                        f"SOURCES:\n{build_abstract_packet(sources=ctx.corpus.sources)}",
                    ]
                ),
            },
        ]

    def _build_markdown_messages(
        self,
        ctx: ResearchStepContext,
        *,
        target_language: str,
    ) -> list[dict[str, str]]:
        section_template = self._section_template()
        target_language_name = clean_whitespace(target_language) or "unspecified"
        return [
            {
                "role": "system",
                "content": (
                    "Role: Research Report Writer and Academic Style Instructor.\n"
                    "Mission: Produce a standardized markdown report grounded strictly in provided evidence.\n"
                    "Instruction Priority:\n"
                    "P1) Section-template compliance.\n"
                    "P2) Evidence traceability.\n"
                    "P3) Target language consistency.\n"
                    "Hard Constraints:\n"
                    "1) Output exactly six sections in this order with six numbered level-2 headings:\n"
                    f"{section_template}\n"
                    "2) Translate each heading title to TARGET_OUTPUT_LANGUAGE while keeping numbering and heading level.\n"
                    "3) Use [citation:x] markers for factual claims, x is source_id.\n"
                    "4) Distinguish direct evidence from inference.\n"
                    "5) Keep claims conservative when evidence is weak.\n"
                    "6) Write all natural-language text in TARGET_OUTPUT_LANGUAGE.\n"
                    "7) Return markdown only.\n"
                    "Allowed Evidence:\n"
                    "- Theme plan, round notes, source abstracts.\n"
                    "Failure Policy:\n"
                    "- If uncertain, state uncertainty explicitly and avoid fabricated detail.\n"
                    "Quality Checklist:\n"
                    "- Traceability, coherence, balanced confidence, explicit gaps."
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"THEME:\n{ctx.request.themes}",
                        f"TARGET_OUTPUT_LANGUAGE:\n{target_language} ({target_language_name})",
                        f"THEME_PLAN:\n{ctx.plan.theme_plan}",
                        f"ROUND_NOTES:\n{self._build_round_notes(ctx)}",
                        f"SOURCES:\n{build_abstract_packet(sources=ctx.corpus.sources)}",
                    ]
                ),
            },
        ]

    def _build_markdown_fallback(self, ctx: ResearchStepContext) -> str:
        top = list(ctx.corpus.sources[:6])
        evidence_lines = [f"- {s.title or s.url} [citation:{s.source_id}]" for s in top]
        if not evidence_lines:
            evidence_lines = ["- No valid source was collected."]
        summary = (
            ctx.notes[-1]
            if ctx.notes
            else f"Research loop completed for theme: {ctx.request.themes}."
        )
        return "\n\n".join(
            [
                "## 1) Core Conclusions",
                summary,
                "## 2) Key Findings",
                "- Findings were synthesized from multi-round search and evidence review.",
                "## 3) Evidence and Citations",
                "\n".join(evidence_lines),
                "## 4) Uncertainty and Conflicts",
                "- Remaining uncertainty is explicitly preserved when evidence is insufficient.",
                "## 5) Time Anchors",
                "- Findings are grounded in the pages fetched during this run.",
                "## 6) Next Research Questions",
                "- Which unresolved claims need authoritative primary-source verification?",
            ]
        )

    def _resolve_target_language(self, ctx: ResearchStepContext) -> str:
        token = clean_whitespace(str(ctx.plan.output_language or ctx.plan.input_language or ""))
        if token:
            return token
        return "same as user input language"

    def _section_template(self) -> str:
        return (
            "## 1) <Core Conclusions>\n"
            "## 2) <Key Findings>\n"
            "## 3) <Evidence and Citations>\n"
            "## 4) <Uncertainty and Conflicts>\n"
            "## 5) <Time Anchors>\n"
            "## 6) <Next Research Questions>"
        )

    def _build_round_notes(self, ctx: ResearchStepContext) -> str:
        if not ctx.rounds:
            return "- (none)"
        lines: list[str] = []
        for round_state in ctx.rounds[-8:]:
            line = (
                f"- round={round_state.round_index}; "
                f"strategy={round_state.query_strategy or 'n/a'}; "
                f"queries={len(round_state.queries)}; "
                f"results={round_state.result_count}; "
                f"new_sources={len(round_state.new_source_ids)}; "
                f"coverage={float(round_state.coverage_ratio):.2f}; "
                f"confidence={float(round_state.confidence):.2f}; "
                f"unresolved={int(round_state.unresolved_conflicts)}; "
                f"gaps={int(round_state.critical_gaps)}; "
                f"stop={str(bool(round_state.stop)).lower()}; "
                f"reason={round_state.stop_reason or 'n/a'}"
            )
            if round_state.abstract_summary:
                line = f"{line}; abstract={round_state.abstract_summary}"
            if round_state.content_summary:
                line = f"{line}; content={round_state.content_summary}"
            lines.append(line)
        return "\n".join(lines)

    def _replace_numeric_citations_with_urls(
        self,
        text: str,
        *,
        index_to_url: dict[int, str],
    ) -> tuple[str, list[int]]:
        invalid: list[int] = []

        def repl(match: re.Match[str]) -> str:
            raw = (match.group(1) or "").strip()
            if not raw:
                return ""
            tokens = [part.strip() for part in re.split(r"[,\s;|]+", raw) if part.strip()]
            replaced: list[str] = []
            for token in tokens:
                lower = token.casefold()
                if lower.startswith(("http://", "https://")):
                    replaced.append(f"[citation:{token}]")
                    continue
                try:
                    idx = int(token)
                except Exception:  # noqa: S112
                    continue
                url = index_to_url.get(idx)
                if not url:
                    invalid.append(idx)
                    continue
                replaced.append(f"[citation:{url}]")
            return " ".join(replaced)

        rewritten = _CITATION_RE.sub(repl, str(text or ""))
        return rewritten, sorted(set(invalid))

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

    def _strip_citation_markers(self, value: object) -> tuple[object, int]:
        removed = 0

        def walk(node: object) -> object:
            nonlocal removed
            if isinstance(node, str):
                count = len(_CITATION_RE.findall(node))
                if count:
                    removed += count
                return _CITATION_RE.sub("", node).strip()
            if isinstance(node, list):
                return [walk(item) for item in node]
            if isinstance(node, tuple):
                return tuple(walk(item) for item in node)
            if isinstance(node, dict):
                return {key: walk(item) for key, item in node.items()}
            return node

        return walk(value), removed

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


__all__ = ["ResearchRenderStep"]
