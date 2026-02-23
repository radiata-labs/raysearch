from __future__ import annotations

import json
import re
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.models.errors import AppError
from serpsage.models.pipeline import ResearchSource, ResearchStepContext
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

_CITATION_RE = re.compile(
    r"\[\s*(?:citation|cite|ref|reference|[\u4e00-\u9fff]{1,4})\s*(?:[:\uFF1A])\s*([^\]]+?)\s*\]",
    re.IGNORECASE,
)


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
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
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
                now_utc=now_utc,
            )
            span.set_attr("mode", "structured")
            span.set_attr("target_language", target_language)
            span.set_attr("content_chars", int(len(ctx.output.content)))
            span.set_attr("has_structured", bool(ctx.output.structured is not None))
            return ctx

        await self._render_markdown(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
        )
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
        now_utc: datetime,
    ) -> None:
        model = resolve_research_model(
            ctx=ctx,
            stage="synthesize",
            fallback=self.settings.answer.generate.use_model,
        )
        messages = self._build_structured_messages(
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
            print(
                "[research.render.structured]",
                json.dumps(
                    {
                        "target_language": target_language,
                        "removed_citation_markers": int(removed),
                        "structured": ctx.output.structured,
                        "content": ctx.output.content,
                    },
                    ensure_ascii=False,
                ),
            )
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
                "[research.render.structured]",
                json.dumps(
                    {
                        "target_language": target_language,
                        "error": str(exc),
                        "structured": ctx.output.structured,
                        "content": ctx.output.content,
                    },
                    ensure_ascii=False,
                ),
            )

    async def _render_markdown(
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
        messages = self._build_markdown_messages(
            ctx,
            target_language=target_language,
            now_utc=now_utc,
        )
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
        denoised = self._reduce_citation_noise(rewritten)
        ctx.output.content = self._normalize_markdown(denoised)
        print(
            "[research.render.markdown]",
            json.dumps(
                {
                    "target_language": target_language,
                    "invalid_citation_indexes": invalid,
                    "content": ctx.output.content,
                },
                ensure_ascii=False,
            ),
        )

    def _build_structured_messages(
        self,
        ctx: ResearchStepContext,
        *,
        target_language: str,
        now_utc: datetime,
    ) -> list[dict[str, str]]:
        target_language_name = clean_whitespace(target_language) or "unspecified"
        selected_sources = self._select_sources_for_render(ctx, max_sources=12)
        abstract_packet = build_abstract_packet(
            sources=selected_sources,
            max_abstracts_per_source=1,
        )
        content_packet = self._build_render_content_packet(
            sources=selected_sources,
            max_chars=5200,
        )
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
                    "6) Temporal claims must be interpreted against CURRENT_UTC_DATE.\n"
                    "7) Keep URLs, IDs, numbers, and entity names unchanged where appropriate.\n"
                    "8) Evidence priority: SOURCE_CONTENT_PACKET > SOURCE_ABSTRACT_PACKET.\n"
                    "9) If SOURCE_CONTENT_PACKET is non-empty, decisive fields must be derived from it.\n"
                    "10) SOURCE_ABSTRACT_PACKET is supplementary only; never let it override conflicting content evidence.\n"
                    "11) Prefer richer, decision-useful detail over terse summaries.\n"
                    "12) For key decision fields, include scope, constraints, and confidence qualifiers when evidence supports them.\n"
                    "Allowed Evidence:\n"
                    "- Theme plan, round notes, source content packet, source abstract packet.\n"
                    "Failure Policy:\n"
                    "- If content evidence is insufficient for a field, use conservative neutral/tentative values.\n"
                    "- Do not invent facts to fill schema fields.\n"
                    "Quality Checklist:\n"
                    "- Internal consistency, factual precision, no overclaiming, content-first grounding.\n"
                    "- Fill schema fields with substantive content when evidence exists; avoid shallow placeholders."
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"THEME:\n{ctx.request.themes}",
                        f"CURRENT_UTC_TIMESTAMP:\n{now_utc.isoformat()}",
                        f"CURRENT_UTC_DATE:\n{now_utc.date().isoformat()}",
                        f"TARGET_OUTPUT_LANGUAGE:\n{target_language} ({target_language_name})",
                        "OUTPUT_DEPTH_POLICY:\n"
                        "- Prefer complete and informative field values over minimal wording.\n"
                        "- Include boundaries/assumptions where uncertainty matters.\n"
                        "- Preserve nuanced distinctions instead of flattening into generic phrases.\n",
                        "EVIDENCE_POLICY:\n"
                        "- SOURCE_CONTENT_PACKET is the primary evidence base.\n"
                        "- Use SOURCE_ABSTRACT_PACKET only as supplemental context.\n"
                        "- If content and abstract disagree, follow content and mark uncertainty if needed.\n",
                        f"THEME_PLAN:\n{ctx.plan.theme_plan}",
                        f"ROUND_NOTES:\n{self._build_round_notes(ctx)}",
                        f"SOURCE_CONTENT_PACKET:\n{content_packet}",
                        f"SOURCE_ABSTRACT_PACKET:\n{abstract_packet}",
                    ]
                ),
            },
        ]

    def _build_markdown_messages(
        self,
        ctx: ResearchStepContext,
        *,
        target_language: str,
        now_utc: datetime,
    ) -> list[dict[str, str]]:
        section_template = self._section_template()
        target_language_name = clean_whitespace(target_language) or "unspecified"
        selected_sources = self._select_sources_for_render(ctx, max_sources=12)
        abstract_packet = build_abstract_packet(
            sources=selected_sources,
            max_abstracts_per_source=1,
        )
        content_packet = self._build_render_content_packet(
            sources=selected_sources,
            max_chars=5200,
        )
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
                    "3) Citation token is language-invariant and must be exactly [citation:x] with ASCII colon.\n"
                    "4) Never translate citation token into local-language labels. Any non-[citation:x] label is invalid.\n"
                    "5) Use citations sparsely: cite only high-value factual claims, not every sentence.\n"
                    "6) Limit citation density: at most one citation cluster per sentence and avoid repeating the same marker in the same paragraph.\n"
                    "7) Prefer SOURCE_CONTENT_PACKET over SOURCE_ABSTRACT_PACKET for claim construction.\n"
                    "8) If content and abstract conflict, prefer content.\n"
                    "9) If SOURCE_CONTENT_PACKET is non-empty, major conclusions in sections 1/2/4 must be grounded in content evidence.\n"
                    "10) If only abstracts support a claim and full content is missing/weak, label the claim as tentative.\n"
                    "11) Distinguish direct evidence from inference.\n"
                    "12) Keep claims conservative when evidence is weak.\n"
                    "13) Resolve relative time expressions against CURRENT_UTC_DATE.\n"
                    "14) Write all natural-language text in TARGET_OUTPUT_LANGUAGE.\n"
                    "15) Return markdown only.\n"
                    "16) Ensure each section is substantively developed; avoid one-line placeholder sections.\n"
                    "17) For key claims, include context, rationale, and practical implication when evidence allows.\n"
                    "18) Prefer information density: synthesize mechanisms, trade-offs, and boundary conditions, not just outcomes.\n"
                    "Allowed Evidence:\n"
                    "- Theme plan, round notes, source content packet, source abstract packet.\n"
                    "Failure Policy:\n"
                    "- If uncertain, state uncertainty explicitly and avoid fabricated detail.\n"
                    "Quality Checklist:\n"
                    "- Traceability, coherence, balanced confidence, explicit gaps, low citation noise, strict citation token format."
                ),
            },
            {
                "role": "user",
                "content": "\n\n".join(
                    [
                        f"THEME:\n{ctx.request.themes}",
                        f"CURRENT_UTC_TIMESTAMP:\n{now_utc.isoformat()}",
                        f"CURRENT_UTC_DATE:\n{now_utc.date().isoformat()}",
                        f"TARGET_OUTPUT_LANGUAGE:\n{target_language} ({target_language_name})",
                        "OUTPUT_DEPTH_POLICY:\n"
                        "- Prefer detailed, information-dense writing over terse summaries.\n"
                        "- Expand key findings with mechanisms, constraints, and trade-offs.\n"
                        "- Include explicit uncertainty boundaries and realistic next investigation paths.\n",
                        "EVIDENCE_POLICY:\n"
                        "- Primary factual support must come from SOURCE_CONTENT_PACKET.\n"
                        "- SOURCE_ABSTRACT_PACKET can be used only for supplemental context.\n"
                        "- If content evidence exists for a topic, do not let abstract-only statements dominate that topic.\n",
                        f"THEME_PLAN:\n{ctx.plan.theme_plan}",
                        f"ROUND_NOTES:\n{self._build_round_notes(ctx)}",
                        f"SOURCE_CONTENT_PACKET:\n{content_packet}",
                        f"SOURCE_ABSTRACT_PACKET:\n{abstract_packet}",
                    ]
                ),
            },
        ]

    def _build_markdown_fallback(self, ctx: ResearchStepContext) -> str:
        top = self._select_sources_for_render(ctx, max_sources=6)
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
            tokens = [
                part.strip()
                for part in re.split(r"[,\s;|\u3001\uFF0C]+", raw)
                if part.strip()
            ]
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

    def _reduce_citation_noise(self, text: str) -> str:
        lines: list[str] = []
        for line in str(text or "").splitlines():
            seen: set[str] = set()
            kept = 0
            parts: list[str] = []
            cursor = 0
            for match in _CITATION_RE.finditer(line):
                parts.append(line[cursor : match.start()])
                token = clean_whitespace(match.group(1) or "")
                norm = token.casefold()
                if token and norm not in seen and kept < 1:
                    seen.add(norm)
                    kept += 1
                    parts.append(f"[citation:{token}]")
                cursor = match.end()
            parts.append(line[cursor:])
            merged = "".join(parts)
            merged = re.sub(r"[ \t]{2,}", " ", merged).rstrip()
            lines.append(merged)
        return "\n".join(lines)

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

    def _select_sources_for_render(
        self,
        ctx: ResearchStepContext,
        *,
        max_sources: int,
    ) -> list[ResearchSource]:
        with_content = [
            item for item in ctx.corpus.sources if clean_whitespace(str(item.content or ""))
        ]
        with_content.sort(
            key=lambda item: (
                int(item.round_index),
                len(clean_whitespace(str(item.content or ""))),
                int(item.source_id),
            ),
            reverse=True,
        )
        if with_content:
            return list(with_content[:max_sources])

        fallback = sorted(
            ctx.corpus.sources,
            key=lambda item: (int(item.round_index), int(item.source_id)),
            reverse=True,
        )
        return list(fallback[:max_sources])

    def _build_render_content_packet(
        self,
        *,
        sources: list[ResearchSource],
        max_chars: int,
    ) -> str:
        blocks: list[str] = []
        for source in sources:
            content = clean_whitespace(str(source.content or ""))
            if not content:
                continue
            if len(content) > max_chars:
                content = content[:max_chars]
            blocks.append(
                "\n".join(
                    [
                        f"[citation:{source.source_id}]",
                        f"url={source.url}",
                        f"title={clean_whitespace(source.title)}",
                        f"round_index={int(source.round_index)}",
                        "content:",
                        content,
                    ]
                )
            )
        return "\n\n".join(blocks) if blocks else "- (none)"


__all__ = ["ResearchRenderStep"]
