from __future__ import annotations

import json
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.models.pipeline import ResearchStepContext
from serpsage.steps.base import StepBase
from serpsage.steps.research.utils import (
    add_error,
    build_abstract_packet,
    build_round_notes,
    normalize_markdown,
    replace_numeric_citations_with_urls,
    resolve_research_model,
    strip_citation_markers,
    try_parse_json_value,
)

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


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
        schema = (
            dict(ctx.request.json_schema)
            if isinstance(ctx.request.json_schema, dict)
            else None
        )
        if schema is not None:
            await self._render_structured(ctx=ctx, schema=schema)
            span.set_attr("mode", "structured")
            span.set_attr("content_chars", int(len(ctx.output.content)))
            span.set_attr("has_structured", bool(ctx.output.structured is not None))
            return ctx

        await self._render_markdown(ctx=ctx)
        span.set_attr("mode", "markdown")
        span.set_attr("content_chars", int(len(ctx.output.content)))
        span.set_attr("has_structured", False)
        return ctx

    async def _render_structured(
        self,
        *,
        ctx: ResearchStepContext,
        schema: dict[str, Any],
    ) -> None:
        model = resolve_research_model(
            ctx=ctx,
            stage="synthesize",
            fallback=self.settings.answer.generate.use_model,
        )
        messages = _build_structured_messages(ctx)
        try:
            result = await self._llm.chat(model=model, messages=messages, schema=schema)
            payload = result.data if result.data is not None else try_parse_json_value(result.text)
            if not isinstance(payload, dict):
                raise TypeError("structured output must be a JSON object")
            cleaned, removed = strip_citation_markers(payload)
            if removed > 0:
                add_error(
                    ctx,
                    code="research_structured_citation_removed",
                    message="structured output must not contain citation markers",
                    details={"removed_count": int(removed)},
                )
            ctx.output.structured = cleaned
            ctx.output.content = json.dumps(cleaned, ensure_ascii=False, indent=2)
        except Exception as exc:  # noqa: BLE001
            add_error(
                ctx,
                code="research_render_structured_failed",
                message=str(exc),
                details={},
            )
            fallback: dict[str, object] = {}
            ctx.output.structured = fallback
            ctx.output.content = json.dumps(fallback, ensure_ascii=False, indent=2)

    async def _render_markdown(self, *, ctx: ResearchStepContext) -> None:
        model = resolve_research_model(
            ctx=ctx,
            stage="markdown",
            fallback=self.settings.answer.generate.use_model,
        )
        messages = _build_markdown_messages(ctx)
        raw_text = ""
        try:
            result = await self._llm.chat(model=model, messages=messages, schema=None)
            raw_text = str(result.text or "").strip()
        except Exception as exc:  # noqa: BLE001
            add_error(
                ctx,
                code="research_render_markdown_failed",
                message=str(exc),
                details={},
            )
        if not raw_text:
            raw_text = _build_markdown_fallback(ctx)

        index_to_url = {
            int(source.source_id): str(source.url)
            for source in ctx.corpus.sources
        }
        rewritten, invalid = replace_numeric_citations_with_urls(
            raw_text,
            index_to_url=index_to_url,
        )
        for idx in invalid:
            add_error(
                ctx,
                code="research_invalid_citation",
                message=f"invalid citation index: {idx}",
                details={"index": int(idx)},
            )
        ctx.output.structured = None
        ctx.output.content = normalize_markdown(rewritten)


def _build_structured_messages(ctx: ResearchStepContext) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Role: Senior Research Synthesizer.\n"
                "Mission: Convert multi-round evidence into a strict JSON object matching schema.\n"
                "Hard Constraints:\n"
                "1) Output must validate the provided schema.\n"
                "2) Do not include citation markers.\n"
                "3) Do not include markdown, comments, or additional keys.\n"
                "4) Separate verified facts from uncertainty in textual fields.\n"
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
                    f"THEME_PLAN:\n{ctx.plan.theme_plan}",
                    f"ROUND_NOTES:\n{build_round_notes(ctx)}",
                    f"SOURCES:\n{build_abstract_packet(sources=ctx.corpus.sources)}",
                ]
            ),
        },
    ]


def _build_markdown_messages(ctx: ResearchStepContext) -> list[dict[str, str]]:
    return [
        {
            "role": "system",
            "content": (
                "Role: Research Report Writer.\n"
                "Mission: Produce a standardized markdown report grounded in provided evidence.\n"
                "Hard Constraints:\n"
                "1) Output exactly six sections in this order:\n"
                "## 1) Core Conclusions\n"
                "## 2) Key Findings\n"
                "## 3) Evidence and Citations\n"
                "## 4) Uncertainty and Conflicts\n"
                "## 5) Time Anchors\n"
                "## 6) Next Research Questions\n"
                "2) Use [citation:x] markers for factual claims, x is source_id.\n"
                "3) Distinguish direct evidence from inference.\n"
                "4) Keep claims conservative when evidence is weak.\n"
                "5) Return markdown only.\n"
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
                    f"THEME_PLAN:\n{ctx.plan.theme_plan}",
                    f"ROUND_NOTES:\n{build_round_notes(ctx)}",
                    f"SOURCES:\n{build_abstract_packet(sources=ctx.corpus.sources)}",
                ]
            ),
        },
    ]


def _build_markdown_fallback(ctx: ResearchStepContext) -> str:
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


__all__ = ["ResearchRenderStep"]

