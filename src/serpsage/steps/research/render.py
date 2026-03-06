from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any
from typing_extensions import override

import anyio

from serpsage.models.pipeline import (
    ResearchQuestionCard,
    ResearchStepContext,
    ResearchTrackResult,
)
from serpsage.models.research import (
    RenderArchitectOutput,
    RenderArchitectSectionPlan,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.language import (
    document_language_alignment,
)
from serpsage.steps.research.prompt import (
    build_final_language_repair_messages,
    build_render_architect_prompt_messages,
    build_render_structured_prompt_messages,
    build_render_writer_prompt_messages,
    normalize_block_text,
)
from serpsage.steps.research.utils import resolve_research_model
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime


class _WriterSectionError(RuntimeError):
    def __init__(
        self,
        *,
        section_id: str,
        subhead: str,
        index: int,
        cause: Exception,
    ) -> None:
        self.section_id = clean_whitespace(section_id)
        self.subhead = clean_whitespace(subhead)
        self.index = index
        self.cause_type = type(cause).__name__
        self.cause_message = clean_whitespace(str(cause))
        label = self.section_id or self.subhead or f"section-{self.index}"
        super().__init__(
            f"writer section failed: {label}; cause={self.cause_type}: {self.cause_message}"
        )


@dataclass(slots=True)
class _WriterSectionFailure:
    index: int
    section_id: str
    subhead: str
    cause_type: str
    cause_message: str

    def to_payload(self) -> dict[str, object]:
        return {
            "index": self.index,
            "section_id": self.section_id,
            "subhead": self.subhead,
            "cause_type": self.cause_type,
            "cause_message": self.cause_message,
        }


class ResearchRenderStep(StepBase[ResearchStepContext]):
    _FINAL_LANGUAGE_ALIGNMENT_MIN = 0.62
    _FINAL_LANGUAGE_REPAIR_MIN_CHARS = 1200
    _FINAL_LANGUAGE_REPAIR_MIN_IMPROVEMENT = 0.05

    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        target_language = ctx.plan.theme_plan.output_language
        report_style = ctx.plan.theme_plan.report_style
        schema = (
            dict(ctx.request.json_schema)
            if isinstance(ctx.request.json_schema, dict)
            else None
        )
        if schema is None:
            await self._render_markdown_architect_writer(
                ctx=ctx,
                target_language=target_language,
                now_utc=now_utc,
            )
            await self.emit_tracking_event(
                event_name="research.render.summary",
                request_id=ctx.request_id,
                stage="render",
                attrs={
                    "mode": "final_markdown",
                    "track_results": len(ctx.parallel.track_results),
                    "content_chars": len(ctx.output.content),
                    "mode_depth_profile": ctx.runtime.mode_depth.mode_key,
                    "report_style_selected": report_style,
                },
            )
            return ctx
        await self._render_structured_once(
            ctx=ctx,
            schema=schema,
            target_language=target_language,
            now_utc=now_utc,
        )
        await self.emit_tracking_event(
            event_name="research.render.summary",
            request_id=ctx.request_id,
            stage="render",
            attrs={
                "mode": "final_structured",
                "track_results": len(ctx.parallel.track_results),
                "has_structured": ctx.output.structured is not None,
                "mode_depth_profile": ctx.runtime.mode_depth.mode_key,
                "report_style_selected": report_style,
            },
        )
        return ctx

    async def _render_markdown_architect_writer(
        self,
        *,
        ctx: ResearchStepContext,
        target_language: str,
        now_utc: datetime,
    ) -> None:
        architect_output = await self._run_architect(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
        )
        architect_output = await self._validate_architect_question_coverage(
            ctx=ctx,
            architect_output=architect_output,
        )
        writer_outputs = await self._run_writers(
            ctx=ctx,
            architect_output=architect_output,
            target_language=target_language,
            now_utc=now_utc,
        )
        assembled = self._assemble_markdown_from_sections(
            ctx=ctx,
            architect_output=architect_output,
            writer_outputs=writer_outputs,
        )
        assembled = await self._repair_final_language_if_needed(
            ctx=ctx,
            markdown=assembled,
            target_language=target_language,
            now_utc=now_utc,
        )
        ctx.output.structured = None
        ctx.output.content = self._normalize_markdown(assembled)

    async def _run_architect(
        self,
        *,
        ctx: ResearchStepContext,
        target_language: str,
        now_utc: datetime,
    ) -> RenderArchitectOutput:
        model = resolve_research_model(
            ctx=ctx,
            stage="synthesize",
            fallback=self.settings.answer.generate.use_model,
        )
        try:
            chat_result = await self._llm.create(
                model=model,
                messages=build_render_architect_prompt_messages(
                    ctx=ctx,
                    target_language=target_language,
                    now_utc=now_utc,
                ),
                response_format=RenderArchitectOutput,
                retries=self.settings.research.llm_self_heal_retries,
            )
            return chat_result.data
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="research.render.error",
                request_id=ctx.request_id,
                stage="architect",
                status="error",
                error_code="research_render_architect_failed",
                error_type=type(exc).__name__,
                attrs={
                    "model": model,
                    "message": str(exc),
                },
            )
            raise RuntimeError("research architect render failed") from exc

    async def _validate_architect_question_coverage(
        self,
        *,
        ctx: ResearchStepContext,
        architect_output: RenderArchitectOutput,
    ) -> RenderArchitectOutput:
        required_question_ids = self._resolve_question_ids(ctx)
        if not required_question_ids:
            return architect_output
        sections = list(architect_output.sections or [])
        if not sections:
            return architect_output
        required_map: dict[str, str] = {}
        for item in required_question_ids:
            token = clean_whitespace(item)
            if not token:
                continue
            required_map[token.casefold()] = token
        normalized_sections: list[RenderArchitectSectionPlan] = []
        covered_question_ids: set[str] = set()
        body_indexes: list[int] = []
        for idx, section in enumerate(sections):
            normalized_question_ids: list[str] = []
            seen: set[str] = set()
            for raw in section.question_ids:
                token = clean_whitespace(raw)
                if not token:
                    continue
                canonical = required_map.get(token.casefold())
                if not canonical:
                    continue
                if canonical in seen:
                    continue
                seen.add(canonical)
                normalized_question_ids.append(canonical)
            if section.section_role == "body":
                body_indexes.append(idx)
                covered_question_ids.update(normalized_question_ids)
            normalized_sections.append(
                section.model_copy(update={"question_ids": normalized_question_ids})
            )
        missing_question_ids = [
            item for item in required_question_ids if item not in covered_question_ids
        ]
        if missing_question_ids:
            target_idx = (
                body_indexes[-1] if body_indexes else len(normalized_sections) - 1
            )
            target = normalized_sections[target_idx]
            patched_question_ids = self._merge_question_ids(
                list(target.question_ids),
                missing_question_ids,
            )
            normalized_sections[target_idx] = target.model_copy(
                update={"question_ids": patched_question_ids}
            )
            warning_message = (
                "architect output missed question_ids; patched into final body section "
                f"missing={missing_question_ids}"
            )
            ctx.notes.append(warning_message)
            await self.emit_tracking_event(
                event_name="research.render.coverage_patched",
                request_id=ctx.request_id,
                stage="architect",
                attrs={
                    "missing_question_ids": list(missing_question_ids),
                },
            )
        return architect_output.model_copy(update={"sections": normalized_sections})

    async def _run_writers(
        self,
        *,
        ctx: ResearchStepContext,
        architect_output: RenderArchitectOutput,
        target_language: str,
        now_utc: datetime,
    ) -> list[str]:
        model = resolve_research_model(
            ctx=ctx,
            stage="markdown",
            fallback=self.settings.answer.generate.use_model,
        )
        outputs = [""] * len(architect_output.sections)
        try:
            async with anyio.create_task_group() as tg:
                for index, section in enumerate(architect_output.sections):
                    tg.start_soon(
                        self._run_writer_for_section,
                        ctx,
                        model,
                        target_language,
                        now_utc,
                        architect_output,
                        section,
                        outputs,
                        index,
                    )
        except Exception as exc:  # noqa: BLE001
            failed_sections = self._collect_writer_section_failures(exc)
            failed_section_payload = [item.to_payload() for item in failed_sections]
            await self.emit_tracking_event(
                event_name="research.render.error",
                request_id=ctx.request_id,
                stage="writer",
                status="error",
                error_code="research_render_writer_failed",
                error_type=type(exc).__name__,
                attrs={
                    "model": model,
                    "sections_total": len(architect_output.sections),
                    "failed_sections": failed_section_payload,
                    "message": str(exc),
                },
            )
            raise RuntimeError("research writer render failed") from exc
        return outputs

    async def _run_writer_for_section(
        self,
        ctx: ResearchStepContext,
        model: str,
        target_language: str,
        now_utc: datetime,
        architect_output: RenderArchitectOutput,
        section: RenderArchitectSectionPlan,
        outputs: list[str],
        index: int,
    ) -> None:
        section_track_results = self._select_track_results_for_section(
            ctx=ctx,
            section=section,
        )
        messages = build_render_writer_prompt_messages(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
            architect_output=architect_output,
            section=section,
            track_results=section_track_results,
        )
        try:
            result = await self._llm.create(
                model=model,
                messages=messages,
                response_format=None,
            )
        except Exception as exc:  # noqa: BLE001
            raise _WriterSectionError(
                section_id=section.section_id,
                subhead=section.subhead,
                index=index,
                cause=exc if isinstance(exc, Exception) else Exception(str(exc)),
            ) from exc
        outputs[index] = result.text

    def _select_track_results_for_section(
        self,
        *,
        ctx: ResearchStepContext,
        section: RenderArchitectSectionPlan,
    ) -> list[ResearchTrackResult]:
        track_results = list(ctx.parallel.track_results)
        if not track_results:
            return []
        selected_question_ids: list[str] = []
        seen_question_ids: set[str] = set()
        for raw_question_id in section.question_ids:
            question_id = clean_whitespace(raw_question_id)
            if not question_id or question_id in seen_question_ids:
                continue
            seen_question_ids.add(question_id)
            selected_question_ids.append(question_id)
        if not selected_question_ids:
            return []
        track_result_map = {
            clean_whitespace(item.question_id): item for item in track_results
        }
        return [
            track_result_map[question_id].model_copy(deep=True)
            for question_id in selected_question_ids
            if question_id in track_result_map
        ]

    async def _render_structured_once(
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
        messages = build_render_structured_prompt_messages(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
        )
        try:
            result = await self._llm.create(
                model=model,
                messages=messages,
                response_format=schema,
            )
            payload = (
                result.data
                if result.data is not None
                else self._try_parse_json_value(result.text)
            )
            if not isinstance(payload, dict):
                raise TypeError("structured output must be a JSON object")
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="research.render.error",
                request_id=ctx.request_id,
                stage="structured",
                status="error",
                error_code="research_render_structured_failed",
                error_type=type(exc).__name__,
                attrs={
                    "model": model,
                    "schema_keys": sorted(schema),
                    "message": str(exc),
                },
            )
            raise RuntimeError("research structured render failed") from exc
        ctx.output.structured = payload
        ctx.output.content = json.dumps(payload, ensure_ascii=False, indent=2)

    def _assemble_markdown_from_sections(
        self,
        *,
        ctx: ResearchStepContext,
        architect_output: RenderArchitectOutput,
        writer_outputs: list[str],
    ) -> str:
        report_title = self._resolve_report_title(ctx)
        parts: list[str] = [f"# {report_title}"]
        objective = (architect_output.report_objective).strip()
        if objective:
            parts.append(objective)
        for section_plan, section_content in zip(
            architect_output.sections,
            writer_outputs,
            strict=False,
        ):
            subhead = section_plan.subhead or section_plan.section_id
            parts.append(f"## {subhead or 'Section'}")
            parts.append((section_content).strip())
        return "\n\n".join(parts).strip()

    def _resolve_report_title(self, ctx: ResearchStepContext) -> str:
        return (
            ctx.plan.theme_plan.core_question or ctx.request.themes or "Research Report"
        )

    async def _repair_final_language_if_needed(
        self,
        *,
        ctx: ResearchStepContext,
        markdown: str,
        target_language: str,
        now_utc: datetime,
    ) -> str:
        if target_language == "other":
            return markdown
        if len(normalize_block_text(markdown)) < self._FINAL_LANGUAGE_REPAIR_MIN_CHARS:
            return markdown
        alignment = document_language_alignment(
            text=markdown,
            target_language=target_language,
        )
        if alignment >= float(self._FINAL_LANGUAGE_ALIGNMENT_MIN):
            return markdown
        model = resolve_research_model(
            ctx=ctx,
            stage="markdown",
            fallback=self.settings.answer.generate.use_model,
        )
        try:
            repaired = await self._llm.create(
                model=model,
                messages=build_final_language_repair_messages(
                    target_language=target_language,
                    current_utc_date=now_utc.date().isoformat(),
                    markdown=markdown,
                ),
                retries=self.settings.research.llm_self_heal_retries,
            )
            candidate = normalize_block_text(repaired.text)
            if not candidate:
                return markdown
            candidate_alignment = document_language_alignment(
                text=candidate,
                target_language=target_language,
            )
            if (candidate_alignment - alignment) < float(
                self._FINAL_LANGUAGE_REPAIR_MIN_IMPROVEMENT
            ):
                return markdown
            await self.emit_tracking_event(
                event_name="research.render.language_repair_applied",
                request_id=ctx.request_id,
                stage="render",
                attrs={
                    "target_language": target_language,
                    "alignment_before": float(alignment),
                    "alignment_after": float(candidate_alignment),
                },
            )
            return candidate
        except Exception:
            return markdown

    def _resolve_question_ids(self, ctx: ResearchStepContext) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        cards = list(ctx.parallel.question_cards)
        if not cards:
            cards = [
                ResearchQuestionCard(
                    question_id=item.question_id,
                    question=item.question,
                    priority=item.priority,
                    seed_queries=list(item.seed_queries),
                    evidence_focus=list(item.evidence_focus),
                    expected_gain=item.expected_gain,
                )
                for item in ctx.plan.theme_plan.question_cards
            ]
        for card in cards:
            question_id = card.question_id
            if not question_id:
                continue
            if question_id in seen:
                continue
            seen.add(question_id)
            out.append(question_id)
        return out

    def _merge_question_ids(self, left: list[str], right: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in [*left, *right]:
            token = item
            if not token:
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    def _resolve_uncovered_question_ids(
        self,
        *,
        required_question_ids: list[str],
        body_sections: list[RenderArchitectSectionPlan],
    ) -> list[str]:
        if not required_question_ids:
            return []
        required: set[str] = set()
        for item in required_question_ids:
            token = clean_whitespace(item)
            if token:
                required.add(token)
        covered: set[str] = set()
        for section in body_sections:
            for raw_question_id in section.question_ids:
                token = clean_whitespace(raw_question_id)
                if not token:
                    continue
                if token not in required:
                    continue
                covered.add(token)
        return [item for item in required_question_ids if item not in covered]

    def _resolve_question_text_map(self, ctx: ResearchStepContext) -> dict[str, str]:
        out: dict[str, str] = {}
        cards = list(ctx.parallel.question_cards)
        if not cards:
            cards = [
                ResearchQuestionCard(
                    question_id=item.question_id,
                    question=item.question,
                    priority=item.priority,
                    seed_queries=list(item.seed_queries),
                    evidence_focus=list(item.evidence_focus),
                    expected_gain=item.expected_gain,
                )
                for item in ctx.plan.theme_plan.question_cards
            ]
        for card in cards:
            question_id = card.question_id
            question = card.question
            if not question_id or not question:
                continue
            out[question_id] = question
        return out

    def _merge_nonempty_strings(
        self,
        left: list[str],
        right: list[str],
        *,
        limit: int,
    ) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for raw in [*left, *right]:
            token = raw
            if not token:
                continue
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(token)
            if len(out) >= max(1, limit):
                break
        return out

    def _collect_writer_section_failures(
        self, exc: BaseException
    ) -> list[_WriterSectionFailure]:
        out: list[_WriterSectionFailure] = []
        stack: list[BaseException] = [exc]
        while stack and len(out) < 16:
            node = stack.pop()
            children = getattr(node, "exceptions", None)
            if isinstance(children, tuple):
                stack.extend(
                    child
                    for child in reversed(children)
                    if isinstance(child, BaseException)
                )
                continue
            if not isinstance(node, _WriterSectionError):
                continue
            out.append(
                _WriterSectionFailure(
                    index=node.index,
                    section_id=node.section_id,
                    subhead=node.subhead,
                    cause_type=node.cause_type,
                    cause_message=node.cause_message,
                )
            )
        return out

    def _try_parse_json_value(self, text: str) -> object:
        if not text:
            return {}
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            start = text.find("{")
            end = text.rfind("}")
            if 0 <= start < end:
                return json.loads(text[start : end + 1])
            raise

    def _normalize_markdown(self, text: str) -> str:
        content = text.replace("\r\n", "\n").replace("\r", "\n")
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
