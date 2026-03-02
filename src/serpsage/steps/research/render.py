from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, Any, cast
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
    ReportStyle,
    ResearchThemePlan,
    TrackInsightCardPayload,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.context import (
    normalize_block_text,
    render_architect_plan_markdown,
    render_question_cards_markdown,
    render_section_plan_markdown,
    render_theme_plan_markdown,
)
from serpsage.steps.research.prompt import (
    build_density_gate_messages as build_density_gate_prompt_messages,
)
from serpsage.steps.research.prompt import (
    build_render_architect_messages as build_render_architect_prompt_messages,
)
from serpsage.steps.research.prompt import (
    build_render_structured_messages as build_render_structured_prompt_messages,
)
from serpsage.steps.research.prompt import (
    build_render_writer_messages as build_render_writer_prompt_messages,
)
from serpsage.steps.research.prompt import (
    resolve_report_style,
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
        self.index = int(index)
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
            "index": int(self.index),
            "section_id": str(self.section_id),
            "subhead": str(self.subhead),
            "cause_type": str(self.cause_type),
            "cause_message": str(self.cause_message),
        }


@dataclass(slots=True)
class _RenderTrackResultPacket:
    question_id: str
    question: str
    stop_reason: str
    rounds: int
    search_calls: int
    fetch_calls: int
    confidence: float
    coverage_ratio: float
    unresolved_conflicts: int
    track_insight_card: TrackInsightCardPayload | None = None
    key_findings: list[str] = field(default_factory=list)
    subreport_excerpt: str = ""


@dataclass(slots=True)
class _RenderFinalContextPacket:
    theme: str
    target_output_language: str
    mode_depth_profile: str
    utc_timestamp: str
    utc_date: str
    target_length_ratio: float
    theme_plan: ResearchThemePlan
    question_cards: list[ResearchQuestionCard] = field(default_factory=list)
    track_results: list[_RenderTrackResultPacket] = field(default_factory=list)
    render_objective: str = ""


class ResearchRenderStep(StepBase[ResearchStepContext]):
    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        target_language = self._resolve_target_language(ctx)
        report_style, style_applied = self._resolve_report_style(ctx)
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
                    "track_results": int(len(ctx.parallel.track_results)),
                    "content_chars": int(len(str(ctx.output.content or ""))),
                    "mode_depth_profile": str(ctx.runtime.mode_depth.mode_key),
                    "density_gate_passes_applied": int(
                        ctx.runtime.density_gate_passes_applied
                    ),
                    "target_output_chars": int(ctx.runtime.target_output_chars),
                    "output_length_ratio_vs_target": float(
                        ctx.runtime.output_length_ratio_vs_target
                    ),
                    "report_style_selected": str(report_style),
                    "style_applied_stage": "render" if style_applied else "none",
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
                "track_results": int(len(ctx.parallel.track_results)),
                "has_structured": bool(ctx.output.structured is not None),
                "mode_depth_profile": str(ctx.runtime.mode_depth.mode_key),
                "report_style_selected": str(report_style),
                "style_applied_stage": "render" if style_applied else "none",
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
        context_packet = self._build_final_context_packet(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
        )
        context_packet_markdown = self._render_final_context_packet_markdown(
            context_packet
        )
        architect_output = await self._run_architect(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
            context_packet_markdown=context_packet_markdown,
        )
        architect_output = await self._validate_architect_question_coverage(
            ctx=ctx,
            architect_output=architect_output,
        )
        architect_output = self._enforce_section_range(
            ctx=ctx,
            architect_output=architect_output,
        )
        writer_outputs = await self._run_writers(
            ctx=ctx,
            architect_output=architect_output,
            target_language=target_language,
            now_utc=now_utc,
            context_packet_markdown=context_packet_markdown,
        )
        assembled = self._assemble_markdown_from_sections(
            ctx=ctx,
            architect_output=architect_output,
            writer_outputs=writer_outputs,
        )
        target_chars = max(
            0,
            int(
                len(str(assembled))
                * float(ctx.runtime.mode_depth.target_length_ratio_vs_current)
            ),
        )
        if target_chars > 0 and int(ctx.runtime.target_output_chars) <= 0:
            ctx.runtime.target_output_chars = int(target_chars)
            ctx.runtime.output_length_ratio_vs_target = float(
                len(str(assembled)) / float(target_chars)
            )
        assembled = await self._apply_density_gate(
            ctx=ctx,
            markdown=assembled,
            target_language=target_language,
            now_utc=now_utc,
            context_packet_markdown=context_packet_markdown,
        )
        ctx.output.structured = None
        ctx.output.content = self._normalize_markdown(assembled)

    async def _run_architect(
        self,
        *,
        ctx: ResearchStepContext,
        target_language: str,
        now_utc: datetime,
        context_packet_markdown: str,
    ) -> RenderArchitectOutput:
        model = resolve_research_model(
            ctx=ctx,
            stage="synthesize",
            fallback=self.settings.answer.generate.use_model,
        )
        report_style, style_applied = self._resolve_report_style(ctx)
        try:
            chat_result = await self._llm.chat(
                model=model,
                messages=self._build_architect_messages(
                    ctx=ctx,
                    report_style=report_style,
                    style_applied=style_applied,
                    target_language=target_language,
                    now_utc=now_utc,
                    context_packet_markdown=context_packet_markdown,
                ),
                response_format=RenderArchitectOutput,
                retries=int(self.settings.research.llm_self_heal_retries),
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
                    "model": str(model),
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
        required_map = {
            clean_whitespace(item).casefold(): clean_whitespace(item)
            for item in required_question_ids
            if clean_whitespace(item)
        }
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
        context_packet_markdown: str,
    ) -> list[str]:
        model = resolve_research_model(
            ctx=ctx,
            stage="markdown",
            fallback=self.settings.answer.generate.use_model,
        )
        report_style, style_applied = self._resolve_report_style(ctx)
        outputs = [""] * len(architect_output.sections)
        try:
            async with anyio.create_task_group() as tg:
                for index, section in enumerate(architect_output.sections):
                    tg.start_soon(
                        self._run_writer_for_section,
                        model,
                        target_language,
                        now_utc,
                        context_packet_markdown,
                        architect_output,
                        section,
                        report_style,
                        style_applied,
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
                    "model": str(model),
                    "sections_total": int(len(architect_output.sections)),
                    "failed_sections": failed_section_payload,
                    "message": str(exc),
                },
            )
            raise RuntimeError("research writer render failed") from exc
        return outputs

    async def _run_writer_for_section(
        self,
        model: str,
        target_language: str,
        now_utc: datetime,
        context_packet_markdown: str,
        architect_output: RenderArchitectOutput,
        section: RenderArchitectSectionPlan,
        report_style: ReportStyle,
        style_applied: bool,
        outputs: list[str],
        index: int,
    ) -> None:
        messages = self._build_writer_messages(
            report_style=report_style,
            style_applied=style_applied,
            target_language=target_language,
            now_utc=now_utc,
            context_packet_markdown=context_packet_markdown,
            architect_output=architect_output,
            section=section,
        )
        try:
            result = await self._llm.chat(
                model=model,
                messages=messages,
                response_format=None,
            )
        except Exception as exc:  # noqa: BLE001
            raise _WriterSectionError(
                section_id=str(section.section_id or ""),
                subhead=str(section.subhead or ""),
                index=int(index),
                cause=exc if isinstance(exc, Exception) else Exception(str(exc)),
            ) from exc
        outputs[index] = str(result.text or "")

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
        messages = self._build_final_structured_messages(
            ctx,
            target_language=target_language,
            now_utc=now_utc,
        )
        try:
            result = await self._llm.chat(
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
                    "model": str(model),
                    "schema_keys": sorted(str(key) for key in schema),
                    "message": str(exc),
                },
            )
            raise RuntimeError("research structured render failed") from exc
        ctx.output.structured = payload
        ctx.output.content = json.dumps(payload, ensure_ascii=False, indent=2)

    def _build_architect_messages(
        self,
        *,
        ctx: ResearchStepContext,
        report_style: ReportStyle,
        style_applied: bool,
        target_language: str,
        now_utc: datetime,
        context_packet_markdown: str,
    ) -> list[dict[str, str]]:
        target_language_name = clean_whitespace(target_language) or "unspecified"
        section_min = max(1, int(ctx.runtime.mode_depth.render_section_min))
        section_max = max(section_min, int(ctx.runtime.mode_depth.render_section_max))
        return build_render_architect_prompt_messages(
            target_output_language=target_language,
            target_output_language_label=target_language_name,
            current_utc_date=now_utc.date().isoformat(),
            mode_depth_profile=str(ctx.runtime.mode_depth.mode_key),
            report_style=report_style,
            style_applied=bool(style_applied),
            section_min=int(section_min),
            section_max=int(section_max),
            context_packet_markdown=context_packet_markdown,
        )

    def _build_writer_messages(
        self,
        *,
        report_style: ReportStyle,
        style_applied: bool,
        target_language: str,
        now_utc: datetime,
        context_packet_markdown: str,
        architect_output: RenderArchitectOutput,
        section: RenderArchitectSectionPlan,
    ) -> list[dict[str, str]]:
        target_language_name = clean_whitespace(target_language) or "unspecified"
        section_subhead = clean_whitespace(section.subhead)
        section_prefix_h2 = f"## {section_subhead or 'Section'}"
        section_packet_markdown = render_section_plan_markdown(section)
        all_section_plan_markdown = render_architect_plan_markdown(architect_output)
        return build_render_writer_prompt_messages(
            target_output_language=target_language,
            target_output_language_label=target_language_name,
            current_utc_date=now_utc.date().isoformat(),
            report_style=report_style,
            style_applied=bool(style_applied),
            section_subhead=section_subhead,
            section_prefix_h2=section_prefix_h2,
            all_section_plan_markdown=all_section_plan_markdown,
            section_plan_markdown=section_packet_markdown,
            context_packet_markdown=context_packet_markdown,
        )

    def _build_final_structured_messages(
        self,
        ctx: ResearchStepContext,
        *,
        target_language: str,
        now_utc: datetime,
    ) -> list[dict[str, str]]:
        target_language_name = clean_whitespace(target_language) or "unspecified"
        report_style, style_applied = self._resolve_report_style(ctx)
        context_packet = self._build_final_context_packet(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
        )
        context_packet_markdown = self._render_final_context_packet_markdown(
            context_packet
        )
        return build_render_structured_prompt_messages(
            target_output_language=target_language,
            target_output_language_label=target_language_name,
            current_utc_date=now_utc.date().isoformat(),
            report_style=report_style,
            style_applied=bool(style_applied),
            context_packet_markdown=context_packet_markdown,
        )

    def _assemble_markdown_from_sections(
        self,
        *,
        ctx: ResearchStepContext,
        architect_output: RenderArchitectOutput,
        writer_outputs: list[str],
    ) -> str:
        parts: list[str] = [f"# {ctx.request.themes}"]
        objective = str(architect_output.report_objective or "").strip()
        if objective:
            parts.append(objective)
        for section_plan, section_content in zip(
            architect_output.sections,
            writer_outputs,
            strict=False,
        ):
            subhead = clean_whitespace(section_plan.subhead or section_plan.section_id)
            parts.append(f"## {subhead or 'Section'}")
            parts.append(str(section_content or "").strip())
        return "\n\n".join(parts).strip()

    def _build_final_context_packet(
        self,
        *,
        ctx: ResearchStepContext,
        target_language: str,
        now_utc: datetime,
    ) -> _RenderFinalContextPacket:
        mode_depth = ctx.runtime.mode_depth
        return _RenderFinalContextPacket(
            theme=str(ctx.request.themes),
            target_output_language=str(target_language),
            mode_depth_profile=str(mode_depth.mode_key),
            utc_timestamp=now_utc.isoformat(),
            utc_date=now_utc.date().isoformat(),
            target_length_ratio=float(mode_depth.target_length_ratio_vs_current),
            theme_plan=ctx.plan.theme_plan.model_copy(deep=True),
            question_cards=[
                item.model_copy(deep=True) for item in ctx.parallel.question_cards
            ],
            track_results=self._build_track_result_packet(ctx.parallel.track_results),
            render_objective=self._render_objective_for_mode(
                mode_key=str(mode_depth.mode_key)
            ),
        )

    def _render_final_context_packet_markdown(
        self, packet: _RenderFinalContextPacket
    ) -> str:
        lines: list[str] = [
            "# Final Context Packet",
            "## Theme",
            normalize_block_text(packet.theme) or "n/a",
            "## Target Output Language",
            normalize_block_text(packet.target_output_language) or "n/a",
            "## Mode Depth Profile",
            normalize_block_text(packet.mode_depth_profile) or "n/a",
            "## Time Context",
            f"- UTC timestamp: {packet.utc_timestamp}",
            f"- UTC date: {packet.utc_date}",
            "## Length Policy",
            f"- target_length_ratio_vs_current={float(packet.target_length_ratio):.2f}",
            "## Render Objective",
            packet.render_objective,
            "## Theme Plan",
            render_theme_plan_markdown(
                packet.theme_plan,
                include_title=False,
                include_question_cards=False,
            ),
            "## Private Rendering Rules",
            "- Internal metadata is private and must never appear in final user-facing report text.",
            "- Private fields include question IDs, track IDs, rounds, search/fetch call counts, stop reasons, section IDs, and coverage audit status.",
            "## Question Cards",
            render_question_cards_markdown(packet.question_cards),
            "## Track Results",
            self._render_track_results_markdown(packet.track_results),
        ]
        return "\n".join(lines).strip()

    def _render_objective_for_mode(self, *, mode_key: str) -> str:
        mode_name = clean_whitespace(mode_key).casefold()
        if mode_name == "research-fast":
            return (
                "Produce a concise synthesis that answers the theme directly with only "
                "the highest-impact findings."
            )
        if mode_name == "research-pro":
            return (
                "Produce a rich, high-density synthesis with explicit tradeoffs, "
                "boundary conditions, and action-ready implications."
            )
        return (
            "Produce a stable high-density synthesis with clear conclusions, conflicts, "
            "uncertainty boundaries, and actionable implications."
        )

    def _enforce_section_range(
        self,
        *,
        ctx: ResearchStepContext,
        architect_output: RenderArchitectOutput,
    ) -> RenderArchitectOutput:
        section_min = max(1, int(ctx.runtime.mode_depth.render_section_min))
        section_max = max(section_min, int(ctx.runtime.mode_depth.render_section_max))
        sections = list(architect_output.sections or [])
        if not sections:
            return architect_output
        opening = sections[0]
        closing = sections[-1]
        body_sections = list(sections[1:-1]) if len(sections) >= 2 else []
        if len(sections) > section_max:
            keep_body = max(0, section_max - 2)
            body_sections = body_sections[:keep_body]
            sections = [opening, *body_sections, closing]
        while len(sections) < section_min:
            idx = len(body_sections) + 1
            body_sections.append(
                RenderArchitectSectionPlan(
                    section_id=f"body_{idx}",
                    subhead=f"Additional Insight {idx}",
                    section_role="body",
                    question_ids=[],
                    scope_requirements=[
                        "Add one high-impact information layer not covered above."
                    ],
                    writing_boundaries=["Avoid repeating prior section content."],
                    must_cover_points=[
                        "One new conclusion with explicit condition and implication."
                    ],
                    angle="density_extension",
                    progression_hint="Expand with non-overlapping incremental value.",
                )
            )
            sections = [opening, *body_sections, closing]
        return architect_output.model_copy(update={"sections": sections})

    async def _apply_density_gate(
        self,
        *,
        ctx: ResearchStepContext,
        markdown: str,
        target_language: str,
        now_utc: datetime,
        context_packet_markdown: str,
    ) -> str:
        mode_depth = ctx.runtime.mode_depth
        if not bool(mode_depth.enable_density_gate):
            return markdown
        pass_cap = max(0, int(mode_depth.density_gate_passes))
        if pass_cap <= 0:
            return markdown
        model = resolve_research_model(
            ctx=ctx,
            stage="markdown",
            fallback=self.settings.answer.generate.use_model,
        )
        current = str(markdown or "")
        target_ratio = float(mode_depth.target_length_ratio_vs_current)
        target_chars = max(0, int(len(current) * target_ratio))
        if target_chars > 0:
            ctx.runtime.target_output_chars = int(target_chars)
        for pass_index in range(pass_cap):
            try:
                result = await self._llm.chat(
                    model=model,
                    messages=self._build_density_gate_messages(
                        ctx=ctx,
                        markdown=current,
                        target_language=target_language,
                        now_utc=now_utc,
                        context_packet_markdown=context_packet_markdown,
                        pass_index=pass_index,
                        target_chars=target_chars,
                    ),
                    response_format=None,
                )
                candidate = self._normalize_markdown(str(result.text or ""))
            except Exception as exc:  # noqa: BLE001
                await self.emit_tracking_event(
                    event_name="research.density_gate.error",
                    request_id=ctx.request_id,
                    stage="render",
                    status="error",
                    error_code="research_density_gate_failed",
                    error_type=type(exc).__name__,
                    attrs={
                        "pass_index": int(pass_index + 1),
                        "model": str(model),
                        "message": str(exc),
                    },
                )
                break
            if not candidate:
                continue
            if target_chars > 0 and len(candidate) < int(target_chars * 0.75):
                continue
            current = candidate
            ctx.runtime.density_gate_passes_applied += 1
        if target_chars > 0:
            ctx.runtime.output_length_ratio_vs_target = float(
                len(current) / float(target_chars)
            )
        return current

    def _build_density_gate_messages(
        self,
        *,
        ctx: ResearchStepContext,
        markdown: str,
        target_language: str,
        now_utc: datetime,
        context_packet_markdown: str,
        pass_index: int,
        target_chars: int,
    ) -> list[dict[str, str]]:
        return build_density_gate_prompt_messages(
            target_output_language=target_language,
            current_utc_date=now_utc.date().isoformat(),
            mode_depth_profile=str(ctx.runtime.mode_depth.mode_key),
            pass_index=int(pass_index),
            target_chars=int(target_chars),
            context_packet_markdown=context_packet_markdown,
            current_markdown=markdown,
        )

    def _resolve_target_language(self, ctx: ResearchStepContext) -> str:
        token = clean_whitespace(
            str(ctx.plan.output_language or ctx.plan.input_language or "")
        )
        if token:
            return token
        return "same as user input language"

    def _resolve_report_style(
        self,
        ctx: ResearchStepContext,
    ) -> tuple[ReportStyle, bool]:
        cfg = self.settings.research.report_style
        fallback_style_key = clean_whitespace(str(cfg.fallback_style)).casefold()
        if fallback_style_key not in {"decision", "explainer", "execution"}:
            fallback_style_key = "explainer"
        style = resolve_report_style(
            raw_style=ctx.plan.theme_plan.report_style,
            theme=ctx.plan.core_question or ctx.request.themes,
            enabled=bool(cfg.enabled),
            fallback_style=cast("ReportStyle", fallback_style_key),
            strict_style_lock=bool(cfg.strict_style_lock),
        )
        return style, bool(cfg.enabled and cfg.apply_render)

    def _build_track_result_packet(
        self, track_results: list[ResearchTrackResult]
    ) -> list[_RenderTrackResultPacket]:
        return [
            _RenderTrackResultPacket(
                question_id=str(item.question_id),
                question=str(item.question),
                stop_reason=str(item.stop_reason),
                rounds=int(item.rounds),
                search_calls=int(item.search_calls),
                fetch_calls=int(item.fetch_calls),
                confidence=float(item.confidence),
                coverage_ratio=float(item.coverage_ratio),
                unresolved_conflicts=int(item.unresolved_conflicts),
                track_insight_card=self._coerce_track_insight_card(
                    item.track_insight_card
                ),
                key_findings=list(item.key_findings),
                subreport_excerpt=normalize_block_text(
                    str(item.subreport_markdown or "")
                ),
            )
            for item in track_results
        ]

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
            question_id = clean_whitespace(card.question_id)
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
            token = clean_whitespace(item)
            if not token:
                continue
            if token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out

    def _render_track_results_markdown(
        self, track_results: list[_RenderTrackResultPacket]
    ) -> str:
        if not track_results:
            return "- (none)"
        lines: list[str] = []
        for index, item in enumerate(track_results, start=1):
            question = normalize_block_text(item.question) or "n/a"
            lines.extend(
                [
                    f"### Evidence Cluster {index}",
                    f"- Research question: {question}",
                    "- Insight card:",
                ]
            )
            insight_card = item.track_insight_card
            if insight_card is None:
                lines.append("  - (none)")
            else:
                lines.append(
                    f"  - Direct answer: {normalize_block_text(insight_card.direct_answer) or 'n/a'}"
                )
                lines.append("  - High-value points:")
                if insight_card.high_value_points:
                    for point in insight_card.high_value_points:
                        conclusion = normalize_block_text(point.conclusion) or "n/a"
                        condition = normalize_block_text(point.condition) or "n/a"
                        impact = normalize_block_text(point.impact) or "n/a"
                        lines.append(
                            "    - "
                            f"conclusion={conclusion}; condition={condition}; impact={impact}"
                        )
                else:
                    lines.append("    - (none)")
                lines.append("  - Tradeoffs/mechanisms:")
                if insight_card.key_tradeoffs_or_mechanisms:
                    for token in insight_card.key_tradeoffs_or_mechanisms:
                        text = normalize_block_text(token)
                        if text:
                            lines.append(f"    - {text}")
                else:
                    lines.append("    - (none)")
                lines.append("  - Unknowns/risks:")
                if insight_card.unknowns_and_risks:
                    for token in insight_card.unknowns_and_risks:
                        text = normalize_block_text(token)
                        if text:
                            lines.append(f"    - {text}")
                else:
                    lines.append("    - (none)")
                lines.append("  - Next actions:")
                if insight_card.next_actions:
                    for token in insight_card.next_actions:
                        text = normalize_block_text(token)
                        if text:
                            lines.append(f"    - {text}")
                else:
                    lines.append("    - (none)")
            lines.extend(
                [
                    "- Key findings:",
                ]
            )
            if item.key_findings:
                for token in item.key_findings:
                    finding = normalize_block_text(token)
                    if not finding:
                        continue
                    if "\n" not in finding:
                        lines.append(f"  - {finding}")
                        continue
                    lines.extend(
                        ["  -", "    ```text"]
                        + [f"    {line}" for line in finding.split("\n")]
                        + ["    ```"]
                    )
            else:
                lines.append("  - (none)")
            lines.append("- Subreport excerpt:")
            excerpt = normalize_block_text(item.subreport_excerpt)
            if excerpt:
                lines.extend(
                    ["  ```markdown"]
                    + [f"  {line}" for line in excerpt.split("\n")]
                    + ["  ```"]
                )
            else:
                lines.append("  - (none)")
        return "\n".join(lines).strip()

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
                    index=int(node.index),
                    section_id=str(node.section_id),
                    subhead=str(node.subhead),
                    cause_type=str(node.cause_type),
                    cause_message=str(node.cause_message),
                )
            )
        return out

    def _coerce_track_insight_card(
        self, raw: object | None
    ) -> TrackInsightCardPayload | None:
        if raw is None:
            return None
        if isinstance(raw, TrackInsightCardPayload):
            return raw.model_copy(deep=True)
        if isinstance(raw, dict):
            try:
                return TrackInsightCardPayload.model_validate(raw)
            except Exception:  # noqa: S112
                return None
        return None

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
