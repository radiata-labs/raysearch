from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any
from typing_extensions import override

import anyio

from serpsage.components.base import Depends
from serpsage.components.llm.base import LLMClientBase
from serpsage.core.runtime import Runtime
from serpsage.models.steps.research import (
    RenderArchitectOutput,
    RenderArchitectSectionPlan,
    ResearchStepContext,
    ResearchTrackResult,
    ResearchWriterSectionFailure,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import (
    build_render_architect_prompt_messages,
    build_render_structured_prompt_messages,
    build_render_writer_prompt_messages,
)
from serpsage.steps.research.utils import resolve_research_model


class _WriterSectionError(RuntimeError):
    def __init__(
        self,
        *,
        section_id: str,
        subhead: str,
        index: int,
        cause: Exception,
    ) -> None:
        self.section_id = section_id.strip()
        self.subhead = subhead.strip()
        self.index = index
        self.cause_type = type(cause).__name__
        self.cause_message = str(cause).strip()
        label = self.section_id or self.subhead or f"section-{self.index}"
        super().__init__(
            f"writer section failed: {label}; cause={self.cause_type}: {self.cause_message}"
        )


class ResearchRenderStep(StepBase[ResearchStepContext]):
    def __init__(self, *, rt: Runtime, llm: LLMClientBase = Depends()) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        schema = (
            dict(ctx.request.json_schema)
            if isinstance(ctx.request.json_schema, dict)
            else None
        )
        if schema is None:
            await self._render_markdown_architect_writer(
                ctx=ctx,
                target_language=ctx.task.output_language,
                now_utc=now_utc,
            )
            await self.emit_tracking_event(
                event_name="research.render.summary",
                request_id=ctx.request_id,
                stage="render",
                attrs={
                    "mode": "final_markdown",
                    "track_results": len(ctx.result.tracks),
                    "content_chars": len(ctx.result.content),
                    "mode_depth_profile": ctx.run.limits.mode_key,
                    "report_style_selected": ctx.task.style,
                },
            )
            return ctx
        await self._render_structured_once(
            ctx=ctx,
            schema=schema,
            target_language=ctx.task.output_language,
            now_utc=now_utc,
        )
        await self.emit_tracking_event(
            event_name="research.render.summary",
            request_id=ctx.request_id,
            stage="render",
            attrs={
                "mode": "final_structured",
                "track_results": len(ctx.result.tracks),
                "has_structured": ctx.result.structured is not None,
                "mode_depth_profile": ctx.run.limits.mode_key,
                "report_style_selected": ctx.task.style,
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
        writer_outputs = await self._run_writers(
            ctx=ctx,
            architect_output=architect_output,
            target_language=target_language,
            now_utc=now_utc,
        )
        ctx.result.structured = None
        ctx.result.content = self._assemble_markdown_from_sections(
            ctx=ctx,
            architect_output=architect_output,
            writer_outputs=writer_outputs,
        )

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
            raise
        return chat_result.data

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
            async with anyio.create_task_group() as task_group:
                for index, section in enumerate(architect_output.sections):
                    task_group.start_soon(
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
                    "failed_sections": [item.to_payload() for item in failed_sections],
                    "message": str(exc),
                },
            )
            raise
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
        try:
            result = await self._llm.create(
                model=model,
                messages=build_render_writer_prompt_messages(
                    ctx=ctx,
                    target_language=target_language,
                    now_utc=now_utc,
                    architect_output=architect_output,
                    section=section,
                    track_results=self._select_track_results_for_section(
                        ctx=ctx,
                        section=section,
                    ),
                ),
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
        if not ctx.result.tracks or not section.question_ids:
            return []
        track_result_map = {item.question_id: item for item in ctx.result.tracks}
        selected = [
            track_result_map[question_id].model_copy(deep=True)
            for question_id in section.question_ids
            if question_id in track_result_map
        ]
        selected.sort(
            key=lambda item: (
                float(item.confidence),
                float(item.coverage_ratio),
                -int(item.unresolved_conflicts),
                len(item.key_findings),
            ),
            reverse=True,
        )
        return selected

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
        try:
            result = await self._llm.create(
                model=model,
                messages=build_render_structured_prompt_messages(
                    ctx=ctx,
                    target_language=target_language,
                    now_utc=now_utc,
                ),
                response_format=schema,
            )
            if not isinstance(result.data, dict):
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
            raise
        ctx.result.structured = result.data
        ctx.result.content = json.dumps(result.data, ensure_ascii=False, indent=2)

    def _assemble_markdown_from_sections(
        self,
        *,
        ctx: ResearchStepContext,
        architect_output: RenderArchitectOutput,
        writer_outputs: list[str],
    ) -> str:
        parts: list[str] = [f"# {self._resolve_report_title(ctx)}"]
        if architect_output.report_objective:
            parts.append(architect_output.report_objective)
        for section_plan, section_content in zip(
            architect_output.sections,
            writer_outputs,
            strict=False,
        ):
            parts.append(f"## {section_plan.subhead}")
            parts.append(section_content)
        return "\n\n".join(parts)

    def _resolve_report_title(self, ctx: ResearchStepContext) -> str:
        return ctx.task.question or ctx.request.themes or "Research Report"

    def _collect_writer_section_failures(
        self,
        exc: BaseException,
    ) -> list[ResearchWriterSectionFailure]:
        failures: list[ResearchWriterSectionFailure] = []
        stack: list[BaseException] = [exc]
        while stack and len(failures) < 16:
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
            failures.append(
                ResearchWriterSectionFailure(
                    index=node.index,
                    section_id=node.section_id,
                    subhead=node.subhead,
                    cause_type=node.cause_type,
                    cause_message=node.cause_message,
                )
            )
        return failures


__all__ = ["ResearchRenderStep"]
