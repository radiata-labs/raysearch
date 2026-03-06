from __future__ import annotations

from datetime import UTC, datetime
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import ResearchSource, ResearchStepContext
from serpsage.models.research import (
    ReportStyle,
    SubreportOutputPayload,
    TrackInsightCardPayload,
    TrackInsightPointPayload,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.prompt import (
    build_subreport_prompt_messages,
)
from serpsage.steps.research.search import (
    pick_sources_by_ids,
    select_context_source_ids,
)
from serpsage.steps.research.utils import resolve_research_model

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime


class ResearchSubreportStep(StepBase[ResearchStepContext]):
    _CONTEXT_NEW_RESULT_TARGET_RATIO = 0.60
    _CONTEXT_MIN_HISTORY_SOURCES = 3

    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        target_language = ctx.plan.theme_plan.output_language
        await self._render_subreport(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
        )
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
        require_insight_card = self._require_insight_card(ctx)
        messages = build_subreport_prompt_messages(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
            source_evidence=self._select_sources_for_render(ctx),
            source_evidence_max_chars=max(1, ctx.runtime.mode_depth.source_chars),
            notes=self._collect_recent_notes(ctx, limit=12),
            require_insight_card=require_insight_card,
        )
        markdown_text = ""
        insight_card: TrackInsightCardPayload | None = None
        try:
            result = await self._llm.create(
                model=model,
                messages=messages,
                response_format=SubreportOutputPayload,
                retries=self.settings.research.llm_self_heal_retries,
            )
            payload = result.data
            markdown_text = payload.subreport_markdown
            insight_card = payload.track_insight_card
        except Exception as exc:  # noqa: BLE001
            await self.emit_tracking_event(
                event_name="research.subreport.error",
                request_id=ctx.request_id,
                stage="subreport",
                status="error",
                error_code="research_render_subreport_failed",
                error_type=type(exc).__name__,
                attrs={
                    "model": model,
                    "message": str(exc),
                },
            )
        if not markdown_text.strip():
            markdown_text = self._build_subreport_fallback(ctx)
        if require_insight_card and insight_card is None:
            insight_card = self._build_fallback_insight_card(ctx=ctx)
        ctx.output.structured = (
            insight_card.model_dump(mode="json") if insight_card is not None else None
        )
        ctx.output.content = self._normalize_markdown(markdown_text)
        report_style = ctx.plan.theme_plan.report_style
        await self.emit_tracking_event(
            event_name="research.style.applied",
            request_id=ctx.request_id,
            stage="subreport",
            attrs={
                "report_style_selected": report_style,
                "require_insight_card": require_insight_card,
                "has_insight_card": insight_card is not None,
            },
        )

    def _build_subreport_fallback(self, ctx: ResearchStepContext) -> str:
        core_question = ctx.plan.theme_plan.core_question
        sources = self._select_sources_for_render(ctx)
        round_state = ctx.rounds[-1] if ctx.rounds else None
        report_style = ctx.plan.theme_plan.report_style
        answer_status = self._fallback_answer_status(
            round_state=round_state,
            has_sources=len(sources) > 0,
        )
        evidence_lines = self._fallback_evidence_lines(sources, limit=6)
        uncertainty_lines = self._fallback_uncertainty_lines(
            round_state=round_state,
            has_sources=len(sources) > 0,
        )
        next_checks = self._fallback_next_checks(round_state=round_state)
        sections = [f"## {core_question or 'Core Question'}"]
        sections.extend(
            self._build_style_fallback_sections(
                report_style=report_style,
                answer_status=answer_status,
                evidence_lines=evidence_lines,
                uncertainty_lines=uncertainty_lines,
                next_checks=next_checks,
            )
        )
        return "\n\n".join(sections)

    def _build_fallback_insight_card(
        self, *, ctx: ResearchStepContext
    ) -> TrackInsightCardPayload:
        round_state = ctx.rounds[-1] if ctx.rounds else None
        confidence = (
            float(getattr(round_state, "confidence", 0.0)) if round_state else 0.0
        )
        coverage_ratio = (
            float(getattr(round_state, "coverage_ratio", 0.0)) if round_state else 0.0
        )
        unresolved_conflicts = (
            getattr(round_state, "unresolved_conflicts", 0) if round_state else 0
        )
        missing_entities = (
            list(getattr(round_state, "missing_entities", []))[:4]
            if round_state is not None
            else []
        )
        direct_answer = (
            "Current evidence supports a stable answer."
            if unresolved_conflicts <= 0 and confidence >= 0.75
            else "Current answer remains partial and requires additional verification."
        )
        high_value_points = [
            TrackInsightPointPayload(
                conclusion="Confidence trend is available from the latest round.",
                condition=f"confidence={confidence:.3f}, coverage_ratio={coverage_ratio:.3f}",
                impact=(
                    "Low confidence or coverage indicates additional targeted research is needed."
                ),
            ),
            TrackInsightPointPayload(
                conclusion="Conflict level directly affects final recommendation stability.",
                condition=f"unresolved_conflicts={unresolved_conflicts}",
                impact="Higher unresolved conflict means lower decision reliability.",
            ),
        ]
        return TrackInsightCardPayload(
            direct_answer=direct_answer,
            high_value_points=high_value_points,
            key_tradeoffs_or_mechanisms=[
                "Higher confidence usually requires broader source coverage and conflict resolution."
            ],
            unknowns_and_risks=[
                "Residual ambiguity remains where sources disagree on high-impact claims."
            ],
            next_actions=[
                (
                    "Add targeted verification queries for missing entities: "
                    + ", ".join(missing_entities)
                )
                if missing_entities
                else "Add one additional authoritative source to validate the key claim."
            ],
        )

    def _require_insight_card(self, ctx: ResearchStepContext) -> bool:
        return ctx.runtime.mode_depth.mode_key != "research-fast"

    def _fallback_answer_status(
        self,
        *,
        round_state: object | None,
        has_sources: bool,
    ) -> str:
        if not has_sources:
            return (
                "Current evidence is insufficient for a confident answer; the question "
                "remains unresolved."
            )
        if round_state is None:
            return (
                "Current evidence supports a partial answer, but verification is still "
                "needed for higher confidence."
            )
        unresolved_conflicts = getattr(round_state, "unresolved_conflicts", 0)
        critical_gaps = getattr(round_state, "critical_gaps", 0)
        if unresolved_conflicts <= 0 and critical_gaps <= 0:
            return (
                "Current evidence supports a stable answer, with no major unresolved "
                "conflicts or critical gaps in the latest review."
            )
        if unresolved_conflicts > 0:
            return (
                "The answer is still partial because key claims remain in conflict "
                "across sources."
            )
        return (
            "The answer is directionally clear, but material evidence gaps still "
            "limit confidence."
        )

    def _fallback_evidence_lines(
        self,
        sources: list[ResearchSource],
        *,
        limit: int,
    ) -> list[str]:
        if not sources:
            return ["- No source evidence is available yet."]
        out: list[str] = []
        max_items = max(1, limit)
        for source in sources[:max_items]:
            title = source.title or "Untitled source"
            url = source.url
            snippet_raw = source.overview
            if not snippet_raw:
                snippet_raw = source.content
            snippet = self._truncate_inline_text(snippet_raw, max_chars=240)
            if not snippet:
                snippet = "Relevant context is available from this source."
            if url:
                out.append(f"- [{title}]({url}): {snippet}")
                continue
            out.append(f"- {title}: {snippet}")
        return out or ["- No source evidence is available yet."]

    def _fallback_uncertainty_lines(
        self,
        *,
        round_state: object | None,
        has_sources: bool,
    ) -> list[str]:
        if not has_sources:
            return [
                "- No reliable evidence cluster is available yet.",
                "- Any conclusion at this stage would be speculative.",
            ]
        if round_state is None:
            return [
                "- Evidence quality varies across sources and still needs targeted verification.",
                "- Confidence remains provisional until conflicts and missing data are resolved.",
            ]
        unresolved_conflicts = getattr(round_state, "unresolved_conflicts", 0)
        critical_gaps = getattr(round_state, "critical_gaps", 0)
        lines: list[str] = []
        if unresolved_conflicts > 0:
            lines.append(
                "- Conflicting claims remain on key points and prevent a fully decisive conclusion."
            )
        else:
            lines.append(
                "- No major claim conflict is currently blocking the main conclusion."
            )
        if critical_gaps > 0:
            lines.append(
                "- Important evidence gaps remain and still affect confidence boundaries."
            )
        else:
            lines.append(
                "- No major critical gap is currently identified in available evidence."
            )
        return lines

    def _fallback_next_checks(self, *, round_state: object | None) -> list[str]:
        lines = [
            "- Validate the most consequential claim with one additional high-authority primary source.",
            "- Cross-check numerical or time-sensitive statements against the most recent official publication.",
        ]
        if round_state is None:
            return lines
        raw_entities = getattr(round_state, "missing_entities", [])
        missing_entities = [item for item in list(raw_entities)[:4] if item]
        if missing_entities:
            entity_text = ", ".join(missing_entities)
            lines.append(f"- Add direct evidence for missing entities: {entity_text}.")
        return lines

    def _build_style_fallback_sections(
        self,
        *,
        report_style: ReportStyle,
        answer_status: str,
        evidence_lines: list[str],
        uncertainty_lines: list[str],
        next_checks: list[str],
    ) -> list[str]:
        if report_style == "decision":
            return [
                "### Verdict snapshot",
                f"- {answer_status}",
                "### Trade-offs",
                "\n".join(evidence_lines),
                "### Scenario recommendation",
                "- Recommendation remains conditional on scenario constraints and evidence quality.",
                "### Risk triggers",
                "\n".join(uncertainty_lines),
                "### Next checks",
                "\n".join(next_checks),
            ]
        if report_style == "execution":
            return [
                "### Goal and prerequisites",
                f"- {answer_status}",
                "### Step sequence",
                "\n".join(evidence_lines),
                "### Validation criteria",
                "- Confirm each critical claim against at least one authoritative primary source before execution.",
                "- Re-check time-sensitive requirements against the latest official publication date.",
                "### Failure handling",
                "\n".join(uncertainty_lines),
                "### Next actions",
                "\n".join(next_checks),
            ]
        return [
            "### Core model",
            f"- {answer_status}",
            "### Mechanisms",
            "\n".join(evidence_lines),
            "### Boundary cases",
            "\n".join(uncertainty_lines),
            "### Common misconceptions",
            "- Evidence consensus can still hide condition-sensitive exceptions.",
            "- High-level summaries should not be treated as universal across all contexts.",
            "### Practical takeaway",
            "\n".join(next_checks),
        ]

    def _truncate_inline_text(self, text: str, *, max_chars: int) -> str:
        raw = text.strip()
        if not raw:
            return ""
        limit = max(1, max_chars)
        if len(raw) <= limit:
            return raw
        clipped = raw[: limit - 1].rstrip()
        if not clipped:
            return raw[:limit]
        return f"{clipped}..."

    def _collect_recent_notes(
        self,
        ctx: ResearchStepContext,
        *,
        limit: int,
    ) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for item in reversed(ctx.notes):
            if not item:
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            out.append(item)
            if len(out) >= max(1, limit):
                break
        out.reverse()
        return out

    def _select_sources_for_render(
        self,
        ctx: ResearchStepContext,
    ) -> list[ResearchSource]:
        mode_depth = ctx.runtime.mode_depth
        limit = max(1, mode_depth.source_topk)
        latest_round_index = 0
        if ctx.rounds:
            latest_round_index = ctx.rounds[-1].round_index
        elif ctx.current_round is not None:
            latest_round_index = ctx.current_round.round_index
        else:
            latest_round_index = max(
                (item.round_index for item in ctx.corpus.sources),
                default=0,
            )
        selected_ids = select_context_source_ids(
            ctx=ctx,
            round_index=latest_round_index,
            topk=limit,
            new_result_target_ratio=float(self._CONTEXT_NEW_RESULT_TARGET_RATIO),
            min_history_sources=self._CONTEXT_MIN_HISTORY_SOURCES,
        )
        if selected_ids:
            selected = pick_sources_by_ids(
                sources=ctx.corpus.sources,
                source_ids=selected_ids,
            )
            if selected:
                return selected
        fallback = sorted(
            ctx.corpus.sources,
            key=lambda item: (item.round_index, item.source_id),
            reverse=True,
        )
        return list(fallback[:limit])

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


__all__ = ["ResearchSubreportStep"]
