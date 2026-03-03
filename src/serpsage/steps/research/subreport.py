from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import TYPE_CHECKING, cast
from typing_extensions import override

from serpsage.models.pipeline import ResearchSource, ResearchStepContext
from serpsage.models.research import (
    ReportStyle,
    ResearchThemePlan,
    SubreportOutputPayload,
    TrackInsightCardPayload,
    TrackInsightPointPayload,
)
from serpsage.steps.base import StepBase
from serpsage.steps.research.context import render_theme_plan_markdown
from serpsage.steps.research.language import normalize_language_code
from serpsage.steps.research.prompt import (
    build_subreport_messages as build_subreport_prompt_messages,
)
from serpsage.steps.research.prompt import (
    resolve_report_style,
)
from serpsage.steps.research.search import (
    pick_sources_by_ids,
    select_context_source_ids,
)
from serpsage.steps.research.utils import resolve_research_model
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.llm.base import LLMClientBase
    from serpsage.core.runtime import Runtime


@dataclass(slots=True)
class _SubreportRoundTrajectoryItem:
    round_index: int
    query_strategy: str
    queries: list[str] = field(default_factory=list)
    result_count: int = 0
    confidence: float = 0.0
    coverage_ratio: float = 0.0
    unresolved_conflicts: int = 0
    critical_gaps: int = 0
    stop: bool = False
    stop_reason: str = "n/a"


@dataclass(slots=True)
class _SubreportSourceEvidenceItem:
    source_id: int
    url: str
    title: str
    round_index: int
    is_subpage: bool
    overview: str = ""
    content_excerpt: str = ""


@dataclass(slots=True)
class _SubreportContextPacket:
    theme: str
    core_question: str
    report_style: str
    target_output_language: str
    utc_timestamp: str
    utc_date: str
    theme_plan: ResearchThemePlan
    round_trajectory: list[_SubreportRoundTrajectoryItem] = field(default_factory=list)
    source_evidence: list[_SubreportSourceEvidenceItem] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    subreport_objective: str = ""


class ResearchSubreportStep(StepBase[ResearchStepContext]):
    _MAX_SOURCES_FOR_CONTEXT = 12
    _MAX_OVERVIEW_CHARS = 3200
    _MAX_CONTENT_EXCERPT_CHARS = 2200
    _MAX_TOTAL_CONTENT_CHARS = 22000
    _CONTEXT_NEW_RESULT_TARGET_RATIO = 0.60
    _CONTEXT_MIN_HISTORY_SOURCES = 3

    def __init__(self, *, rt: Runtime, llm: LLMClientBase) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self.bind_deps(llm)

    @override
    async def run_inner(self, ctx: ResearchStepContext) -> ResearchStepContext:
        now_utc = datetime.fromtimestamp(self.clock.now_ms() / 1000, tz=UTC)
        target_language = self._resolve_target_language(ctx)
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
        messages = self._build_subreport_messages(
            ctx,
            target_language=target_language,
            now_utc=now_utc,
        )
        require_insight_card = self._require_insight_card(ctx)
        markdown_text = ""
        insight_card: TrackInsightCardPayload | None = None
        try:
            result = await self._llm.create(
                model=model,
                messages=messages,
                response_format=SubreportOutputPayload,
                retries=int(self.settings.research.llm_self_heal_retries),
            )
            payload = result.data
            markdown_text = str(payload.subreport_markdown or "")
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
                    "model": str(model),
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
        report_style, style_applied = self._resolve_report_style(ctx)
        await self.emit_tracking_event(
            event_name="research.style.applied",
            request_id=ctx.request_id,
            stage="subreport",
            attrs={
                "report_style_selected": str(report_style),
                "style_applied_stage": "subreport" if style_applied else "none",
                "require_insight_card": bool(require_insight_card),
                "has_insight_card": bool(insight_card is not None),
            },
        )

    def _build_subreport_messages(
        self,
        ctx: ResearchStepContext,
        *,
        target_language: str,
        now_utc: datetime,
    ) -> list[dict[str, str]]:
        target_language_name = clean_whitespace(target_language) or "unspecified"
        core_question = clean_whitespace(ctx.plan.core_question or ctx.request.themes)
        report_style, style_applied = self._resolve_report_style(ctx)
        context_packet_markdown = self._build_subreport_context_packet_markdown(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
            core_question=core_question,
        )
        require_insight_card = self._require_insight_card(ctx)
        return build_subreport_prompt_messages(
            target_output_language=target_language,
            target_output_language_label=target_language_name,
            current_utc_date=now_utc.date().isoformat(),
            core_question=core_question,
            mode_depth_profile=str(ctx.runtime.mode_depth.mode_key),
            report_style=report_style,
            style_applied=bool(style_applied),
            require_insight_card=bool(require_insight_card),
            context_packet_markdown=context_packet_markdown,
        )

    def _build_subreport_fallback(self, ctx: ResearchStepContext) -> str:
        core_question = clean_whitespace(ctx.plan.core_question or ctx.request.themes)
        sources = self._select_sources_for_render(ctx, max_sources=10)
        round_state = ctx.rounds[-1] if ctx.rounds else None
        report_style, style_applied = self._resolve_report_style(ctx)
        answer_status = self._fallback_answer_status(
            round_state=round_state,
            has_sources=bool(sources),
        )
        evidence_lines = self._fallback_evidence_lines(sources, limit=6)
        uncertainty_lines = self._fallback_uncertainty_lines(
            round_state=round_state,
            has_sources=bool(sources),
        )
        next_checks = self._fallback_next_checks(round_state=round_state)
        sections = [f"## {core_question or 'Core Question'}"]
        if not style_applied:
            sections.extend(
                [
                    "### Direct answer",
                    f"- {answer_status}",
                    "### Evidence highlights",
                    "\n".join(evidence_lines),
                    "### Conflicts and uncertainty",
                    "\n".join(uncertainty_lines),
                    "### Targeted next checks",
                    "\n".join(next_checks),
                ]
            )
            return "\n\n".join(sections)
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
            int(getattr(round_state, "unresolved_conflicts", 0)) if round_state else 0
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
                condition=f"unresolved_conflicts={int(unresolved_conflicts)}",
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
                    + ", ".join(str(item) for item in missing_entities)
                )
                if missing_entities
                else "Add one additional authoritative source to validate the key claim."
            ],
        )

    def _require_insight_card(self, ctx: ResearchStepContext) -> bool:
        return str(ctx.runtime.mode_depth.mode_key) != "research-fast"

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
        unresolved_conflicts = int(getattr(round_state, "unresolved_conflicts", 0))
        critical_gaps = int(getattr(round_state, "critical_gaps", 0))
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
        max_items = max(1, int(limit))
        for source in sources[:max_items]:
            title = clean_whitespace(source.title or "") or "Untitled source"
            url = clean_whitespace(str(source.url or ""))
            snippet_raw = clean_whitespace(str(source.overview or ""))
            if not snippet_raw:
                snippet_raw = clean_whitespace(str(source.content or ""))
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
        unresolved_conflicts = int(getattr(round_state, "unresolved_conflicts", 0))
        critical_gaps = int(getattr(round_state, "critical_gaps", 0))
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
        missing_entities = [
            token
            for item in list(raw_entities)[:4]
            if (token := clean_whitespace(str(item)))
        ]
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
        raw = clean_whitespace(str(text or ""))
        if not raw:
            return ""
        limit = max(1, int(max_chars))
        if len(raw) <= limit:
            return raw
        clipped = raw[: limit - 1].rstrip()
        if not clipped:
            return raw[:limit]
        return f"{clipped}..."

    def _build_subreport_context_packet_markdown(
        self,
        *,
        ctx: ResearchStepContext,
        target_language: str,
        now_utc: datetime,
        core_question: str,
    ) -> str:
        packet = self._build_subreport_context_packet(
            ctx=ctx,
            target_language=target_language,
            now_utc=now_utc,
            core_question=core_question,
        )
        lines: list[str] = [
            "# Subreport Context Packet",
            "## Theme",
            self._normalize_block_text(packet.theme) or "n/a",
            "## Core Question",
            self._normalize_block_text(packet.core_question) or "n/a",
            "## Report Style",
            self._normalize_block_text(packet.report_style) or "n/a",
            "## Target Output Language",
            self._normalize_block_text(packet.target_output_language) or "n/a",
            "## Time Context",
            f"- UTC timestamp: {packet.utc_timestamp}",
            f"- UTC date: {packet.utc_date}",
            "## Subreport Objective",
            packet.subreport_objective,
            "## Private Rendering Rules",
            "- SUBREPORT_CONTEXT_PACKET is private working context and must not be exposed verbatim in user-facing output.",
            "- Never expose internal metadata: source IDs, round indexes, query logs, stop reasons, confidence/coverage metrics, or packet labels.",
            "## Theme Plan",
        ]
        lines.append(render_theme_plan_markdown(packet.theme_plan, include_title=False))
        lines.extend(["## Round Trajectory"])
        if packet.round_trajectory:
            for trajectory_item in packet.round_trajectory:
                round_index = int(trajectory_item.round_index)
                lines.extend(
                    [
                        f"### Round {round_index}",
                        f"- Query strategy: {self._normalize_block_text(trajectory_item.query_strategy) or 'n/a'}",
                        f"- Result count: {int(trajectory_item.result_count)}",
                        f"- Confidence: {float(trajectory_item.confidence):.3f}",
                        f"- Coverage ratio: {float(trajectory_item.coverage_ratio):.3f}",
                        f"- Unresolved conflicts: {int(trajectory_item.unresolved_conflicts)}",
                        f"- Critical gaps: {int(trajectory_item.critical_gaps)}",
                        f"- Stop: {bool(trajectory_item.stop)}",
                        f"- Stop reason: {self._normalize_block_text(trajectory_item.stop_reason) or 'n/a'}",
                    ]
                )
                if trajectory_item.queries:
                    lines.append("- Queries:")
                    for query in trajectory_item.queries:
                        token = self._normalize_block_text(str(query))
                        if not token:
                            continue
                        if "\n" not in token:
                            lines.append(f"  - {token}")
                            continue
                        lines.extend(
                            ["  -", "    ```text"]
                            + [f"    {line}" for line in token.split("\n")]
                            + ["    ```"]
                        )
                else:
                    lines.append("- Queries: (none)")
        else:
            lines.append("- No round trajectory available.")
        lines.extend(["## Source Evidence"])
        if packet.source_evidence:
            for source in packet.source_evidence:
                source_id = int(source.source_id)
                title = clean_whitespace(source.title) or "Untitled"
                url = clean_whitespace(source.url) or "n/a"
                round_index = int(source.round_index)
                is_subpage = bool(source.is_subpage)
                lines.extend(
                    [
                        f"### Source {source_id}: {title}",
                        f"- URL: {url}",
                        f"- Round index: {round_index}",
                        f"- Is subpage: {is_subpage}",
                    ]
                )
                overview = self._normalize_block_text(source.overview)
                lines.append("- Overview:")
                if overview:
                    lines.extend(
                        ["  ```text"]
                        + [f"  {line}" for line in overview.split("\n")]
                        + ["  ```"]
                    )
                else:
                    lines.append("  - (none)")
                excerpt = self._normalize_block_text(source.content_excerpt)
                if excerpt:
                    lines.extend(
                        [
                            "- Content excerpt:",
                            "  ```text",
                            *[f"  {line}" for line in excerpt.split("\n")],
                            "  ```",
                        ]
                    )
                else:
                    lines.append("- Content excerpt: (none)")
        else:
            lines.append("- No source evidence available.")
        lines.extend(["## Notes"])
        if packet.notes:
            lines.extend(f"- {item}" for item in packet.notes)
        else:
            lines.append("- (none)")
        return "\n".join(lines).strip()

    def _build_subreport_context_packet(
        self,
        *,
        ctx: ResearchStepContext,
        target_language: str,
        now_utc: datetime,
        core_question: str,
    ) -> _SubreportContextPacket:
        selected_sources = self._select_sources_for_render(
            ctx,
            max_sources=self._MAX_SOURCES_FOR_CONTEXT,
        )
        report_style, style_applied = self._resolve_report_style(ctx)
        style_label = report_style if style_applied else "baseline"
        return _SubreportContextPacket(
            theme=self._normalize_block_text(str(ctx.request.themes)),
            core_question=self._normalize_block_text(core_question),
            report_style=self._normalize_block_text(style_label),
            target_output_language=self._normalize_block_text(target_language),
            utc_timestamp=now_utc.isoformat(),
            utc_date=now_utc.date().isoformat(),
            theme_plan=ctx.plan.theme_plan.model_copy(deep=True),
            round_trajectory=self._build_round_trajectory_packet(ctx),
            source_evidence=self._build_source_evidence_packet(selected_sources),
            notes=self._collect_recent_notes(ctx, limit=12),
            subreport_objective=self._subreport_objective_for_style(
                report_style=report_style,
                style_applied=style_applied,
            ),
        )

    def _subreport_objective_for_style(
        self,
        *,
        report_style: ReportStyle,
        style_applied: bool,
    ) -> str:
        if not style_applied:
            return (
                "Build an evidence archive for one core question with complete, "
                "traceable detail and explicit uncertainty boundaries."
            )
        if report_style == "decision":
            return (
                "Produce a decision-focused subreport with scenario-fit recommendations, "
                "trade-offs, and explicit risk triggers."
            )
        if report_style == "execution":
            return (
                "Produce an execution-focused subreport with prerequisites, step sequence, "
                "validation criteria, and failure handling boundaries."
            )
        return (
            "Produce an explainer-focused subreport that clarifies mechanisms, "
            "boundary conditions, and practical understanding."
        )

    def _build_round_trajectory_packet(
        self,
        ctx: ResearchStepContext,
    ) -> list[_SubreportRoundTrajectoryItem]:
        rounds = ctx.rounds[-8:]
        return [
            _SubreportRoundTrajectoryItem(
                round_index=int(round_state.round_index),
                query_strategy=clean_whitespace(round_state.query_strategy or "n/a"),
                queries=[
                    token
                    for item in round_state.queries[:8]
                    if (token := self._normalize_block_text(item))
                ],
                result_count=int(round_state.result_count),
                confidence=float(round_state.confidence),
                coverage_ratio=float(round_state.coverage_ratio),
                unresolved_conflicts=int(round_state.unresolved_conflicts),
                critical_gaps=int(round_state.critical_gaps),
                stop=bool(round_state.stop),
                stop_reason=clean_whitespace(round_state.stop_reason or "n/a"),
            )
            for round_state in rounds
        ]

    def _build_source_evidence_packet(
        self,
        sources: list[ResearchSource],
    ) -> list[_SubreportSourceEvidenceItem]:
        out: list[_SubreportSourceEvidenceItem] = []
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
                _SubreportSourceEvidenceItem(
                    source_id=int(source.source_id),
                    url=str(source.url),
                    title=clean_whitespace(source.title or ""),
                    round_index=int(source.round_index),
                    is_subpage=bool(source.is_subpage),
                    overview=self._normalize_block_text(str(source.overview or ""))[
                        : self._MAX_OVERVIEW_CHARS
                    ],
                    content_excerpt=content_excerpt,
                )
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
            item = self._normalize_block_text(raw)
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
        language_code = normalize_language_code(
            ctx.plan.output_language or ctx.plan.input_language,
            default="other",
        )
        if language_code != "other":
            return language_code
        return "en"

    def _select_sources_for_render(
        self,
        ctx: ResearchStepContext,
        *,
        max_sources: int,
    ) -> list[ResearchSource]:
        mode_depth = ctx.runtime.mode_depth
        subreport_topk = max(1, int(mode_depth.subreport_context_topk_override))
        limit = max(
            1,
            min(
                int(max_sources),
                int(subreport_topk),
            ),
        )
        latest_round_index = 0
        if ctx.rounds:
            latest_round_index = int(ctx.rounds[-1].round_index)
        elif ctx.current_round is not None:
            latest_round_index = int(ctx.current_round.round_index)
        else:
            latest_round_index = max(
                (int(item.round_index) for item in ctx.corpus.sources),
                default=0,
            )
        selected_ids = select_context_source_ids(
            ctx=ctx,
            round_index=latest_round_index,
            topk=limit,
            new_result_target_ratio=float(self._CONTEXT_NEW_RESULT_TARGET_RATIO),
            min_history_sources=int(self._CONTEXT_MIN_HISTORY_SOURCES),
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
            key=lambda item: (int(item.round_index), int(item.source_id)),
            reverse=True,
        )
        return list(fallback[:limit])

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
        return style, bool(cfg.enabled and cfg.apply_subreport)


__all__ = ["ResearchSubreportStep"]
