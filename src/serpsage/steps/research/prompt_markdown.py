from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.models.pipeline import (
    ResearchQuestionCard,
    ResearchRoundState,
)
from serpsage.models.research import (
    OverviewOutputPayload,
    RenderArchitectOutput,
    RenderArchitectSectionPlan,
    ResearchThemePlan,
    ResearchThemePlanCard,
)
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from collections.abc import Iterable


def normalize_block_text(text: str) -> str:
    return str(text or "").replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_scalar_text(value: object) -> str:
    return clean_whitespace(str(value or ""))


def _render_markdown_bullets(values: Iterable[str], *, indent: str = "") -> list[str]:
    out: list[str] = []
    for item in values:
        token = normalize_block_text(str(item))
        if not token:
            continue
        if "\n" not in token:
            out.append(f"{indent}- {token}")
            continue
        out.append(f"{indent}-")
        out.append(f"{indent}  ```text")
        out.extend(f"{indent}  {line}" for line in token.split("\n"))
        out.append(f"{indent}  ```")
    return out


def render_theme_plan_markdown(
    plan: ResearchThemePlan,
    *,
    include_title: bool = True,
    title_level: int = 3,
    include_question_cards: bool = True,
) -> str:
    lines: list[str] = []
    if include_title:
        level = max(1, int(title_level))
        lines.append(f"{'#' * level} Theme Plan")
    lines.extend(
        [
            f"- Core question: {_normalize_scalar_text(plan.core_question) or 'n/a'}",
            f"- Report style: {_normalize_scalar_text(plan.report_style) or 'n/a'}",
            f"- Input language: {_normalize_scalar_text(plan.input_language) or 'n/a'}",
            f"- Output language: {_normalize_scalar_text(plan.output_language) or 'n/a'}",
        ]
    )
    lines.append("- Subthemes:")
    lines.extend(
        _render_markdown_bullets(plan.subthemes, indent="  ") or ["  - (none)"]
    )
    lines.append("- Required entities:")
    lines.extend(
        _render_markdown_bullets(plan.required_entities, indent="  ") or ["  - (none)"]
    )
    if include_question_cards:
        lines.append("- Question cards:")
        if plan.question_cards:
            for index, card in enumerate(plan.question_cards, start=1):
                lines.extend(
                    _render_theme_plan_card_lines(card=card, index=index, indent="  ")
                )
        else:
            lines.append("  - (none)")
    return "\n".join(lines).strip()


def _render_theme_plan_card_lines(
    *,
    card: ResearchThemePlanCard | ResearchQuestionCard,
    index: int,
    indent: str,
) -> list[str]:
    subindent = f"{indent}  "
    leaf_indent = f"{subindent}  "
    lines: list[str] = [
        (
            f"{indent}- Card {index} "
            f"(id={_normalize_scalar_text(getattr(card, 'question_id', '')) or f'q{index}'}, "
            f"priority={int(getattr(card, 'priority', 0) or 0)})"
        ),
        f"{subindent}- Question: {_normalize_scalar_text(getattr(card, 'question', '')) or 'n/a'}",
    ]
    seed_queries = list(getattr(card, "seed_queries", []))
    evidence_focus = list(getattr(card, "evidence_focus", []))
    expected_gain = _normalize_scalar_text(getattr(card, "expected_gain", "")) or "n/a"
    lines.append(f"{subindent}- Seed queries:")
    lines.extend(
        _render_markdown_bullets(seed_queries, indent=leaf_indent)
        or [f"{leaf_indent}- (none)"]
    )
    lines.append(f"{subindent}- Evidence focus:")
    lines.extend(
        _render_markdown_bullets(evidence_focus, indent=leaf_indent)
        or [f"{leaf_indent}- (none)"]
    )
    lines.append(f"{subindent}- Expected gain: {expected_gain}")
    return lines


def render_rounds_markdown(rounds: list[ResearchRoundState], *, limit: int) -> str:
    selected = rounds[-max(1, int(limit)) :]
    if not selected:
        return "- (none)"
    lines: list[str] = []
    for round_state in selected:
        lines.extend(
            [
                f"### Round {int(round_state.round_index)}",
                (
                    f"- Query strategy: "
                    f"{_normalize_scalar_text(round_state.query_strategy) or 'n/a'}"
                ),
                f"- Result count: {int(round_state.result_count)}",
                f"- Confidence: {float(round_state.confidence):.3f}",
                f"- Coverage ratio: {float(round_state.coverage_ratio):.3f}",
                f"- Entity coverage complete: {bool(round_state.entity_coverage_complete)}",
                f"- Unresolved conflicts: {int(round_state.unresolved_conflicts)}",
                f"- Critical gaps: {int(round_state.critical_gaps)}",
                f"- Stop: {bool(round_state.stop)}",
                f"- Stop reason: {_normalize_scalar_text(round_state.stop_reason) or 'n/a'}",
                "- Queries:",
            ]
        )
        lines.extend(
            _render_markdown_bullets(round_state.queries, indent="  ") or ["  - (none)"]
        )
        overview_summary = normalize_block_text(str(round_state.overview_summary or ""))
        content_summary = normalize_block_text(str(round_state.content_summary or ""))
        lines.append("- Overview summary:")
        if overview_summary:
            lines.extend(
                ["  ```text"]
                + [f"  {line}" for line in overview_summary.split("\n")]
                + ["  ```"]
            )
        else:
            lines.append("  - (none)")
        lines.append("- Content summary:")
        if content_summary:
            lines.extend(
                ["  ```text"]
                + [f"  {line}" for line in content_summary.split("\n")]
                + ["  ```"]
            )
        else:
            lines.append("  - (none)")
        lines.append("- Missing entities:")
        lines.extend(
            _render_markdown_bullets(round_state.missing_entities, indent="  ")
            or ["  - (none)"]
        )
    return "\n".join(lines).strip()


def render_overview_review_markdown(review: OverviewOutputPayload) -> str:
    lines: list[str] = [
        "### Overview Review",
        f"- Confidence: {float(review.confidence):.3f}",
        f"- Entity coverage complete: {bool(review.entity_coverage_complete)}",
        f"- Next query strategy: {_normalize_scalar_text(review.next_query_strategy) or 'n/a'}",
        f"- Stop: {bool(review.stop)}",
        "- Findings:",
    ]
    lines.extend(
        _render_markdown_bullets(review.findings, indent="  ") or ["  - (none)"]
    )
    lines.append("- Covered subthemes:")
    lines.extend(
        _render_markdown_bullets(review.covered_subthemes, indent="  ")
        or ["  - (none)"]
    )
    lines.append("- Critical gaps:")
    lines.extend(
        _render_markdown_bullets(review.critical_gaps, indent="  ") or ["  - (none)"]
    )
    lines.append("- Need content source IDs:")
    if review.need_content_source_ids:
        lines.extend(
            f"  - {int(source_id)}" for source_id in review.need_content_source_ids
        )
    else:
        lines.append("  - (none)")
    lines.append("- Conflict arbitration:")
    if review.conflict_arbitration:
        for item in review.conflict_arbitration:
            topic = _normalize_scalar_text(item.topic) or "n/a"
            status = _normalize_scalar_text(item.status) or "n/a"
            lines.append(f"  - topic={topic}; status={status}")
    else:
        lines.append("  - (none)")
    lines.append("- Next queries:")
    lines.extend(
        _render_markdown_bullets(review.next_queries, indent="  ") or ["  - (none)"]
    )
    lines.append("- Covered entities:")
    lines.extend(
        _render_markdown_bullets(review.covered_entities, indent="  ") or ["  - (none)"]
    )
    lines.append("- Missing entities:")
    lines.extend(
        _render_markdown_bullets(review.missing_entities, indent="  ") or ["  - (none)"]
    )
    return "\n".join(lines).strip()


def render_queries_markdown(queries: list[str]) -> str:
    lines = _render_markdown_bullets(queries)
    if not lines:
        return "- (none)"
    return "\n".join(lines).strip()


def render_question_cards_markdown(cards: list[ResearchQuestionCard]) -> str:
    if not cards:
        return "- (none)"
    lines: list[str] = []
    for index, card in enumerate(cards, start=1):
        lines.extend(_render_theme_plan_card_lines(card=card, index=index, indent=""))
    return "\n".join(lines).strip()


def render_architect_plan_markdown(plan: RenderArchitectOutput) -> str:
    lines: list[str] = [
        "### Architect Plan",
        f"- Report objective: {_normalize_scalar_text(plan.report_objective) or 'n/a'}",
        "- Sections:",
    ]
    if not plan.sections:
        lines.append("  - (none)")
        return "\n".join(lines).strip()
    for index, section in enumerate(plan.sections, start=1):
        lines.append(
            f"  - Section {index}: "
            f"{_normalize_scalar_text(section.subhead) or _normalize_scalar_text(section.section_id) or 'Section'}"
        )
        lines.extend(
            [
                f"    - section_id: {_normalize_scalar_text(section.section_id) or 'n/a'}",
                f"    - section_role: {_normalize_scalar_text(section.section_role) or 'n/a'}",
                f"    - angle: {_normalize_scalar_text(section.angle) or 'n/a'}",
                (
                    "    - progression_hint: "
                    f"{_normalize_scalar_text(section.progression_hint) or 'n/a'}"
                ),
                "    - question_ids:",
            ]
        )
        lines.extend(
            _render_markdown_bullets(section.question_ids, indent="      ")
            or ["      - (none)"]
        )
        lines.extend(
            [
                "    - scope_requirements:",
            ]
        )
        lines.extend(
            _render_markdown_bullets(section.scope_requirements, indent="      ")
            or ["      - (none)"]
        )
        lines.append("    - writing_boundaries:")
        lines.extend(
            _render_markdown_bullets(section.writing_boundaries, indent="      ")
            or ["      - (none)"]
        )
        lines.append("    - must_cover_points:")
        lines.extend(
            _render_markdown_bullets(section.must_cover_points, indent="      ")
            or ["      - (none)"]
        )
    return "\n".join(lines).strip()


def render_section_plan_markdown(section: RenderArchitectSectionPlan) -> str:
    lines: list[str] = [
        "### Current Section Plan",
        f"- section_id: {_normalize_scalar_text(section.section_id) or 'n/a'}",
        f"- subhead: {_normalize_scalar_text(section.subhead) or 'n/a'}",
        f"- section_role: {_normalize_scalar_text(section.section_role) or 'n/a'}",
        f"- angle: {_normalize_scalar_text(section.angle) or 'n/a'}",
        f"- progression_hint: {_normalize_scalar_text(section.progression_hint) or 'n/a'}",
        "- question_ids:",
    ]
    lines.extend(
        _render_markdown_bullets(section.question_ids, indent="  ") or ["  - (none)"]
    )
    lines.extend(
        [
            "- scope_requirements:",
        ]
    )
    lines.extend(
        _render_markdown_bullets(section.scope_requirements, indent="  ")
        or ["  - (none)"]
    )
    lines.append("- writing_boundaries:")
    lines.extend(
        _render_markdown_bullets(section.writing_boundaries, indent="  ")
        or ["  - (none)"]
    )
    lines.append("- must_cover_points:")
    lines.extend(
        _render_markdown_bullets(section.must_cover_points, indent="  ")
        or ["  - (none)"]
    )
    return "\n".join(lines).strip()


__all__ = [
    "normalize_block_text",
    "render_overview_review_markdown",
    "render_architect_plan_markdown",
    "render_question_cards_markdown",
    "render_queries_markdown",
    "render_rounds_markdown",
    "render_section_plan_markdown",
    "render_theme_plan_markdown",
]
