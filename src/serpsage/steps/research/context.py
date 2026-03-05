from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.models.pipeline import (
    ResearchLinkCandidate,
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

if TYPE_CHECKING:
    from collections.abc import Iterable

_NONE_BULLET = ["  - (none)"]
_NONE_BULLET_L3 = ["      - (none)"]


def normalize_block_text(text: str) -> str:
    return text.replace("\r\n", "\n").replace("\r", "\n").strip()


def _normalize_scalar_text(value: str) -> str:
    return value


def _render_markdown_bullets(values: Iterable[str], *, indent: str = "") -> list[str]:
    out: list[str] = []
    for item in values:
        token = normalize_block_text(item)
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
        level = max(1, title_level)
        lines.append(f"{'#' * level} Theme Plan")
    lines.extend(
        [
            f"- Core question: {_normalize_scalar_text(plan.core_question) or 'n/a'}",
            f"- Report style: {_normalize_scalar_text(plan.report_style) or 'n/a'}",
            f"- Task intent: {_normalize_scalar_text(plan.task_intent) or 'n/a'}",
            f"- Complexity tier: {_normalize_scalar_text(plan.complexity_tier) or 'n/a'}",
            f"- Question card count: {len(plan.question_cards)}",
            f"- Input language: {_normalize_scalar_text(plan.input_language) or 'n/a'}",
            f"- Search language: {_normalize_scalar_text(plan.search_language) or 'n/a'}",
            f"- Output language: {_normalize_scalar_text(plan.output_language) or 'n/a'}",
        ]
    )
    lines.append("- Subthemes:")
    lines.extend(_render_markdown_bullets(plan.subthemes, indent="  ") or _NONE_BULLET)
    lines.append("- Required entities:")
    lines.extend(
        _render_markdown_bullets(plan.required_entities, indent="  ") or _NONE_BULLET
    )
    if include_question_cards:
        lines.append("- Question cards:")
        if plan.question_cards:
            for index, card in enumerate(plan.question_cards, start=1):
                lines.extend(
                    _render_theme_plan_card_lines(card=card, index=index, indent="  ")
                )
        else:
            lines.extend(_NONE_BULLET)
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
            f"priority={getattr(card, 'priority', 0) or 0})"
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
    selected = rounds[-max(1, limit) :]
    if not selected:
        return "- (none)"
    lines: list[str] = []
    for round_state in selected:
        lines.extend(
            [
                f"### Round {round_state.round_index}",
                (
                    f"- Query strategy: "
                    f"{_normalize_scalar_text(round_state.query_strategy) or 'n/a'}"
                ),
                f"- Result count: {round_state.result_count}",
                f"- Confidence: {float(round_state.confidence):.3f}",
                f"- Coverage ratio: {float(round_state.coverage_ratio):.3f}",
                f"- Entity coverage complete: {round_state.entity_coverage_complete}",
                f"- Unresolved conflicts: {round_state.unresolved_conflicts}",
                f"- Critical gaps: {round_state.critical_gaps}",
                f"- Stop: {round_state.stop}",
                f"- Stop reason: {_normalize_scalar_text(round_state.stop_reason) or 'n/a'}",
                "- Queries:",
            ]
        )
        lines.extend(
            _render_markdown_bullets(round_state.queries, indent="  ") or _NONE_BULLET
        )
        overview_summary = normalize_block_text(round_state.overview_summary)
        content_summary = normalize_block_text(round_state.content_summary)
        lines.append("- Overview summary:")
        if overview_summary:
            lines.extend(
                ["  ```text"]
                + [f"  {line}" for line in overview_summary.split("\n")]
                + ["  ```"]
            )
        else:
            lines.extend(_NONE_BULLET)
        lines.append("- Content summary:")
        if content_summary:
            lines.extend(
                ["  ```text"]
                + [f"  {line}" for line in content_summary.split("\n")]
                + ["  ```"]
            )
        else:
            lines.extend(_NONE_BULLET)
        lines.append("- Missing entities:")
        lines.extend(
            _render_markdown_bullets(round_state.missing_entities, indent="  ")
            or _NONE_BULLET
        )
    return "\n".join(lines).strip()


def render_overview_review_markdown(review: OverviewOutputPayload) -> str:
    lines: list[str] = [
        "### Overview Review",
        f"- Confidence: {float(review.confidence):.3f}",
        f"- Entity coverage complete: {review.entity_coverage_complete}",
        f"- Next query strategy: {_normalize_scalar_text(review.next_query_strategy) or 'n/a'}",
        f"- Stop: {review.stop}",
        "- Findings:",
    ]
    lines.extend(_render_markdown_bullets(review.findings, indent="  ") or _NONE_BULLET)
    lines.append("- Covered subthemes:")
    lines.extend(
        _render_markdown_bullets(review.covered_subthemes, indent="  ") or _NONE_BULLET
    )
    lines.append("- Critical gaps:")
    lines.extend(
        _render_markdown_bullets(review.critical_gaps, indent="  ") or _NONE_BULLET
    )
    lines.append("- Need content source IDs:")
    if review.need_content_source_ids:
        lines.extend(f"  - {source_id}" for source_id in review.need_content_source_ids)
    else:
        lines.extend(_NONE_BULLET)
    lines.append("- Conflict arbitration:")
    if review.conflict_arbitration:
        for item in review.conflict_arbitration:
            topic = _normalize_scalar_text(item.topic) or "n/a"
            status = _normalize_scalar_text(item.status) or "n/a"
            lines.append(f"  - topic={topic}; status={status}")
    else:
        lines.extend(_NONE_BULLET)
    lines.append("- Next queries:")
    lines.extend(
        _render_markdown_bullets(review.next_queries, indent="  ") or _NONE_BULLET
    )
    lines.append("- Covered entities:")
    lines.extend(
        _render_markdown_bullets(review.covered_entities, indent="  ") or _NONE_BULLET
    )
    lines.append("- Missing entities:")
    lines.extend(
        _render_markdown_bullets(review.missing_entities, indent="  ") or _NONE_BULLET
    )
    return "\n".join(lines).strip()


def render_queries_markdown(queries: list[str]) -> str:
    lines = _render_markdown_bullets(queries)
    if not lines:
        return "- (none)"
    return "\n".join(lines).strip()


def render_link_candidates_markdown(
    candidates: list[ResearchLinkCandidate],
    *,
    max_pages: int = 8,
    max_links_per_page: int = 6,
) -> str:
    if not candidates:
        return _NONE_BULLET[0]
    page_limit = max(1, max_pages)
    link_limit = max(1, max_links_per_page)
    lines: list[str] = []
    for item in list(candidates)[:page_limit]:
        main_links = list(item.links or [])
        flat_subpage_links = [
            link
            for group in list(item.subpage_links or [])
            for link in list(group or [])
        ]
        lines.extend(
            [
                f"### Source {item.source_id}",
                f"- URL: {_normalize_scalar_text(item.url) or 'n/a'}",
                f"- Title: {_normalize_scalar_text(item.title) or 'n/a'}",
                f"- Round index: {item.round_index}",
                f"- Main links count: {len(main_links)}",
                f"- Subpage links count: {len(flat_subpage_links)}",
                "- Link samples:",
            ]
        )
        sample_lines: list[str] = [
            (
                "[main] "
                f"{_normalize_scalar_text(link.anchor_text) or '(no anchor)'} -> "
                f"{_normalize_scalar_text(link.url) or 'n/a'}"
            )
            for link in main_links[:link_limit]
        ]
        sample_lines.extend(
            (
                "[subpage] "
                f"{_normalize_scalar_text(link.anchor_text) or '(no anchor)'} -> "
                f"{_normalize_scalar_text(link.url) or 'n/a'}"
            )
            for link in flat_subpage_links[:link_limit]
        )
        lines.extend(
            _render_markdown_bullets(sample_lines, indent="  ") or _NONE_BULLET
        )
    return "\n".join(lines).strip()


def render_question_cards_markdown(cards: list[ResearchQuestionCard]) -> str:
    if not cards:
        return _NONE_BULLET[0]
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
        lines.extend(_NONE_BULLET)
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
            or _NONE_BULLET_L3
        )
        lines.extend(
            [
                "    - scope_requirements:",
            ]
        )
        lines.extend(
            _render_markdown_bullets(section.scope_requirements, indent="      ")
            or _NONE_BULLET_L3
        )
        lines.append("    - writing_boundaries:")
        lines.extend(
            _render_markdown_bullets(section.writing_boundaries, indent="      ")
            or _NONE_BULLET_L3
        )
        lines.append("    - must_cover_points:")
        lines.extend(
            _render_markdown_bullets(section.must_cover_points, indent="      ")
            or _NONE_BULLET_L3
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
        _render_markdown_bullets(section.question_ids, indent="  ") or _NONE_BULLET
    )
    lines.extend(
        [
            "- scope_requirements:",
        ]
    )
    lines.extend(
        _render_markdown_bullets(section.scope_requirements, indent="  ")
        or _NONE_BULLET
    )
    lines.append("- writing_boundaries:")
    lines.extend(
        _render_markdown_bullets(section.writing_boundaries, indent="  ")
        or _NONE_BULLET
    )
    lines.append("- must_cover_points:")
    lines.extend(
        _render_markdown_bullets(section.must_cover_points, indent="  ") or _NONE_BULLET
    )
    return "\n".join(lines).strip()


__all__ = [
    "normalize_block_text",
    "render_overview_review_markdown",
    "render_architect_plan_markdown",
    "render_link_candidates_markdown",
    "render_question_cards_markdown",
    "render_queries_markdown",
    "render_rounds_markdown",
    "render_section_plan_markdown",
    "render_theme_plan_markdown",
]
