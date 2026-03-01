from __future__ import annotations

from typing import Literal

from pydantic import ConfigDict, Field

from serpsage.core.model_base import MutableModel


class ThemeQuestionCardPayload(MutableModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    question: str
    priority: int = Field(default=3, ge=1, le=5)
    seed_queries: list[str] = Field(default_factory=list, max_length=8)
    evidence_focus: list[str] = Field(default_factory=list, max_length=8)
    expected_gain: str = ""


class ThemeOutputPayload(MutableModel):
    model_config = ConfigDict(extra="ignore", validate_assignment=True)

    detected_input_language: str = "same as user input language"
    core_question: str = ""
    subthemes: list[str] = Field(default_factory=list, max_length=12)
    required_entities: list[str] = Field(default_factory=list, max_length=16)
    question_cards: list[ThemeQuestionCardPayload] = Field(
        default_factory=list,
        max_length=24,
    )


class ResearchThemePlanCard(MutableModel):
    question_id: str = ""
    question: str = ""
    priority: int = Field(default=3, ge=1, le=5)
    seed_queries: list[str] = Field(default_factory=list)
    evidence_focus: list[str] = Field(default_factory=list)
    expected_gain: str = ""


class ResearchThemePlan(MutableModel):
    core_question: str = ""
    subthemes: list[str] = Field(default_factory=list, max_length=12)
    required_entities: list[str] = Field(default_factory=list, max_length=16)
    input_language: str = ""
    output_language: str = ""
    question_cards: list[ResearchThemePlanCard] = Field(
        default_factory=list,
        max_length=24,
    )


class PlanSearchJobPayload(MutableModel):
    query: str
    intent: str = "coverage"
    mode: str = "auto"
    include_domains: list[str] = Field(default_factory=list, max_length=12)
    exclude_domains: list[str] = Field(default_factory=list, max_length=12)
    include_text: list[str] = Field(default_factory=list, max_length=8)
    exclude_text: list[str] = Field(default_factory=list, max_length=8)
    additional_queries: list[str] = Field(default_factory=list, max_length=8)


class PlanOutputPayload(MutableModel):
    query_strategy: str = "mixed"
    search_jobs: list[PlanSearchJobPayload] = Field(default_factory=list, max_length=8)


class AbstractConflictPayload(MutableModel):
    topic: str
    status: str


class AbstractOutputPayload(MutableModel):
    findings: list[str] = Field(default_factory=list, max_length=20)
    conflict_arbitration: list[AbstractConflictPayload] = Field(
        default_factory=list,
        max_length=16,
    )
    covered_subthemes: list[str] = Field(default_factory=list, max_length=16)
    entity_coverage_complete: bool = False
    covered_entities: list[str] = Field(default_factory=list, max_length=24)
    missing_entities: list[str] = Field(default_factory=list, max_length=24)
    critical_gaps: list[str] = Field(default_factory=list, max_length=12)
    confidence: float = 0.0
    need_content_source_ids: list[int] = Field(default_factory=list, max_length=20)
    next_query_strategy: str = "coverage"
    next_queries: list[str] = Field(default_factory=list, max_length=8)
    stop: bool = False


class ContentConflictPayload(MutableModel):
    status: str


class ContentOutputPayload(MutableModel):
    resolved_findings: list[str] = Field(default_factory=list, max_length=20)
    conflict_resolutions: list[ContentConflictPayload] = Field(
        default_factory=list,
        max_length=16,
    )
    entity_coverage_complete: bool = False
    covered_entities: list[str] = Field(default_factory=list, max_length=24)
    missing_entities: list[str] = Field(default_factory=list, max_length=24)
    remaining_gaps: list[str] = Field(default_factory=list, max_length=12)
    confidence_adjustment: float = 0.0
    next_query_strategy: str = "coverage"
    next_queries: list[str] = Field(default_factory=list, max_length=8)
    stop: bool = False


class RenderArchitectSectionPlan(MutableModel):
    section_id: str
    subhead: str
    section_role: Literal["opening", "body", "closing"]
    question_ids: list[str] = Field(default_factory=list, max_length=16)
    scope_requirements: list[str] = Field(default_factory=list, max_length=12)
    writing_boundaries: list[str] = Field(default_factory=list, max_length=12)
    must_cover_points: list[str] = Field(default_factory=list, max_length=12)
    angle: str = ""
    progression_hint: str = ""


class RenderArchitectOutput(MutableModel):
    report_objective: str = ""
    sections: list[RenderArchitectSectionPlan] = Field(
        default_factory=list,
        min_length=5,
        max_length=10,
    )


__all__ = [
    "AbstractConflictPayload",
    "AbstractOutputPayload",
    "ContentConflictPayload",
    "ContentOutputPayload",
    "PlanOutputPayload",
    "PlanSearchJobPayload",
    "RenderArchitectOutput",
    "RenderArchitectSectionPlan",
    "ResearchThemePlan",
    "ResearchThemePlanCard",
    "ThemeOutputPayload",
    "ThemeQuestionCardPayload",
]
