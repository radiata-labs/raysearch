from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from raysearch.models.base import MutableModel
from raysearch.models.steps.search import QuerySourceSpec

ReportStyle = Literal["decision", "explainer", "execution"]
RoundAction = Literal["search", "explore"]
TaskIntent = Literal["how_to", "comparison", "explainer", "diagnosis", "other"]
TaskComplexity = Literal["low", "medium", "high"]
LanguageCode = Literal[
    "zh-Hans",
    "zh-Hant",
    "en",
    "ja",
    "ko",
    "fr",
    "de",
    "es",
    "pt",
    "it",
    "ru",
    "ar",
    "hi",
    "tr",
    "other",
]
SearchJobIntent = Literal["coverage", "deepen", "verify", "refresh"]
SearchJobMode = Literal["auto", "deep"]
ConflictResolutionStatus = Literal["resolved", "unresolved", "insufficient", "closed"]
ResearchTrackState = Literal["active", "waiting_for_budget", "completed", "stopped"]
ResearchBudgetTier = Literal["base", "restored", "extension"]
SubreportUpdateAction = Literal["update", "no_update", "stop_after_update"]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
LooseText = Annotated[str, StringConstraints(strip_whitespace=True)]


class ThemeQuestionCardPayload(MutableModel):
    question: NonEmptyText
    priority: int = Field(ge=1, le=5)
    seed_queries: list[QuerySourceSpec] = Field(min_length=1, max_length=8)
    evidence_focus: list[NonEmptyText] = Field(default_factory=list, max_length=8)
    expected_gain: NonEmptyText


class ThemeOutputPayload(MutableModel):
    detected_input_language: LanguageCode
    core_question: NonEmptyText
    report_style: ReportStyle
    task_intent: TaskIntent
    complexity_tier: TaskComplexity
    subthemes: list[NonEmptyText] = Field(max_length=12)
    required_entities: list[NonEmptyText] = Field(max_length=16)
    question_cards: list[ThemeQuestionCardPayload] = Field(min_length=1, max_length=24)


class ResearchThemePlanCard(MutableModel):
    question_id: NonEmptyText
    question: NonEmptyText
    priority: int = Field(ge=1, le=5)
    seed_queries: list[QuerySourceSpec] = Field(min_length=1)
    evidence_focus: list[NonEmptyText] = Field(default_factory=list)
    expected_gain: NonEmptyText


class ResearchThemePlan(MutableModel):
    core_question: str = ""
    report_style: ReportStyle = "explainer"
    task_intent: TaskIntent = "other"
    complexity_tier: TaskComplexity = "medium"
    subthemes: list[NonEmptyText] = Field(default_factory=list, max_length=12)
    required_entities: list[NonEmptyText] = Field(default_factory=list, max_length=16)
    input_language: LanguageCode = "other"
    output_language: LanguageCode = "other"
    question_cards: list[ResearchThemePlanCard] = Field(
        default_factory=list, max_length=24
    )


class PlanSearchJobPayload(MutableModel):
    query: QuerySourceSpec
    intent: SearchJobIntent
    mode: SearchJobMode
    additional_queries: list[QuerySourceSpec] = Field(max_length=8)


class PlanOutputPayload(MutableModel):
    query_strategy: NonEmptyText
    round_action: RoundAction
    explore_target_source_ids: list[int] = Field(max_length=12)
    search_jobs: list[PlanSearchJobPayload] = Field(max_length=8)


class ConflictTopicPayload(MutableModel):
    topic: NonEmptyText
    source_ids: list[int] = Field(default_factory=list, max_length=8)


class ConflictResolutionPayload(MutableModel):
    topic: NonEmptyText
    status: ConflictResolutionStatus


class OverviewReviewPayload(MutableModel):
    findings: list[NonEmptyText] = Field(max_length=20)
    conflict_topics: list[ConflictTopicPayload] = Field(max_length=16)
    covered_subthemes: list[NonEmptyText] = Field(max_length=16)
    need_content_source_ids: list[int] = Field(max_length=20)
    missing_entities: list[NonEmptyText] = Field(max_length=24)
    confidence_score: float = Field(ge=0.0, le=1.0)
    coverage_score: float = Field(ge=0.0, le=1.0)


class ContentReviewPayload(MutableModel):
    resolved_findings: list[NonEmptyText] = Field(max_length=20)
    conflict_resolutions: list[ConflictResolutionPayload] = Field(max_length=16)
    remaining_gaps: list[NonEmptyText] = Field(max_length=12)
    confidence_score: float = Field(ge=0.0, le=1.0)
    uncertainty_score: float = Field(ge=0.0, le=1.0)


class TrackInsightPointPayload(MutableModel):
    conclusion: NonEmptyText
    condition: NonEmptyText
    impact: NonEmptyText


class TrackInsightCardPayload(MutableModel):
    direct_answer: NonEmptyText
    high_value_points: list[TrackInsightPointPayload] = Field(max_length=12)
    key_tradeoffs_or_mechanisms: list[NonEmptyText] = Field(max_length=10)
    unknowns_and_risks: list[NonEmptyText] = Field(max_length=10)
    next_actions: list[NonEmptyText] = Field(max_length=10)


class SubreportOutputPayload(MutableModel):
    subreport_markdown: NonEmptyText
    track_insight_card: TrackInsightCardPayload | None = None


class SubreportUpdatePayload(MutableModel):
    action: SubreportUpdateAction
    updated_subreport_markdown: str = ""
    updated_track_insight_card: TrackInsightCardPayload | None = None


class RenderArchitectSectionPlan(MutableModel):
    section_id: NonEmptyText
    subhead: NonEmptyText
    section_role: Literal["opening", "body", "closing"]
    question_ids: list[NonEmptyText] = Field(default_factory=list, max_length=16)
    scope_requirements: list[NonEmptyText] = Field(default_factory=list, max_length=12)
    writing_boundaries: list[NonEmptyText] = Field(default_factory=list, max_length=12)
    must_cover_points: list[NonEmptyText] = Field(default_factory=list, max_length=12)
    angle: LooseText = ""
    progression_hint: LooseText = ""


class RenderArchitectOutput(MutableModel):
    report_objective: NonEmptyText
    sections: list[RenderArchitectSectionPlan] = Field(min_length=5, max_length=10)


class ResearchDecideSignalPayload(MutableModel):
    continue_research: bool
    next_queries: list[QuerySourceSpec] = Field(max_length=8)
    reason: LooseText = ""
    information_gain_score: float = Field(ge=0.0, le=1.0)


class ResearchLinkPickerPayload(MutableModel):
    selected_link_ids: list[int] = Field(default_factory=list, max_length=24)


__all__ = [
    "ConflictResolutionPayload",
    "ConflictResolutionStatus",
    "ConflictTopicPayload",
    "ContentReviewPayload",
    "LanguageCode",
    "OverviewReviewPayload",
    "PlanOutputPayload",
    "PlanSearchJobPayload",
    "RenderArchitectOutput",
    "RenderArchitectSectionPlan",
    "ReportStyle",
    "ResearchBudgetTier",
    "ResearchDecideSignalPayload",
    "ResearchLinkPickerPayload",
    "ResearchThemePlan",
    "ResearchThemePlanCard",
    "ResearchTrackState",
    "RoundAction",
    "SearchJobIntent",
    "SearchJobMode",
    "SubreportOutputPayload",
    "SubreportUpdateAction",
    "SubreportUpdatePayload",
    "TaskComplexity",
    "TaskIntent",
    "ThemeOutputPayload",
    "ThemeQuestionCardPayload",
    "TrackInsightCardPayload",
    "TrackInsightPointPayload",
]
