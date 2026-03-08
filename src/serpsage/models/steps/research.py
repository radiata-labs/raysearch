from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from serpsage.models.app.request import ResearchRequest
from serpsage.models.app.response import ResearchResponse
from serpsage.models.base import MutableModel
from serpsage.models.components.extract import ExtractRef
from serpsage.models.steps.base import BaseStepContext
from serpsage.models.steps.search import SearchFetchedCandidate
from serpsage.settings.models import AppSettings

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
OverviewConflictStatus = Literal["resolved", "unresolved"]
ContentConflictStatus = Literal["resolved", "unresolved", "insufficient", "closed"]
NonEmptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
LooseText = Annotated[str, StringConstraints(strip_whitespace=True)]


class ThemeQuestionCardPayload(MutableModel):
    question: NonEmptyText
    priority: int = Field(ge=1, le=5)
    seed_queries: list[NonEmptyText] = Field(min_length=1, max_length=8)
    evidence_focus: list[NonEmptyText] = Field(default_factory=list, max_length=8)
    expected_gain: NonEmptyText


class ThemeOutputPayload(MutableModel):
    detected_input_language: LanguageCode
    search_language: LanguageCode
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
    seed_queries: list[NonEmptyText] = Field(min_length=1)
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
    search_language: LanguageCode = "other"
    output_language: LanguageCode = "other"
    question_cards: list[ResearchThemePlanCard] = Field(
        default_factory=list, max_length=24
    )


class PlanSearchJobPayload(MutableModel):
    query: NonEmptyText
    intent: SearchJobIntent
    mode: SearchJobMode
    include_domains: list[NonEmptyText] = Field(max_length=12)
    exclude_domains: list[NonEmptyText] = Field(max_length=12)
    include_text: list[NonEmptyText] = Field(max_length=8)
    exclude_text: list[NonEmptyText] = Field(max_length=8)
    additional_queries: list[NonEmptyText] = Field(max_length=8)


class PlanOutputPayload(MutableModel):
    query_strategy: NonEmptyText
    round_action: RoundAction
    explore_target_source_ids: list[int] = Field(max_length=12)
    search_jobs: list[PlanSearchJobPayload] = Field(max_length=8)


class OverviewConflictPayload(MutableModel):
    topic: NonEmptyText
    status: OverviewConflictStatus


class OverviewOutputPayload(MutableModel):
    findings: list[NonEmptyText] = Field(max_length=20)
    conflict_arbitration: list[OverviewConflictPayload] = Field(max_length=16)
    covered_subthemes: list[NonEmptyText] = Field(max_length=16)
    entity_coverage_complete: bool
    covered_entities: list[NonEmptyText] = Field(max_length=24)
    missing_entities: list[NonEmptyText] = Field(max_length=24)
    critical_gaps: list[NonEmptyText] = Field(max_length=12)
    confidence: float = Field(ge=-1.0, le=1.0)
    need_content_source_ids: list[int] = Field(max_length=20)
    next_query_strategy: NonEmptyText
    next_queries: list[NonEmptyText] = Field(max_length=8)
    stop: bool


class ContentConflictPayload(MutableModel):
    topic: NonEmptyText
    status: ContentConflictStatus


class ContentOutputPayload(MutableModel):
    resolved_findings: list[NonEmptyText] = Field(max_length=20)
    conflict_resolutions: list[ContentConflictPayload] = Field(max_length=16)
    entity_coverage_complete: bool
    covered_entities: list[NonEmptyText] = Field(max_length=24)
    missing_entities: list[NonEmptyText] = Field(max_length=24)
    remaining_gaps: list[NonEmptyText] = Field(max_length=12)
    confidence_adjustment: float = Field(ge=-1.0, le=1.0)
    next_query_strategy: NonEmptyText
    next_queries: list[NonEmptyText] = Field(max_length=8)
    stop: bool


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
    high_yield_remaining: bool
    next_queries: list[NonEmptyText] = Field(max_length=8)
    reason: LooseText = ""


class ResearchLinkPickerPayload(MutableModel):
    selected_link_ids: list[int] = Field(default_factory=list, max_length=24)
    reason: LooseText = ""


class ResearchTrackAllocation(MutableModel):
    search_grant: int
    fetch_grant: int
    max_queries_per_round: int
    bonus: bool = False
    fetch_only: bool = False


class ResearchBudgetReservationState(MutableModel):
    search_reserved: int = 0
    fetch_reserved: int = 0


class ResearchOrchestratorState(MutableModel):
    last_global_search_used: int = -1
    priorities: dict[str, float] = Field(default_factory=dict)
    query_width_hints: dict[str, int] = Field(default_factory=dict)
    rationale: LooseText = ""
    refresh_interval_search_calls: int = 2


class ResearchTrackOrchestratorPriorityPayload(MutableModel):
    question_id: NonEmptyText
    priority_score: float = Field(default=0.5, ge=0.0, le=1.0)
    query_width_hint: int = Field(default=1, ge=1, le=2)
    reason: LooseText = ""


class ResearchTrackOrchestratorOutputPayload(MutableModel):
    priorities: list[ResearchTrackOrchestratorPriorityPayload] = Field(max_length=24)
    rationale: LooseText = ""


class ResearchCorpusUpsertResult(MutableModel):
    source_id: int
    canonical_url: str
    is_new_canonical: bool
    is_new_version: bool


class ResearchWriterSectionFailure(MutableModel):
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


class ResearchSource(MutableModel):
    source_id: int
    url: str
    canonical_url: str = ""
    title: str = ""
    overview: str = ""
    content: str = ""
    round_index: int = 0
    is_subpage: bool = False
    seen_count: int = 1
    content_fingerprint: str = ""


class ResearchSearchJob(MutableModel):
    query: NonEmptyText
    intent: SearchJobIntent = "coverage"
    mode: SearchJobMode = "auto"
    additional_queries: list[NonEmptyText] = Field(default_factory=list)
    include_domains: list[NonEmptyText] = Field(default_factory=list)
    exclude_domains: list[NonEmptyText] = Field(default_factory=list)
    include_text: list[NonEmptyText] = Field(default_factory=list)
    exclude_text: list[NonEmptyText] = Field(default_factory=list)


class ResearchQuestionCard(MutableModel):
    question_id: NonEmptyText
    question: NonEmptyText
    priority: int = Field(default=3, ge=1, le=5)
    seed_queries: list[NonEmptyText] = Field(default_factory=list)
    evidence_focus: list[NonEmptyText] = Field(default_factory=list)
    expected_gain: str = ""


class ResearchTrackResult(MutableModel):
    question_id: str
    question: str
    stop_reason: str = ""
    rounds: int = 0
    search_calls: int = 0
    fetch_calls: int = 0
    confidence: float = 0.0
    coverage_ratio: float = 0.0
    unresolved_conflicts: int = 0
    subreport_markdown: str = ""
    track_insight_card: TrackInsightCardPayload | None = None
    key_findings: list[str] = Field(default_factory=list)


class ResearchLimits(MutableModel):
    mode_key: Literal["research-fast", "research", "research-pro"] = "research"
    max_rounds: int = 1
    max_search_calls: int = 1
    max_fetch_calls: int = 1
    max_results_per_search: int = 1
    max_queries_per_round: int = 1
    stop_confidence: float = 0.80
    min_coverage_ratio: float = 0.80
    max_question_cards_effective: int = 4
    min_rounds_per_track: int = 2
    source_topk: int = 20
    source_chars: int = 180_000
    content_chars: int = 10_000
    explore_target_pages_per_round: int = 3
    explore_links_per_page: int = 8


class ResearchLinkCandidate(MutableModel):
    source_id: int
    url: str = ""
    title: str = ""
    links: list[ExtractRef] = Field(default_factory=list)
    subpage_links: list[list[ExtractRef]] = Field(default_factory=list)
    subpage_urls: list[str] = Field(default_factory=list)
    round_index: int = 0


class ResearchTask(MutableModel):
    question: str = ""
    style: ReportStyle = "explainer"
    intent: TaskIntent = "other"
    complexity: TaskComplexity = "medium"
    input_language: LanguageCode = "other"
    search_language: LanguageCode = "other"
    output_language: LanguageCode = "other"
    subthemes: list[NonEmptyText] = Field(default_factory=list, max_length=12)
    entities: list[NonEmptyText] = Field(default_factory=list, max_length=16)
    cards: list[ResearchQuestionCard] = Field(default_factory=list, max_length=24)


class ResearchKnowledge(MutableModel):
    sources: list[ResearchSource] = Field(default_factory=list)
    source_ids_by_url: dict[str, list[int]] = Field(default_factory=dict)
    ranked_source_ids: list[int] = Field(default_factory=list)
    source_scores: dict[int, float] = Field(default_factory=dict)
    covered_subthemes: list[str] = Field(default_factory=list)


class ResearchRound(MutableModel):
    round_index: int = 0
    round_action: RoundAction = "search"
    query_strategy: str = ""
    queries: list[str] = Field(default_factory=list)
    search_jobs: list[ResearchSearchJob] = Field(default_factory=list)
    explore_target_source_ids: list[int] = Field(default_factory=list)
    search_fetched_candidates: list[SearchFetchedCandidate] = Field(
        default_factory=list
    )
    overview_review: OverviewOutputPayload | None = None
    content_review: ContentOutputPayload | None = None
    need_content_source_ids: list[int] = Field(default_factory=list)
    next_queries: list[str] = Field(default_factory=list)
    result_count: int = 0
    new_source_ids: list[int] = Field(default_factory=list)
    context_source_ids: list[int] = Field(default_factory=list)
    corpus_score_gain: float = 0.0
    overview_summary: str = ""
    content_summary: str = ""
    confidence: float = 0.0
    coverage_ratio: float = 0.0
    entity_coverage_complete: bool = False
    missing_entities: list[str] = Field(default_factory=list)
    unresolved_conflicts: int = 0
    unresolved_conflict_topics: list[str] = Field(default_factory=list)
    critical_gaps: int = 0
    stop_ready: bool = False
    remaining_objectives: list[str] = Field(default_factory=list)
    low_gain_streak: int = 0
    stop_reason: str = ""
    stop: bool = False


class ResearchRun(MutableModel):
    mode: Literal["research-fast", "research", "research-pro"] = "research"
    limits: ResearchLimits = Field(default_factory=ResearchLimits)
    search_calls: int = 0
    fetch_calls: int = 0
    provider_language_param_applied: bool = False
    explore_resolved_relative_links: int = 0
    stop: bool = False
    stop_reason: str = ""
    round_index: int = 0
    next_queries: list[str] = Field(default_factory=list)
    link_candidates: list[ResearchLinkCandidate] = Field(default_factory=list)
    link_candidates_round: int = 0
    notes: list[str] = Field(default_factory=list)
    current: ResearchRound | None = None
    history: list[ResearchRound] = Field(default_factory=list)
    global_search_budget: int = 0
    global_fetch_budget: int = 0
    global_search_used: int = 0
    global_fetch_used: int = 0


class ResearchResult(MutableModel):
    content: str = ""
    structured: object | None = None
    tracks: list[ResearchTrackResult] = Field(default_factory=list)


class ResearchStepContext(BaseStepContext[ResearchRequest, ResearchResponse]):
    settings: AppSettings
    request: ResearchRequest
    response: ResearchResponse
    task: ResearchTask = Field(default_factory=ResearchTask)
    run: ResearchRun = Field(default_factory=ResearchRun)
    knowledge: ResearchKnowledge = Field(default_factory=ResearchKnowledge)
    result: ResearchResult = Field(default_factory=ResearchResult)


__all__ = [
    "ContentConflictPayload",
    "ContentConflictStatus",
    "ContentOutputPayload",
    "LanguageCode",
    "OverviewConflictPayload",
    "OverviewConflictStatus",
    "OverviewOutputPayload",
    "PlanOutputPayload",
    "PlanSearchJobPayload",
    "RenderArchitectOutput",
    "RenderArchitectSectionPlan",
    "ReportStyle",
    "ResearchBudgetReservationState",
    "ResearchCorpusUpsertResult",
    "ResearchDecideSignalPayload",
    "ResearchKnowledge",
    "ResearchLinkCandidate",
    "ResearchLinkPickerPayload",
    "ResearchLimits",
    "ResearchOrchestratorState",
    "ResearchQuestionCard",
    "ResearchResult",
    "ResearchRound",
    "ResearchRun",
    "ResearchSearchJob",
    "ResearchSource",
    "ResearchStepContext",
    "ResearchTask",
    "ResearchThemePlan",
    "ResearchThemePlanCard",
    "ResearchTrackAllocation",
    "ResearchTrackOrchestratorOutputPayload",
    "ResearchTrackOrchestratorPriorityPayload",
    "ResearchTrackResult",
    "ResearchWriterSectionFailure",
    "RoundAction",
    "SearchJobIntent",
    "SearchJobMode",
    "SubreportOutputPayload",
    "TaskComplexity",
    "TaskIntent",
    "ThemeOutputPayload",
    "ThemeQuestionCardPayload",
    "TrackInsightCardPayload",
    "TrackInsightPointPayload",
]
