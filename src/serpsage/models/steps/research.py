from __future__ import annotations

from typing import Annotated, Literal

from pydantic import Field, StringConstraints

from serpsage.models.app.request import ResearchRequest
from serpsage.models.app.response import ResearchResponse
from serpsage.models.base import MutableModel
from serpsage.models.components.extract import ExtractRef
from serpsage.models.steps.base import BaseStepContext
from serpsage.models.steps.search import QuerySourceSpec, SearchFetchedCandidate

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
    next_queries: list[QuerySourceSpec] = Field(max_length=8)


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
    next_queries: list[QuerySourceSpec] = Field(max_length=8)


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


class ResearchLinkPickerPayload(MutableModel):
    selected_link_ids: list[int] = Field(default_factory=list, max_length=24)


class ResearchBudgetTierState(MutableModel):
    search_total: int = 0
    fetch_total: int = 0
    search_used: int = 0
    fetch_used: int = 0


class ResearchBudgetLedger(MutableModel):
    original_search_budget: int = 0
    original_fetch_budget: int = 0
    base_budget: ResearchBudgetTierState = Field(
        default_factory=ResearchBudgetTierState
    )
    restored_budget: ResearchBudgetTierState = Field(
        default_factory=ResearchBudgetTierState
    )
    extension_budget: ResearchBudgetTierState = Field(
        default_factory=ResearchBudgetTierState
    )
    restore_used: bool = False
    extension_used: bool = False
    extension_multiplier: float = 0.0


class ResearchTrackRuntime(MutableModel):
    question_id: str = ""
    state: ResearchTrackState = "active"
    current_budget_tier: ResearchBudgetTier = "base"
    waiting_reason: str = ""
    waiting_rounds: int = 0
    allocated_search_calls: int = 0
    allocated_fetch_calls: int = 0
    used_search_calls: int = 0
    used_fetch_calls: int = 0
    reclaimed_search_calls: int = 0
    reclaimed_fetch_calls: int = 0
    completed_rounds: int = 0
    stop_reason: str = ""
    last_budget_event: str = ""


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
    admitted: bool = True
    used_in_report: bool = False
    used_in_final_pass: bool = False


class ResearchSearchJob(MutableModel):
    query: QuerySourceSpec
    intent: SearchJobIntent = "coverage"
    mode: SearchJobMode = "auto"
    additional_queries: list[QuerySourceSpec] = Field(default_factory=list)


class ResearchQuestionCard(MutableModel):
    question_id: NonEmptyText
    question: NonEmptyText
    priority: int = Field(default=3, ge=1, le=5)
    seed_queries: list[QuerySourceSpec] = Field(default_factory=list)
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
    budget_tier: ResearchBudgetTier = "base"
    waiting_rounds: int = 0


class ResearchLimits(MutableModel):
    mode_key: Literal["research-fast", "research", "research-pro"] = "research"
    max_rounds: int = 1
    max_search_calls: int = 1
    max_fetch_calls: int = 1
    max_results_per_search: int = 1
    max_queries_per_round: int = 1
    max_question_cards_effective: int = 4
    min_rounds_per_track: int = 2
    round_search_budget: int = 2
    round_fetch_budget: int = 6
    review_source_window: int = 48
    report_source_batch_size: int = 8
    report_source_batch_chars: int = 48_000
    fetch_page_max_chars: int = 10_000
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
    admitted_source_ids: list[int] = Field(default_factory=list)
    pending_source_ids: list[int] = Field(default_factory=list)
    report_used_source_ids: list[int] = Field(default_factory=list)


class ResearchRound(MutableModel):
    round_index: int = 0
    round_action: RoundAction = "search"
    query_strategy: str = ""
    queries: list[QuerySourceSpec] = Field(default_factory=list)
    search_jobs: list[ResearchSearchJob] = Field(default_factory=list)
    explore_target_source_ids: list[int] = Field(default_factory=list)
    search_fetched_candidates: list[SearchFetchedCandidate] = Field(
        default_factory=list
    )
    pending_search_jobs: list[ResearchSearchJob] = Field(default_factory=list)
    overview_review: OverviewOutputPayload | None = None
    content_review: ContentOutputPayload | None = None
    need_content_source_ids: list[int] = Field(default_factory=list)
    next_queries: list[QuerySourceSpec] = Field(default_factory=list)
    result_count: int = 0
    new_source_ids: list[int] = Field(default_factory=list)
    fetched_source_ids: list[int] = Field(default_factory=list)
    admitted_source_ids: list[int] = Field(default_factory=list)
    deferred_candidate_urls: list[str] = Field(default_factory=list)
    remaining_source_ids: list[int] = Field(default_factory=list)
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
    remaining_objectives: list[str] = Field(default_factory=list)
    waiting_for_budget: bool = False
    waiting_reason: str = ""
    budget_tier_applied: ResearchBudgetTier = "base"
    stop_reason: str = ""
    stop: bool = False


class ResearchRun(MutableModel):
    mode: Literal["research-fast", "research", "research-pro"] = "research"
    limits: ResearchLimits = Field(default_factory=ResearchLimits)
    search_calls: int = 0
    fetch_calls: int = 0
    explore_resolved_relative_links: int = 0
    stop: bool = False
    stop_reason: str = ""
    round_index: int = 0
    next_queries: list[QuerySourceSpec] = Field(default_factory=list)
    link_candidates: list[ResearchLinkCandidate] = Field(default_factory=list)
    link_candidates_round: int = 0
    notes: list[str] = Field(default_factory=list)
    current: ResearchRound | None = None
    history: list[ResearchRound] = Field(default_factory=list)
    global_search_budget: int = 0
    global_fetch_budget: int = 0
    global_search_used: int = 0
    global_fetch_used: int = 0
    budget_ledger: ResearchBudgetLedger = Field(default_factory=ResearchBudgetLedger)
    track_runtime: ResearchTrackRuntime | None = None
    track_runtimes: dict[str, ResearchTrackRuntime] = Field(default_factory=dict)
    restored_budget_applied: bool = False
    extension_budget_applied: bool = False
    budget_events: list[str] = Field(default_factory=list)


class ResearchResult(MutableModel):
    content: str = ""
    structured: object | None = None
    tracks: list[ResearchTrackResult] = Field(default_factory=list)


class ResearchStepContext(BaseStepContext[ResearchRequest, ResearchResponse]):
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
    "ResearchBudgetLedger",
    "ResearchBudgetTier",
    "ResearchBudgetTierState",
    "ResearchCorpusUpsertResult",
    "ResearchDecideSignalPayload",
    "ResearchKnowledge",
    "ResearchLinkCandidate",
    "ResearchLinkPickerPayload",
    "ResearchLimits",
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
    "ResearchTrackResult",
    "ResearchTrackRuntime",
    "ResearchTrackState",
    "ResearchWriterSectionFailure",
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
