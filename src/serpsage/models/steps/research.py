from __future__ import annotations

from typing import Literal

from pydantic import Field

from serpsage.models.app.request import (
    ResearchRequest,
)
from serpsage.models.app.response import ResearchResponse
from serpsage.models.base import MutableModel, UnvalidatedModel
from serpsage.models.components.extract import (
    ExtractRef,
)
from serpsage.models.steps.base import BaseStepContext
from serpsage.models.steps.search import SearchFetchedCandidate
from serpsage.settings.models import AppSettings

ReportStyle = Literal["decision", "explainer", "execution"]
RoundAction = Literal["search", "explore"]
TaskIntent = Literal["how_to", "comparison", "explainer", "diagnosis", "other"]
TaskComplexity = Literal["low", "medium", "high"]


class ThemeQuestionCardPayload(UnvalidatedModel):
    question: str
    priority: int = Field(default=3, ge=1, le=5)
    seed_queries: list[str] = Field(default_factory=list, max_length=8)
    evidence_focus: list[str] = Field(default_factory=list, max_length=8)
    expected_gain: str = ""


class ThemeOutputPayload(UnvalidatedModel):
    detected_input_language: str = "other"
    search_language: str = "other"
    core_question: str = ""
    report_style: ReportStyle = "explainer"
    task_intent: TaskIntent = "other"
    complexity_tier: TaskComplexity = "medium"
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
    report_style: ReportStyle = "explainer"
    task_intent: TaskIntent = "other"
    complexity_tier: TaskComplexity = "medium"
    subthemes: list[str] = Field(default_factory=list, max_length=12)
    required_entities: list[str] = Field(default_factory=list, max_length=16)
    input_language: str = ""
    search_language: str = ""
    output_language: str = ""
    question_cards: list[ResearchThemePlanCard] = Field(
        default_factory=list,
        max_length=24,
    )


class PlanSearchJobPayload(UnvalidatedModel):
    query: str
    intent: str = "coverage"
    mode: str = "auto"
    include_domains: list[str] = Field(default_factory=list, max_length=12)
    exclude_domains: list[str] = Field(default_factory=list, max_length=12)
    include_text: list[str] = Field(default_factory=list, max_length=8)
    exclude_text: list[str] = Field(default_factory=list, max_length=8)
    additional_queries: list[str] = Field(default_factory=list, max_length=8)


class PlanOutputPayload(UnvalidatedModel):
    query_strategy: str = "mixed"
    round_action: RoundAction = "search"
    explore_target_source_ids: list[int] = Field(default_factory=list, max_length=12)
    search_jobs: list[PlanSearchJobPayload] = Field(default_factory=list, max_length=8)


class OverviewConflictPayload(UnvalidatedModel):
    topic: str
    status: str


class OverviewOutputPayload(UnvalidatedModel):
    findings: list[str] = Field(default_factory=list, max_length=20)
    conflict_arbitration: list[OverviewConflictPayload] = Field(
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


class ContentConflictPayload(UnvalidatedModel):
    topic: str = ""
    status: str


class ContentOutputPayload(UnvalidatedModel):
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


class TrackInsightPointPayload(UnvalidatedModel):
    conclusion: str = ""
    condition: str = ""
    impact: str = ""


class TrackInsightCardPayload(UnvalidatedModel):
    direct_answer: str = ""
    high_value_points: list[TrackInsightPointPayload] = Field(
        default_factory=list,
        max_length=12,
    )
    key_tradeoffs_or_mechanisms: list[str] = Field(default_factory=list, max_length=10)
    unknowns_and_risks: list[str] = Field(default_factory=list, max_length=10)
    next_actions: list[str] = Field(default_factory=list, max_length=10)


class SubreportOutputPayload(UnvalidatedModel):
    subreport_markdown: str = ""
    track_insight_card: TrackInsightCardPayload | None = None


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


class ResearchDecideSignalPayload(MutableModel):
    continue_research: bool = False
    high_yield_remaining: bool = False
    next_queries: list[str] = Field(default_factory=list, max_length=8)
    reason: str = ""


class ResearchLinkPickerPayload(MutableModel):
    selected_link_ids: list[int] = Field(default_factory=list, max_length=24)
    reason: str = ""


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
    rationale: str = ""
    refresh_interval_search_calls: int = 2


class ResearchTrackOrchestratorPriorityPayload(MutableModel):
    question_id: str
    priority_score: float = Field(default=0.5, ge=0.0, le=1.0)
    query_width_hint: int = Field(default=1, ge=1, le=2)
    reason: str = ""


class ResearchTrackOrchestratorOutputPayload(MutableModel):
    priorities: list[ResearchTrackOrchestratorPriorityPayload] = Field(
        default_factory=list,
        max_length=24,
    )
    rationale: str = ""


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


class ResearchQueryLanguageRepairJobPayload(MutableModel):
    query: str = ""
    additional_queries: list[str] = Field(default_factory=list, max_length=8)


class ResearchQueryLanguageRepairOutputPayload(MutableModel):
    search_jobs: list[ResearchQueryLanguageRepairJobPayload] = Field(
        default_factory=list,
        max_length=8,
    )


class ResearchBudgetState(MutableModel):
    max_rounds: int = 1
    max_search_calls: int = 1
    max_fetch_calls: int = 1
    max_results_per_search: int = 1
    max_queries_per_round: int = 1
    stop_confidence: float = 0.80
    min_coverage_ratio: float = 0.80


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
    query: str
    intent: str = "coverage"
    mode: Literal["auto", "deep"] = "auto"
    additional_queries: list[str] = Field(default_factory=list)
    include_domains: list[str] = Field(default_factory=list)
    exclude_domains: list[str] = Field(default_factory=list)
    include_text: list[str] = Field(default_factory=list)
    exclude_text: list[str] = Field(default_factory=list)


class ResearchQuestionCard(MutableModel):
    question_id: str
    question: str
    priority: int = 3
    seed_queries: list[str] = Field(default_factory=list)
    evidence_focus: list[str] = Field(default_factory=list)
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


class ResearchParallelState(MutableModel):
    question_cards: list[ResearchQuestionCard] = Field(default_factory=list)
    track_results: list[ResearchTrackResult] = Field(default_factory=list)
    global_search_budget: int = 0
    global_fetch_budget: int = 0
    global_search_used: int = 0
    global_fetch_used: int = 0


class ResearchCoverageState(MutableModel):
    total_subthemes: int = 0
    covered_subthemes: list[str] = Field(default_factory=list)


class ResearchModeDepthState(MutableModel):
    mode_key: Literal["research-fast", "research", "research-pro"] = "research"
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


class ResearchRuntimeState(MutableModel):
    mode_depth: ResearchModeDepthState = Field(default_factory=ResearchModeDepthState)
    budget: ResearchBudgetState = Field(default_factory=ResearchBudgetState)
    search_calls: int = 0
    fetch_calls: int = 0
    provider_language_param_applied: bool = False
    query_language_repair_applied: bool = False
    search_language_fallback_applied: bool = False
    explore_resolved_relative_links: int = 0
    stop: bool = False
    stop_reason: str = ""
    round_index: int = 0


class ResearchPlanState(MutableModel):
    theme_plan: ResearchThemePlan = Field(default_factory=ResearchThemePlan)
    next_queries: list[str] = Field(default_factory=list)
    last_round_link_candidates: list[ResearchLinkCandidate] = Field(
        default_factory=list
    )
    last_round_link_candidates_round: int = 0


class ResearchCorpusState(MutableModel):
    sources: list[ResearchSource] = Field(default_factory=list)
    source_url_to_ids: dict[str, list[int]] = Field(default_factory=dict)
    ranked_source_ids: list[int] = Field(default_factory=list)
    source_scores: dict[int, float] = Field(default_factory=dict)
    coverage_state: ResearchCoverageState = Field(default_factory=ResearchCoverageState)


class ResearchRoundWorkState(MutableModel):
    search_jobs: list[ResearchSearchJob] = Field(default_factory=list)
    round_action: Literal["search", "explore"] = "search"
    explore_target_source_ids: list[int] = Field(default_factory=list)
    search_fetched_candidates: list[SearchFetchedCandidate] = Field(
        default_factory=list
    )
    overview_review: OverviewOutputPayload = Field(
        default_factory=OverviewOutputPayload
    )
    content_review: ContentOutputPayload = Field(default_factory=ContentOutputPayload)
    need_content_source_ids: list[int] = Field(default_factory=list)
    next_queries: list[str] = Field(default_factory=list)


class ResearchRoundState(MutableModel):
    round_index: int = 0
    query_strategy: str = ""
    queries: list[str] = Field(default_factory=list)
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


class ResearchOutputState(MutableModel):
    content: str = ""
    structured: object | None = None


class ResearchStepContext(BaseStepContext[ResearchRequest, ResearchResponse]):
    settings: AppSettings
    request: ResearchRequest
    response: ResearchResponse
    runtime: ResearchRuntimeState = Field(default_factory=ResearchRuntimeState)
    plan: ResearchPlanState = Field(default_factory=ResearchPlanState)
    parallel: ResearchParallelState = Field(default_factory=ResearchParallelState)
    corpus: ResearchCorpusState = Field(default_factory=ResearchCorpusState)
    work: ResearchRoundWorkState = Field(default_factory=ResearchRoundWorkState)
    rounds: list[ResearchRoundState] = Field(default_factory=list)
    current_round: ResearchRoundState | None = None
    notes: list[str] = Field(default_factory=list)
    output: ResearchOutputState = Field(default_factory=ResearchOutputState)


__all__ = [
    "ReportStyle",
    "RoundAction",
    "TaskComplexity",
    "TaskIntent",
    "OverviewConflictPayload",
    "OverviewOutputPayload",
    "ContentConflictPayload",
    "ContentOutputPayload",
    "PlanOutputPayload",
    "PlanSearchJobPayload",
    "RenderArchitectOutput",
    "RenderArchitectSectionPlan",
    "ResearchBudgetReservationState",
    "SubreportOutputPayload",
    "ResearchCorpusUpsertResult",
    "ResearchDecideSignalPayload",
    "TrackInsightCardPayload",
    "TrackInsightPointPayload",
    "ResearchLinkPickerPayload",
    "ResearchOrchestratorState",
    "ResearchQueryLanguageRepairJobPayload",
    "ResearchQueryLanguageRepairOutputPayload",
    "ResearchTrackAllocation",
    "ResearchTrackOrchestratorOutputPayload",
    "ResearchTrackOrchestratorPriorityPayload",
    "ResearchThemePlan",
    "ResearchThemePlanCard",
    "ResearchWriterSectionFailure",
    "ThemeOutputPayload",
    "ThemeQuestionCardPayload",
    "ResearchCorpusState",
    "ResearchBudgetState",
    "ResearchCoverageState",
    "ResearchLinkCandidate",
    "ResearchModeDepthState",
    "ResearchOutputState",
    "ResearchParallelState",
    "ResearchPlanState",
    "ResearchQuestionCard",
    "ResearchRoundState",
    "ResearchRoundWorkState",
    "ResearchRuntimeState",
    "ResearchSearchJob",
    "ResearchSource",
    "ResearchStepContext",
    "ResearchTrackResult",
]
