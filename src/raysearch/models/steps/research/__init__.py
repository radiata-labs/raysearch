from __future__ import annotations

from typing import Literal

from pydantic import Field

from raysearch.models.app.request import ResearchRequest
from raysearch.models.app.response import ResearchResponse
from raysearch.models.base import MutableModel
from raysearch.models.steps.base import BaseStepContext
from raysearch.models.steps.research.payloads import (
    ContentReviewPayload,
    LanguageCode,
    OverviewReviewPayload,
    PlanSearchJobPayload,
    ReportStyle,
    ResearchBudgetTier,
    ResearchThemePlan,
    ResearchThemePlanCard,
    ResearchTrackState,
    RoundAction,
    TaskComplexity,
    TaskIntent,
    TrackInsightCardPayload,
)
from raysearch.models.steps.search import QuerySourceSpec, SearchFetchedCandidate


class ResearchLimits(MutableModel):
    """Research mode configuration and limits."""

    mode_key: Literal["research-fast", "research", "research-pro"] = "research"
    max_rounds: int = 1
    max_search_calls: int = 1
    max_fetch_calls: int = 1
    max_results_per_search: int = 1
    max_queries_per_round: int = 1
    max_question_cards_effective: int = 4
    max_concurrent_tracks: int = 4
    min_rounds_per_track: int = 2
    round_search_budget: int = 2
    round_fetch_budget: int = 6
    review_source_window: int = 48
    report_source_batch_size: int = 8
    report_source_batch_chars: int = 48_000
    fetch_page_max_chars: int = 10_000
    explore_target_pages_per_round: int = 3
    explore_links_per_page: int = 8


class GlobalBudget(MutableModel):
    total_search: int = 0
    total_fetch: int = 0
    search_used: int = 0
    fetch_used: int = 0
    tier: ResearchBudgetTier = "base"

    @property
    def search_remaining(self) -> int:
        return max(0, self.total_search - self.search_used)

    @property
    def fetch_remaining(self) -> int:
        return max(0, self.total_fetch - self.fetch_used)

    @property
    def is_exhausted(self) -> bool:
        return self.search_remaining <= 0 and self.fetch_remaining <= 0


class TrackAllocation(MutableModel):
    search_quota: int = 0
    fetch_quota: int = 0
    search_used: int = 0
    fetch_used: int = 0
    minimum_search: int = 0
    minimum_fetch: int = 0

    @property
    def search_remaining(self) -> int:
        return max(0, self.search_quota - self.search_used)

    @property
    def fetch_remaining(self) -> int:
        return max(0, self.fetch_quota - self.fetch_used)

    @property
    def reclaimable_search(self) -> int:
        return max(0, self.search_remaining - self.minimum_search)

    @property
    def reclaimable_fetch(self) -> int:
        return max(0, self.fetch_remaining - self.minimum_fetch)

    @property
    def is_exhausted(self) -> bool:
        return self.search_remaining <= 0 and self.fetch_remaining <= 0


class ResearchTrackRuntime(MutableModel):
    """Track runtime state with budget tracking.

    Uses composition with TrackAllocation for budget properties.
    Access allocation fields directly: runtime.allocation.search_quota
    """

    question_id: str = ""
    priority: int = 3
    state: ResearchTrackState = "active"
    budget_tier: ResearchBudgetTier = "base"
    completed_rounds: int = 0
    waiting_rounds: int = 0
    stop_reason: str = ""
    # Budget allocation using composition
    allocation: TrackAllocation = Field(default_factory=TrackAllocation)


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
    used_in_report: bool = False
    # Advanced fields for world-class research
    domain: str = ""
    content_type: Literal[
        "article",
        "documentation",
        "paper",
        "blog",
        "forum",
        "official",
        "news",
        "reference",
        "unknown",
    ] = "unknown"
    publication_date: str = ""  # ISO date string if available
    last_updated_date: str = ""  # ISO date string if available
    freshness_score: float = 0.0  # 0.0 to 1.0 based on recency
    credibility_signals: list[str] = Field(default_factory=list)
    bias_indicators: list[str] = Field(default_factory=list)
    geographic_region: str = ""
    language: str = ""


class ResearchTrackResult(MutableModel):
    question_id: str
    question: str
    stop_reason: str = ""
    rounds: int = 0
    search_used: int = 0
    fetch_used: int = 0
    confidence: float = 0.0
    coverage_ratio: float = 0.0
    unresolved_conflicts: int = 0
    subreport_markdown: str = ""
    track_insight_card: TrackInsightCardPayload | None = None
    key_findings: list[str] = Field(default_factory=list)
    budget_tier: ResearchBudgetTier = "base"
    waiting_rounds: int = 0


class ResearchTask(MutableModel):
    question: str = ""
    style: ReportStyle = "explainer"
    intent: TaskIntent = "other"
    complexity: TaskComplexity = "medium"
    input_language: LanguageCode = "other"
    output_language: LanguageCode = "other"
    subthemes: list[str] = Field(default_factory=list, max_length=12)
    entities: list[str] = Field(default_factory=list, max_length=16)
    cards: list[ResearchThemePlanCard] = Field(default_factory=list, max_length=24)


class ResearchKnowledge(MutableModel):
    sources: list[ResearchSource] = Field(default_factory=list)
    source_ids_by_url: dict[str, list[int]] = Field(default_factory=dict)
    ranked_source_ids: list[int] = Field(default_factory=list)
    source_scores: dict[int, float] = Field(default_factory=dict)
    covered_subthemes: list[str] = Field(default_factory=list)
    report_used_source_ids: list[int] = Field(default_factory=list)


class ResearchRound(MutableModel):
    round_index: int = 0
    round_action: RoundAction = "search"
    query_strategy: str = ""
    queries: list[QuerySourceSpec] = Field(default_factory=list)
    search_jobs: list[PlanSearchJobPayload] = Field(default_factory=list)
    explore_target_source_ids: list[int] = Field(default_factory=list)
    search_fetched_candidates: list[SearchFetchedCandidate] = Field(
        default_factory=list
    )
    pending_search_jobs: list[PlanSearchJobPayload] = Field(default_factory=list)
    overview_review: OverviewReviewPayload | None = None
    content_review: ContentReviewPayload | None = None
    result_count: int = 0
    overview_summary: str = ""
    content_summary: str = ""
    coverage_ratio: float = 0.0
    uncertainty_score: float = 0.0
    waiting_for_budget: bool = False
    waiting_reason: str = ""
    stop_reason: str = ""
    stop: bool = False

    @property
    def has_pending_io(self) -> bool:
        return bool(self.pending_search_jobs or self.search_fetched_candidates)

    @property
    def needs_resume(self) -> bool:
        return bool(self.waiting_for_budget or self.has_pending_io)

    @property
    def is_review_ready(self) -> bool:
        return not self.needs_resume

    @property
    def confidence(self) -> float:
        if self.content_review is not None:
            return float(self.content_review.confidence_score)
        if self.overview_review is not None:
            return float(self.overview_review.confidence_score)
        return 0.0

    @property
    def missing_entities(self) -> list[str]:
        if self.overview_review is None:
            return []
        return list(self.overview_review.missing_entities)

    @property
    def unresolved_conflicts(self) -> int:
        if self.content_review is None:
            return 0
        return sum(
            1
            for item in self.content_review.conflict_resolutions
            if item.status in {"unresolved", "insufficient"}
        )

    @property
    def remaining_gap_count(self) -> int:
        if self.content_review is None:
            return 0
        return len(self.content_review.remaining_gaps)


class RoundState(MutableModel):
    limits: ResearchLimits = Field(default_factory=ResearchLimits)
    allocation: TrackAllocation = Field(default_factory=TrackAllocation)
    round_index: int = 0
    current: ResearchRound | None = None
    history: list[ResearchRound] = Field(default_factory=list)
    stop: bool = False
    stop_reason: str = ""
    next_queries: list[QuerySourceSpec] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    link_candidates: dict[int, SearchFetchedCandidate] = Field(default_factory=dict)
    link_candidates_round: int = 0
    explore_resolved_relative_links: int = 0

    @property
    def latest_round(self) -> ResearchRound | None:
        if self.history:
            return self.history[-1]
        return self.current

    def archive_current_round(
        self,
        *,
        stop: bool | None = None,
        stop_reason: str | None = None,
    ) -> ResearchRound | None:
        current = self.current
        if current is None:
            return None
        if stop is not None:
            current.stop = stop
        if stop_reason is not None:
            current.stop_reason = stop_reason
        self.history.append(current)
        self.current = None
        return current


class ResearchRun(MutableModel):
    mode: Literal["research-fast", "research", "research-pro"] = "research"
    limits: ResearchLimits = Field(default_factory=ResearchLimits)
    budget: GlobalBudget = Field(default_factory=GlobalBudget)
    track_runtimes: dict[str, ResearchTrackRuntime] = Field(default_factory=dict)
    stop: bool = False
    stop_reason: str = ""
    notes: list[str] = Field(default_factory=list)
    explore_resolved_relative_links: int = 0


class ResearchResult(MutableModel):
    content: str = ""
    structured: object | None = None
    tracks: list[ResearchTrackResult] = Field(default_factory=list)


class RoundStepContext(BaseStepContext[ResearchRequest, ResearchResponse]):
    """Track-scoped context for round runner execution."""

    request: ResearchRequest
    response: ResearchResponse
    question_id: str = ""
    task: ResearchTask = Field(default_factory=ResearchTask)
    run: RoundState = Field(default_factory=RoundState)
    knowledge: ResearchKnowledge = Field(default_factory=ResearchKnowledge)


class ResearchStepContext(BaseStepContext[ResearchRequest, ResearchResponse]):
    """Global research context.

    For render/subreport steps, track_state provides access to the current
    track's execution state (history, notes, etc.) without duplicating
    those fields in ResearchRun.
    """

    request: ResearchRequest
    response: ResearchResponse
    task: ResearchTask = Field(default_factory=ResearchTask)
    run: ResearchRun = Field(default_factory=ResearchRun)
    knowledge: ResearchKnowledge = Field(default_factory=ResearchKnowledge)
    result: ResearchResult = Field(default_factory=ResearchResult)
    track_state: RoundState | None = None


RoundState.model_rebuild()
ResearchRun.model_rebuild()


__all__ = [
    "GlobalBudget",
    "ResearchCorpusUpsertResult",
    "ResearchKnowledge",
    "ResearchLimits",
    "ResearchResult",
    "ResearchRound",
    "ResearchRun",
    "ResearchSource",
    "ResearchStepContext",
    "ResearchTask",
    "ResearchThemePlan",
    "ResearchTrackResult",
    "ResearchTrackRuntime",
    "ResearchWriterSectionFailure",
    "RoundState",
    "RoundStepContext",
    "TrackAllocation",
]
