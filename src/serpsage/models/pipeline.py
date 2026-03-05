from __future__ import annotations

from typing import Literal

from pydantic import Field

from serpsage.app.request import (
    AnswerRequest,
    CrawlMode,
    FetchAbstractsRequest,
    FetchContentRequest,
    FetchOverviewRequest,
    FetchRequest,
    ResearchRequest,
    SearchRequest,
)
from serpsage.app.response import (
    AnswerCitation,
    FetchErrorTag,
    FetchOthersResult,
    FetchResultItem,
    FetchSubpagesResult,
)
from serpsage.core.model_base import MutableModel
from serpsage.models.extract import (
    ExtractContentOptions,
    ExtractedDocument,
    ExtractedLink,
)
from serpsage.models.fetch import FetchResult
from serpsage.models.research import (
    ContentOutputPayload,
    OverviewOutputPayload,
    ResearchThemePlan,
    TrackInsightCardPayload,
)
from serpsage.settings.models import AppSettings


class BaseStepContext(MutableModel):
    request_id: str = ""


class SearchQueryJob(MutableModel):
    query: str
    weight: float = 1.0
    source: Literal["primary", "manual", "rule", "llm"] = "primary"


class SearchSnippetContext(MutableModel):
    snippet: str
    source_query: str
    source_type: Literal["primary", "manual", "rule", "llm"]
    score: float = 0.0
    order: int = 0


class SearchDeepState(MutableModel):
    aborted: bool = False
    abort_reason: str = ""
    query_jobs: list[SearchQueryJob] = Field(default_factory=list)
    snippet_context: dict[str, list[SearchSnippetContext]] = Field(default_factory=dict)
    query_hit_stats: dict[str, int] = Field(default_factory=dict)
    context_scores: dict[str, float] = Field(default_factory=dict)


class SearchPrefetchState(MutableModel):
    urls: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)


class SearchFetchState(MutableModel):
    candidates: list[SearchFetchedCandidate] = Field(default_factory=list)


class SearchRankedCandidate(MutableModel):
    result: FetchResultItem
    final_score: float = 0.0
    order: int = 0
    page_score: float = 0.0
    context_score: float = 0.0
    prefetch_score: float = 0.0


class SearchRankState(MutableModel):
    candidates: list[SearchRankedCandidate] = Field(default_factory=list)
    filtered_count: int = 0
    sum_page_score: float = 0.0
    sum_context_score: float = 0.0
    sum_prefetch_score: float = 0.0
    deep_enabled: bool = False
    has_sort_feature: bool = False
    max_results: int = 1
    page_weight: float = 1.0
    context_weight: float = 0.0
    prefetch_weight: float = 0.0


class SearchOutputState(MutableModel):
    results: list[FetchResultItem] = Field(default_factory=list)


class AnswerSubQuestionPlan(MutableModel):
    question: str = ""
    search_query: str = ""


class AnswerPlanState(MutableModel):
    answer_mode: str = "summary"
    freshness_intent: bool = False
    query_language: str = "same as query"
    search_query: str = ""
    search_mode: str = "auto"
    max_results: int = 1
    additional_queries: list[str] | None = None
    sub_questions: list[AnswerSubQuestionPlan] = Field(default_factory=list)


class SearchStepContext(BaseStepContext):
    settings: AppSettings
    request: SearchRequest
    disable_internal_llm: bool = False
    provider_params: dict[str, str] = Field(default_factory=dict)
    deep: SearchDeepState = Field(default_factory=SearchDeepState)
    prefetch: SearchPrefetchState = Field(default_factory=SearchPrefetchState)
    fetch: SearchFetchState = Field(default_factory=SearchFetchState)
    rank: SearchRankState = Field(default_factory=SearchRankState)
    output: SearchOutputState = Field(default_factory=SearchOutputState)


class AnswerSubSearchState(MutableModel):
    question: str = ""
    search_query: str = ""
    request: SearchRequest | None = None
    search_mode: str = "auto"
    results: list[FetchResultItem] = Field(default_factory=list)


class AnswerSearchState(MutableModel):
    request: SearchRequest | None = None
    search_mode: str = "auto"
    results: list[FetchResultItem] = Field(default_factory=list)
    sub_searches: list[AnswerSubSearchState] = Field(default_factory=list)


class AnswerOutputState(MutableModel):
    answers: str | object = ""
    citations: list[AnswerCitation] = Field(default_factory=list)


class AnswerStepContext(BaseStepContext):
    settings: AppSettings
    request: AnswerRequest
    plan: AnswerPlanState = Field(default_factory=AnswerPlanState)
    search: AnswerSearchState = Field(default_factory=AnswerSearchState)
    output: AnswerOutputState = Field(default_factory=AnswerOutputState)


class ScoredAbstract(MutableModel):
    abstract_id: str
    text: str
    score: float


class PreparedAbstract(MutableModel):
    text: str
    heading: str = ""
    position: int = 0


class SearchFetchedCandidate(MutableModel):
    result: FetchResultItem
    links: list[ExtractedLink] = Field(default_factory=list)
    subpage_links: list[list[ExtractedLink]] = Field(default_factory=list)
    main_md_for_abstract: str = ""
    subpages_md_for_abstract: list[str] = Field(default_factory=list)
    main_overview_scores: list[float] = Field(default_factory=list)
    subpages_overview_scores: list[list[float]] = Field(default_factory=list)


class FetchRuntimeConfig(MutableModel):
    crawl_mode: CrawlMode = "fallback"
    crawl_timeout_s: float = 0.0
    max_links_for_subpages: int | None = None
    max_links: int | None = None
    max_image_links: int | None = None


class FetchResolvedState(MutableModel):
    return_content: bool = True
    content_request: FetchContentRequest = Field(default_factory=FetchContentRequest)
    content_options: ExtractContentOptions = Field(
        default_factory=ExtractContentOptions
    )
    abstracts_request: FetchAbstractsRequest | None = None
    overview_request: FetchOverviewRequest | None = None


class FetchArtifactsState(MutableModel):
    fetch_result: FetchResult | None = None
    extracted: ExtractedDocument | None = None
    prepared_abstracts: list[PreparedAbstract] = Field(default_factory=list)
    scored_abstracts: list[ScoredAbstract] = Field(default_factory=list)
    overview_scored_abstracts: list[ScoredAbstract] = Field(default_factory=list)
    overview_output: str | object | None = None


class FetchSubpagesState(MutableModel):
    enabled: bool = False
    links: list[ExtractedLink] = Field(default_factory=list)
    max_count: int = 0
    query: str = ""
    keywords: list[str] = Field(default_factory=list)
    results: list[FetchSubpagesResult] = Field(default_factory=list)
    result_links: list[list[ExtractedLink]] = Field(default_factory=list)
    md_for_abstract: list[str] = Field(default_factory=list)
    overview_scores: list[list[float]] = Field(default_factory=list)


class FetchOutputState(MutableModel):
    others: FetchOthersResult = Field(default_factory=FetchOthersResult)
    result: FetchResultItem | None = None


class FetchStepContext(BaseStepContext):
    settings: AppSettings
    request: FetchRequest
    url: str
    url_index: int
    runtime: FetchRuntimeConfig
    enable_others_and_subpages: bool = True
    resolved: FetchResolvedState = Field(default_factory=FetchResolvedState)
    artifacts: FetchArtifactsState = Field(default_factory=FetchArtifactsState)
    subpages: FetchSubpagesState = Field(default_factory=FetchSubpagesState)
    output: FetchOutputState = Field(default_factory=FetchOutputState)
    fatal: bool = False
    error_tag: FetchErrorTag = "CRAWL_UNKNOWN_ERROR"
    error_detail: str | None = None


class ResearchBudgetState(MutableModel):
    max_rounds: int = 1
    max_search_calls: int = 1
    max_fetch_calls: int = 1
    max_results_per_search: int = 1
    max_queries_per_round: int = 1
    max_fetch_per_round: int = 1
    stop_confidence: float = 0.80
    min_coverage_ratio: float = 0.80
    max_unresolved_conflicts: int = 1


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
    no_progress_rounds_to_stop_effective: int = 2
    gap_closure_passes: int = 1
    density_gate_passes: int = 1
    overview_source_topk: int = 20
    content_source_topk: int = 10
    content_source_chars: int = 10_000
    explore_target_pages_per_round: int = 3
    explore_links_per_page: int = 8


class ResearchLinkCandidate(MutableModel):
    source_id: int
    url: str = ""
    title: str = ""
    links: list[ExtractedLink] = Field(default_factory=list)
    subpage_links: list[list[ExtractedLink]] = Field(default_factory=list)
    round_index: int = 0


class ResearchRuntimeState(MutableModel):
    mode_depth: ResearchModeDepthState = Field(default_factory=ResearchModeDepthState)
    budget: ResearchBudgetState = Field(default_factory=ResearchBudgetState)
    search_calls: int = 0
    fetch_calls: int = 0
    no_progress_rounds: int = 0
    gap_closure_passes_applied: int = 0
    density_gate_passes_applied: int = 0
    provider_language_param_applied: bool = False
    query_language_repair_applied: bool = False
    search_language_fallback_applied: bool = False
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
    critical_gaps: int = 0
    stop_reason: str = ""
    stop: bool = False


class ResearchOutputState(MutableModel):
    content: str = ""
    structured: object | None = None


class ResearchStepContext(BaseStepContext):
    settings: AppSettings
    request: ResearchRequest
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
    "AnswerOutputState",
    "AnswerPlanState",
    "AnswerSubQuestionPlan",
    "AnswerSubSearchState",
    "AnswerSearchState",
    "AnswerStepContext",
    "BaseStepContext",
    "FetchArtifactsState",
    "FetchOutputState",
    "FetchResolvedState",
    "FetchStepContext",
    "FetchRuntimeConfig",
    "FetchSubpagesState",
    "PreparedAbstract",
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
    "ScoredAbstract",
    "SearchDeepState",
    "SearchFetchState",
    "SearchFetchedCandidate",
    "SearchRankedCandidate",
    "SearchRankState",
    "SearchQueryJob",
    "SearchOutputState",
    "SearchPrefetchState",
    "SearchSnippetContext",
    "SearchStepContext",
]
