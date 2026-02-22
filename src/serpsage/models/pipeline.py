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
    SearchRequest,
)
from serpsage.app.response import (
    AnswerCitation,
    FetchOthersResult,
    FetchResultItem,
    FetchSubpagesResult,
)
from serpsage.core.model_base import MutableModel
from serpsage.models.errors import AppError
from serpsage.models.extract import (
    ExtractContentOptions,
    ExtractedDocument,
    ExtractedLink,
)
from serpsage.models.fetch import FetchResult
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


class AnswerPlanState(MutableModel):
    answer_mode: str = "summary"
    freshness_intent: bool = False
    query_language: str = "same as query"
    search_query: str = ""
    search_mode: str = "auto"
    max_results: int = 1
    additional_queries: list[str] | None = None


class SearchStepContext(BaseStepContext):
    settings: AppSettings
    request: SearchRequest
    plan: AnswerPlanState = Field(default_factory=AnswerPlanState)
    deep: SearchDeepState = Field(default_factory=SearchDeepState)
    prefetch: SearchPrefetchState = Field(default_factory=SearchPrefetchState)
    fetch: SearchFetchState = Field(default_factory=SearchFetchState)
    rank: SearchRankState = Field(default_factory=SearchRankState)
    output: SearchOutputState = Field(default_factory=SearchOutputState)
    errors: list[AppError] = Field(default_factory=list)


class AnswerSearchState(MutableModel):
    request: SearchRequest | None = None
    search_mode: str = "auto"
    results: list[FetchResultItem] = Field(default_factory=list)


class AnswerOutputState(MutableModel):
    answers: str | object = ""
    citations: list[AnswerCitation] = Field(default_factory=list)


class AnswerStepContext(BaseStepContext):
    settings: AppSettings
    request: AnswerRequest
    plan: AnswerPlanState = Field(default_factory=AnswerPlanState)
    search: AnswerSearchState = Field(default_factory=AnswerSearchState)
    output: AnswerOutputState = Field(default_factory=AnswerOutputState)
    errors: list[AppError] = Field(default_factory=list)


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
    errors: list[AppError] = Field(default_factory=list)


__all__ = [
    "AnswerOutputState",
    "AnswerPlanState",
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
