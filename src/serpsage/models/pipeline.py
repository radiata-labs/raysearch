from __future__ import annotations

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


class SearchPrefetchState(MutableModel):
    urls: list[str] = Field(default_factory=list)
    scores: dict[str, float] = Field(default_factory=dict)


class SearchFetchState(MutableModel):
    candidates: list[SearchFetchedCandidate] = Field(default_factory=list)


class SearchOutputState(MutableModel):
    results: list[FetchResultItem] = Field(default_factory=list)


class SearchStepContext(BaseStepContext):
    settings: AppSettings
    request: SearchRequest
    prefetch: SearchPrefetchState = Field(default_factory=SearchPrefetchState)
    fetch: SearchFetchState = Field(default_factory=SearchFetchState)
    output: SearchOutputState = Field(default_factory=SearchOutputState)
    errors: list[AppError] = Field(default_factory=list)


class AnswerPlanState(MutableModel):
    answer_mode: str = "summary"
    search_query: str = ""
    search_depth: str = "auto"
    max_results: int = 1
    additional_queries: list[str] | None = None


class AnswerSearchState(MutableModel):
    request: SearchRequest | None = None
    search_depth: str = "auto"
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
    "SearchFetchState",
    "SearchFetchedCandidate",
    "SearchOutputState",
    "SearchPrefetchState",
    "SearchStepContext",
]
