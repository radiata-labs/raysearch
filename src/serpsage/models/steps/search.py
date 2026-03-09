from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator

from serpsage.models.app.request import (
    SearchRequest,
)
from serpsage.models.app.response import (
    FetchResultItem,
    SearchResponse,
)
from serpsage.models.base import MutableModel
from serpsage.models.components.extract import (
    ExtractRef,
)
from serpsage.models.steps.base import BaseStepContext
from serpsage.settings.models import AppSettings
from serpsage.utils import clean_whitespace

_RESERVED_PROVIDER_KWARG_KEYS = {"query", "page", "language"}


class SearchQueryJob(MutableModel):
    query: str
    weight: float = 1.0
    source: Literal["primary", "manual", "rule", "llm"] = "primary"


class SearchQueryCandidate(MutableModel):
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


class SearchStepContext(BaseStepContext[SearchRequest, SearchResponse]):
    settings: AppSettings
    request: SearchRequest
    response: SearchResponse
    disable_internal_llm: bool = False
    provider_page: int = 1
    provider_language: str = ""
    provider_extra_kwargs: dict[str, Any] = Field(default_factory=dict)
    deep: SearchDeepState = Field(default_factory=SearchDeepState)
    prefetch: SearchPrefetchState = Field(default_factory=SearchPrefetchState)
    fetch: SearchFetchState = Field(default_factory=SearchFetchState)
    rank: SearchRankState = Field(default_factory=SearchRankState)
    output: SearchOutputState = Field(default_factory=SearchOutputState)

    @field_validator("provider_page")
    @classmethod
    def _validate_provider_page(cls, value: int) -> int:
        if int(value) <= 0:
            raise ValueError("provider_page must be > 0")
        return int(value)

    @field_validator("provider_language")
    @classmethod
    def _validate_provider_language(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @field_validator("provider_extra_kwargs")
    @classmethod
    def _validate_provider_extra_kwargs(cls, value: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for raw_key, raw_value in dict(value or {}).items():
            key = clean_whitespace(str(raw_key or ""))
            if not key:
                continue
            if key.casefold() in _RESERVED_PROVIDER_KWARG_KEYS:
                raise ValueError(
                    f"provider_extra_kwargs must not override reserved key `{key}`"
                )
            out[key] = raw_value
        return out


class SearchFetchedCandidate(MutableModel):
    result: FetchResultItem
    links: list[ExtractRef] = Field(default_factory=list)
    subpage_links: list[list[ExtractRef]] = Field(default_factory=list)
    main_abstract_text: str = ""
    subpage_abstract_texts: list[str] = Field(default_factory=list)
    main_overview_scores: list[float] = Field(default_factory=list)
    subpages_overview_scores: list[list[float]] = Field(default_factory=list)


class SearchNormalizedResult(MutableModel):
    url: str
    canonical_url: str = ""
    title: str = ""
    snippet: str = ""


class SearchScoredHit(MutableModel):
    job_index: int
    order: int
    item: SearchNormalizedResult


class SearchCanonicalBucket(MutableModel):
    representative_url: str
    representative_order: int
    representative_score: float
    hit_indexes: set[int] = Field(default_factory=set)
    hit_scores: list[float] = Field(default_factory=list)
    snippets_by_source: dict[str, SearchSnippetContext] = Field(default_factory=dict)


class SearchRankOptions(MutableModel):
    content_enabled: bool
    abstracts_enabled: bool
    overview_enabled: bool
    has_sort_feature: bool
    include_text: list[str] = Field(default_factory=list)
    exclude_text: list[str] = Field(default_factory=list)
    query_tokens: list[str] = Field(default_factory=list)
    context_query_tokens: list[str] = Field(default_factory=list)
    deep_enabled: bool
    context_docs_limit: int
    context_doc_min_chars: int
    max_results: int
    page_weight: float
    context_weight: float
    prefetch_weight: float


class SearchRankStats(MutableModel):
    filtered_count: int = 0
    sum_page_score: float = 0.0
    sum_context_score: float = 0.0
    sum_prefetch_score: float = 0.0


class SearchCandidateForScoring(MutableModel):
    order: int
    candidate: SearchFetchedCandidate
    main_text: str = ""
    subpage_inputs: list[tuple[str, list[float], list[float]]] = Field(
        default_factory=list
    )


class SearchOptimizedQuery(MutableModel):
    search_query: str
    optimize_query: bool
    freshness_intent: bool
    query_language: str


__all__ = [
    "SearchCandidateForScoring",
    "SearchCanonicalBucket",
    "SearchDeepState",
    "SearchFetchState",
    "SearchFetchedCandidate",
    "SearchNormalizedResult",
    "SearchOptimizedQuery",
    "SearchQueryCandidate",
    "SearchRankedCandidate",
    "SearchRankOptions",
    "SearchRankState",
    "SearchRankStats",
    "SearchQueryJob",
    "SearchOutputState",
    "SearchPrefetchState",
    "SearchScoredHit",
    "SearchSnippetContext",
    "SearchStepContext",
]
