from __future__ import annotations

from typing import Any, Literal

from pydantic import Field, field_validator, model_validator

from serpsage.models.app.request import SearchRequest
from serpsage.models.app.response import FetchResultItem, SearchResponse
from serpsage.models.base import MutableModel
from serpsage.models.components.extract import ExtractRef
from serpsage.models.steps.base import BaseStepContext
from serpsage.utils import clean_whitespace

_RESERVED_PROVIDER_KWARG_KEYS = frozenset(
    {
        "query",
        "limit",
        "locale",
        "start_published_date",
        "end_published_date",
    }
)


class SearchQueryJob(MutableModel):
    query: str
    source: Literal["primary", "manual", "rule", "llm"] = "primary"


class SearchQueryCandidate(MutableModel):
    query: str
    source: Literal["primary", "manual", "rule", "llm"] = "primary"


class SearchSnippetContext(MutableModel):
    snippet: str
    source_query: str
    source_type: Literal["primary", "manual", "rule", "llm"]
    order: int = 0


class SearchRuntimeState(MutableModel):
    disable_internal_llm: bool = False
    provider_limit: int | None = None
    provider_locale: str = ""
    provider_extra_kwargs: dict[str, Any] = Field(default_factory=dict)

    @field_validator("provider_limit")
    @classmethod
    def _validate_provider_limit(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if int(value) <= 0:
            raise ValueError("provider_limit must be > 0")
        return int(value)

    @field_validator("provider_locale")
    @classmethod
    def _validate_provider_text(cls, value: str) -> str:
        return clean_whitespace(str(value or ""))

    @field_validator("provider_extra_kwargs", mode="before")
    @classmethod
    def _validate_provider_extra_kwargs(cls, value: Any) -> dict[str, Any]:
        if value is None:
            return {}
        if not isinstance(value, dict):
            raise TypeError("provider_extra_kwargs must be a mapping")
        out: dict[str, Any] = {}
        overlap: list[str] = []
        for raw_key, raw_value in value.items():
            key = clean_whitespace(str(raw_key or ""))
            if not key:
                continue
            if key.casefold() in _RESERVED_PROVIDER_KWARG_KEYS:
                overlap.append(key)
                continue
            out[key] = raw_value
        if overlap:
            names = ", ".join(sorted(set(overlap)))
            raise ValueError(
                f"provider_extra_kwargs must not override reserved keys: {names}"
            )
        return out


class SearchPlanState(MutableModel):
    mode: Literal["fast", "auto", "deep"] = "auto"
    max_results: int = 1
    max_extra_queries: int = 0
    prefetch_limit: int = 1
    context_docs_limit: int = 0
    context_doc_min_chars: int = 0
    rank_by_context: bool = False
    optimize_query: bool = False
    optimized_query: str = ""
    aborted: bool = False
    abort_reason: str = ""
    query_jobs: list[SearchQueryJob] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_ranges(self) -> SearchPlanState:
        if int(self.max_results) <= 0:
            raise ValueError("max_results must be > 0")
        if int(self.max_extra_queries) < 0:
            raise ValueError("max_extra_queries must be >= 0")
        if int(self.prefetch_limit) <= 0:
            raise ValueError("prefetch_limit must be > 0")
        if int(self.context_docs_limit) < 0:
            raise ValueError("context_docs_limit must be >= 0")
        if int(self.context_doc_min_chars) < 0:
            raise ValueError("context_doc_min_chars must be >= 0")
        return self


class SearchRetrievalState(MutableModel):
    urls: list[str] = Field(default_factory=list)
    snippet_context: dict[str, list[SearchSnippetContext]] = Field(default_factory=dict)
    query_hit_stats: dict[str, int] = Field(default_factory=dict)


class SearchRankedCandidate(MutableModel):
    result: FetchResultItem
    final_score: float = 0.0
    order: int = 0
    page_score: float = 0.0
    context_score: float = 0.0


class SearchRankState(MutableModel):
    candidates: list[SearchRankedCandidate] = Field(default_factory=list)
    filtered_count: int = 0
    sum_page_score: float = 0.0
    sum_context_score: float = 0.0
    has_sort_feature: bool = False
    use_context_score: bool = False
    max_results: int = 1
    context_scores: dict[str, float] = Field(default_factory=dict)


class SearchOutputState(MutableModel):
    results: list[FetchResultItem] = Field(default_factory=list)


class SearchFetchedCandidate(MutableModel):
    result: FetchResultItem
    links: list[ExtractRef] = Field(default_factory=list)
    subpage_links: list[list[ExtractRef]] = Field(default_factory=list)
    main_abstract_text: str = ""
    subpage_abstract_texts: list[str] = Field(default_factory=list)
    main_overview_scores: list[float] = Field(default_factory=list)
    subpages_overview_scores: list[list[float]] = Field(default_factory=list)


class SearchFetchState(MutableModel):
    candidates: list[SearchFetchedCandidate] = Field(default_factory=list)


class SearchStepContext(BaseStepContext[SearchRequest, SearchResponse]):
    request: SearchRequest
    response: SearchResponse
    runtime: SearchRuntimeState = Field(default_factory=SearchRuntimeState)
    plan: SearchPlanState = Field(default_factory=SearchPlanState)
    retrieval: SearchRetrievalState = Field(default_factory=SearchRetrievalState)
    fetch: SearchFetchState = Field(default_factory=SearchFetchState)
    rank: SearchRankState = Field(default_factory=SearchRankState)
    output: SearchOutputState = Field(default_factory=SearchOutputState)


class SearchNormalizedResult(MutableModel):
    url: str
    canonical_url: str = ""
    title: str = ""
    snippet: str = ""


class SearchCanonicalBucket(MutableModel):
    representative_url: str
    representative_order: int
    hit_indexes: set[int] = Field(default_factory=set)
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
    use_context_score: bool
    context_docs_limit: int
    context_doc_min_chars: int
    max_results: int


class SearchRankStats(MutableModel):
    filtered_count: int = 0
    sum_page_score: float = 0.0
    sum_context_score: float = 0.0


class SearchCandidateForScoring(MutableModel):
    order: int
    candidate: SearchFetchedCandidate
    main_text: str = ""
    subpage_inputs: list[tuple[str, list[str], list[float]]] = Field(
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
    "SearchFetchState",
    "SearchFetchedCandidate",
    "SearchNormalizedResult",
    "SearchOptimizedQuery",
    "SearchOutputState",
    "SearchPlanState",
    "SearchQueryCandidate",
    "SearchQueryJob",
    "SearchRankedCandidate",
    "SearchRankOptions",
    "SearchRankState",
    "SearchRankStats",
    "SearchRetrievalState",
    "SearchRuntimeState",
    "SearchSnippetContext",
    "SearchStepContext",
]
