from __future__ import annotations

import re
from collections.abc import Iterable
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
        "language",
        "location",
        "moderation",
        "start_published_date",
        "end_published_date",
    }
)
_LANGUAGE_TAG_RE = re.compile(r"^[A-Za-z]{2,3}(?:[-_][A-Za-z0-9]{2,8})*$")


def _normalize_language_tag(value: str) -> str:
    token = clean_whitespace(str(value or "")).replace("_", "-")
    if not token or token.casefold() == "all":
        return ""
    if not _LANGUAGE_TAG_RE.fullmatch(token):
        raise ValueError("provider_language must be a valid BCP 47 language tag")
    parts = [part for part in token.split("-") if part]
    if not parts:
        return ""
    normalized_parts = [parts[0].lower()]
    for part in parts[1:]:
        if len(part) == 4 and part.isalpha():
            normalized_parts.append(part.title())
            continue
        if (len(part) == 2 and part.isalpha()) or (len(part) == 3 and part.isdigit()):
            normalized_parts.append(part.upper())
            continue
        normalized_parts.append(part.lower())
    return "-".join(normalized_parts)


class QuerySourceSpec(MutableModel):
    query: str
    include_sources: list[str] = Field(default_factory=list)

    @field_validator("include_sources", mode="before")
    @classmethod
    def _validate_include_sources(cls, value: object) -> list[str]:
        if value is None:
            return []
        if isinstance(value, str):
            raw_items = [value]
        elif isinstance(value, Iterable):
            raw_items = list(value)
        else:
            raise TypeError("include_sources must be an iterable of strings")
        out: list[str] = []
        seen: set[str] = set()
        for raw in raw_items:
            token = clean_whitespace(str(raw or "")).casefold()
            if not token or token in seen:
                continue
            seen.add(token)
            out.append(token)
        return out


class SearchQueryJob(MutableModel):
    query: QuerySourceSpec
    source: Literal["primary", "manual", "llm"] = "primary"


class SearchSnippetContext(MutableModel):
    snippet: str
    source_query: str
    source_type: Literal["primary", "manual", "llm"]
    order: int = 0


class SearchRuntimeState(MutableModel):
    disable_internal_llm: bool = False
    engine_selection_subsystem: Literal["", "search", "research", "answer"] = ""
    provider_limit: int | None = None
    provider_language: str = ""
    provider_extra_kwargs: dict[str, Any] = Field(default_factory=dict)
    additional_queries: list[QuerySourceSpec] = Field(default_factory=list)

    @field_validator("provider_limit")
    @classmethod
    def _validate_provider_limit(cls, value: int | None) -> int | None:
        if value is None:
            return None
        if value <= 0:
            raise ValueError("provider_limit must be > 0")
        return value

    @field_validator("provider_language")
    @classmethod
    def _validate_provider_text(cls, value: str) -> str:
        return _normalize_language_tag(value)

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
    primary_query: QuerySourceSpec | None = None
    aborted: bool = False
    abort_reason: str = ""
    query_jobs: list[SearchQueryJob] = Field(default_factory=list)

    @model_validator(mode="after")
    def _validate_ranges(self) -> SearchPlanState:
        if self.max_results <= 0:
            raise ValueError("max_results must be > 0")
        if self.max_extra_queries < 0:
            raise ValueError("max_extra_queries must be >= 0")
        if self.prefetch_limit <= 0:
            raise ValueError("prefetch_limit must be > 0")
        if self.context_docs_limit < 0:
            raise ValueError("context_docs_limit must be >= 0")
        if self.context_doc_min_chars < 0:
            raise ValueError("context_doc_min_chars must be >= 0")
        return self


class SearchRetrievalState(MutableModel):
    urls: list[str] = Field(default_factory=list)
    published_dates: dict[str, str] = Field(default_factory=dict)
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
    published_date: str = ""


class SearchCanonicalBucket(MutableModel):
    representative_url: str
    representative_order: int
    published_date: str = ""
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


__all__ = [
    "SearchCandidateForScoring",
    "SearchCanonicalBucket",
    "SearchFetchState",
    "SearchFetchedCandidate",
    "SearchNormalizedResult",
    "SearchOutputState",
    "SearchPlanState",
    "QuerySourceSpec",
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
