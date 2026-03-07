from __future__ import annotations

from pydantic import Field

from serpsage.models.app.request import (
    CrawlMode,
    FetchAbstractsRequest,
    FetchContentRequest,
    FetchOverviewRequest,
    FetchRequest,
)
from serpsage.models.app.response import (
    FetchErrorTag,
    FetchOthersResult,
    FetchResultItem,
    FetchSubpagesResult,
)
from serpsage.models.base import MutableModel
from serpsage.models.components.extract import (
    ExtractContentOptions,
    ExtractedDocument,
    ExtractedLink,
)
from serpsage.models.components.fetch import FetchResult
from serpsage.models.steps.base import BaseStepContext
from serpsage.settings.models import AppSettings


class ScoredAbstract(MutableModel):
    abstract_id: str
    text: str
    score: float


class PreparedAbstract(MutableModel):
    text: str
    heading: str = ""
    position: int = 0


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


__all__ = [
    "FetchArtifactsState",
    "FetchOutputState",
    "FetchResolvedState",
    "FetchStepContext",
    "FetchRuntimeConfig",
    "FetchSubpagesState",
    "PreparedAbstract",
    "ScoredAbstract",
]
