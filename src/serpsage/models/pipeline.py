from __future__ import annotations

from pydantic import Field

from serpsage.app.request import (
    CrawlMode,
    FetchAbstractsRequest,
    FetchContentRequest,
    FetchOverviewRequest,
    FetchRequest,
    SearchRequest,
)
from serpsage.app.response import FetchOthersResult, FetchResultItem, ResultItem
from serpsage.core.model_base import MutableModel
from serpsage.models.errors import AppError
from serpsage.models.extract import ExtractContentOptions, ExtractedDocument
from serpsage.models.fetch import FetchResult
from serpsage.settings.models import AppSettings


class BaseStepContext(MutableModel):
    request_id: str = ""


class SearchStepContext(BaseStepContext):
    settings: AppSettings
    request: SearchRequest
    raw_results: list[dict[str, object]] = Field(default_factory=list)
    results: list[ResultItem] = Field(default_factory=list)
    query_tokens: list[str] | None = None
    overview: str | object | None = None
    errors: list[AppError] = Field(default_factory=list)


class FetchStepOthersRuntime(MutableModel):
    crawl_mode: CrawlMode = "fallback"
    crawl_timeout_s: float = 0.0
    allow_render: bool = True
    rank_index: int = 0
    max_links: int | None = None
    max_image_links: int | None = None


class ScoredAbstract(MutableModel):
    abstract_id: str
    text: str
    score: float


class FetchStepContext(BaseStepContext):
    settings: AppSettings
    request: FetchRequest
    url: str
    url_index: int
    others_runtime: FetchStepOthersRuntime
    return_content: bool = True
    content_request: FetchContentRequest = Field(default_factory=FetchContentRequest)
    content_options: ExtractContentOptions = Field(
        default_factory=ExtractContentOptions
    )
    abstracts_request: FetchAbstractsRequest | None = None
    overview_request: FetchOverviewRequest | None = None
    fetch_result: FetchResult | None = None
    extracted: ExtractedDocument | None = None
    scored_abstracts: list[ScoredAbstract] = Field(default_factory=list)
    others_result: FetchOthersResult = Field(default_factory=FetchOthersResult)
    result: FetchResultItem | None = None
    fatal: bool = False
    abstract_query_tokens: list[str] | None = None
    overview_output: str | object | None = None
    errors: list[AppError] = Field(default_factory=list)


__all__ = [
    "BaseStepContext",
    "FetchStepContext",
    "FetchStepOthersRuntime",
    "ScoredAbstract",
    "SearchStepContext",
]
