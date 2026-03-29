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
    FetchResponse,
    FetchResultItem,
    FetchSubpagesResult,
)
from serpsage.models.base import MutableModel
from serpsage.models.components.crawl import CrawlResult
from serpsage.models.components.extract import (
    ExtractedDocument,
    ExtractRef,
    ExtractSpec,
)
from serpsage.models.steps.base import BaseStepContext


class PreparedPassage(MutableModel):
    text: str
    heading: str = ""
    order: int = 0


class ScoredPassage(MutableModel):
    passage_id: str
    text: str
    score: float


class FetchErrorState(MutableModel):
    failed: bool = False
    tag: FetchErrorTag = "CRAWL_UNKNOWN_ERROR"
    detail: str | None = None


class FetchPageState(MutableModel):
    crawl_mode: CrawlMode = "fallback"
    crawl_timeout_s: float = 0.0
    raw: CrawlResult | None = None
    return_content: bool = True
    content_request: FetchContentRequest = Field(default_factory=FetchContentRequest)
    extract: ExtractSpec = Field(default_factory=ExtractSpec)
    doc: ExtractedDocument | None = None
    pre_fetched_title: str = ""
    pre_fetched_content: str = ""
    pre_fetched_author: str = ""


class FetchAbstractState(MutableModel):
    request: FetchAbstractsRequest | None = None
    prepared: list[PreparedPassage] = Field(default_factory=list)
    ranked: list[ScoredPassage] = Field(default_factory=list)


class FetchOverviewState(MutableModel):
    request: FetchOverviewRequest | None = None
    ranked: list[ScoredPassage] = Field(default_factory=list)
    output: str | object | None = None


class FetchAnalysisState(MutableModel):
    abstracts: FetchAbstractState = Field(default_factory=FetchAbstractState)
    overview: FetchOverviewState = Field(default_factory=FetchOverviewState)


class FetchSubpageState(MutableModel):
    url: str = ""
    result: FetchSubpagesResult | None = None
    doc: ExtractedDocument | None = None
    overview_scores: list[float] = Field(default_factory=list)


class FetchSubpageGroupState(MutableModel):
    enabled: bool = False
    limit: int = 0
    candidate_limit: int | None = None
    keywords: list[str] = Field(default_factory=list)
    candidates: list[ExtractRef] = Field(default_factory=list)
    items: list[FetchSubpageState] = Field(default_factory=list)


class FetchRelatedState(MutableModel):
    enabled: bool = True
    link_limit: int | None = None
    image_limit: int | None = None
    others: FetchOthersResult = Field(default_factory=FetchOthersResult)
    subpages: FetchSubpageGroupState = Field(default_factory=FetchSubpageGroupState)


class FetchStepContext(BaseStepContext[FetchRequest, FetchResponse]):
    request: FetchRequest
    response: FetchResponse
    url: str
    url_index: int
    page: FetchPageState = Field(default_factory=FetchPageState)
    analysis: FetchAnalysisState = Field(default_factory=FetchAnalysisState)
    related: FetchRelatedState = Field(default_factory=FetchRelatedState)
    result: FetchResultItem | None = None
    error: FetchErrorState = Field(default_factory=FetchErrorState)


__all__ = [
    "FetchAbstractState",
    "FetchAnalysisState",
    "FetchErrorState",
    "FetchOverviewState",
    "FetchPageState",
    "FetchRelatedState",
    "FetchStepContext",
    "FetchSubpageGroupState",
    "FetchSubpageState",
    "PreparedPassage",
    "ScoredPassage",
]
