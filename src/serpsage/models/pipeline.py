from __future__ import annotations

from pydantic import Field

from serpsage.app.request import (
    FetchChunksRequest,
    FetchContentRequest,
    FetchOverviewRequest,
    FetchRequest,
    SearchRequest,
)
from serpsage.app.response import OverviewResult, PageChunk, PageEnrichment, ResultItem
from serpsage.core.model_base import MutableModel
from serpsage.models.errors import AppError
from serpsage.models.extract import ExtractContentOptions, ExtractedDocument
from serpsage.models.fetch import FetchResult
from serpsage.settings.models import AppSettings, ProfileSettings


class BaseStepContext(MutableModel):
    request_id: str = ""


class SearchStepContext(BaseStepContext):
    settings: AppSettings
    request: SearchRequest
    raw_results: list[dict[str, object]] = Field(default_factory=list)
    results: list[ResultItem] = Field(default_factory=list)
    profile_name: str = ""
    profile: ProfileSettings | None = None
    query_tokens: list[str] | None = None
    intent_tokens: list[str] | None = None
    overview: OverviewResult | None = None
    errors: list[AppError] = Field(default_factory=list)


class FetchStepContext(BaseStepContext):
    settings: AppSettings
    request: FetchRequest
    return_content: bool = True
    content_request: FetchContentRequest = Field(default_factory=FetchContentRequest)
    content_options: ExtractContentOptions = Field(default_factory=ExtractContentOptions)
    chunks_request: FetchChunksRequest | None = None
    overview_request: FetchOverviewRequest | None = None
    fetch_result: FetchResult | None = None
    extracted: ExtractedDocument | None = None
    chunks: list[PageChunk] = Field(default_factory=list)
    page: PageEnrichment = Field(default_factory=PageEnrichment)
    profile_name: str = ""
    profile: ProfileSettings | None = None
    chunk_query_tokens: list[str] | None = None
    chunk_intent_tokens: list[str] | None = None
    overview: OverviewResult | None = None
    errors: list[AppError] = Field(default_factory=list)


__all__ = ["BaseStepContext", "FetchStepContext", "SearchStepContext"]
