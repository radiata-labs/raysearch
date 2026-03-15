"""
SerpSage 3.0 (async-only).
Public API:
- Engine: async search pipeline orchestrator
- load_settings: load raw settings data from YAML/JSON/env
- SearchRequest/SearchResponse + FetchRequest/FetchResponse + AnswerRequest/AnswerResponse
  request/response models
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serpsage.app.engine import Engine
    from serpsage.core.overrides import Overrides
from serpsage.models.app.request import (
    AnswerRequest,
    CrawlMode,
    FetchAbstractsRequest,
    FetchContentDetail,
    FetchContentRequest,
    FetchOthersRequest,
    FetchOverviewRequest,
    FetchRequest,
    FetchSubpagesRequest,
    ResearchRequest,
    ResearchSearchMode,
    SearchFetchRequest,
    SearchRequest,
)
from serpsage.models.app.response import (
    AnswerCitation,
    AnswerResponse,
    FetchResponse,
    FetchSubpagesResult,
    ResearchResponse,
    SearchResponse,
)
from serpsage.settings.load import load_settings
from serpsage.settings.models import AppSettings


def __getattr__(name: str) -> object:
    if name == "Engine":
        from serpsage.app.engine import Engine

        return Engine
    if name == "Overrides":
        from serpsage.core.overrides import Overrides

        return Overrides
    raise AttributeError(name)


__all__ = [
    "AppSettings",
    "AnswerRequest",
    "AnswerCitation",
    "AnswerResponse",
    "CrawlMode",
    "Engine",
    "FetchOthersRequest",
    "FetchAbstractsRequest",
    "FetchContentDetail",
    "FetchContentRequest",
    "FetchOverviewRequest",
    "FetchRequest",
    "FetchSubpagesRequest",
    "FetchResponse",
    "FetchSubpagesResult",
    "SearchFetchRequest",
    "SearchRequest",
    "SearchResponse",
    "ResearchSearchMode",
    "ResearchRequest",
    "ResearchResponse",
    "Overrides",
    "load_settings",
]
