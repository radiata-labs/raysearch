"""
SerpSage 3.0 (async-only).
Public API:
- Engine: async search pipeline orchestrator
- load_settings: load AppSettings from YAML/JSON/env
- SearchRequest/SearchResponse + FetchRequest/FetchResponse + AnswerRequest/AnswerResponse
  request/response models
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serpsage.app.engine import Engine
from serpsage.app.request import (
    AnswerRequest,
    CrawlMode,
    FetchAbstractsRequest,
    FetchContentDetail,
    FetchContentRequest,
    FetchOthersRequest,
    FetchOverviewRequest,
    FetchRequest,
    FetchRequestBase,
    FetchSubpagesRequest,
    ResearchRequest,
    ResearchSearchMode,
    SearchRequest,
)
from serpsage.app.response import (
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
    "FetchRequestBase",
    "SearchRequest",
    "SearchResponse",
    "ResearchSearchMode",
    "ResearchRequest",
    "ResearchResponse",
    "load_settings",
]
