"""
RaySearch 3.0 (async-only).
Public API:
- Engine: async search pipeline orchestrator
- create_api_app: build the personal FastAPI service on top of Engine
- load_settings: load raw settings data from YAML/JSON/env
- SearchRequest/SearchResponse + FetchRequest/FetchResponse + AnswerRequest/AnswerResponse
  request/response models
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from raysearch.api import create_api_app
    from raysearch.app.engine import Engine
    from raysearch.core.overrides import Overrides
from raysearch.models.app.request import (
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
from raysearch.models.app.response import (
    AnswerCitation,
    AnswerResponse,
    FetchResponse,
    FetchSubpagesResult,
    ResearchResponse,
    ResearchTaskListResponse,
    ResearchTaskResponse,
    ResearchTaskStatus,
    SearchResponse,
)
from raysearch.settings.load import load_settings
from raysearch.settings.models import AppSettings


def __getattr__(name: str) -> object:
    if name == "Engine":
        from raysearch.app.engine import Engine

        return Engine
    if name == "Overrides":
        from raysearch.core.overrides import Overrides

        return Overrides
    if name == "create_api_app":
        from raysearch.api import create_api_app

        return create_api_app
    raise AttributeError(name)


__all__ = [
    "AppSettings",
    "AnswerRequest",
    "AnswerCitation",
    "AnswerResponse",
    "CrawlMode",
    "create_api_app",
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
    "ResearchTaskStatus",
    "ResearchTaskResponse",
    "ResearchTaskListResponse",
    "Overrides",
    "load_settings",
]
