"""
SerpSage 3.0 (async-only).

Public API:
- Engine: async search pipeline orchestrator
- load_settings: load AppSettings from YAML/JSON/env
- SearchRequest/SearchResponse: request/response models
"""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serpsage.app.engine import Engine


from serpsage.app.request import SearchRequest
from serpsage.app.response import SearchResponse
from serpsage.settings.load import load_settings
from serpsage.settings.models import AppSettings


def __getattr__(name: str):
    if name == "Engine":
        from serpsage.app.engine import Engine

        return Engine
    raise AttributeError(name)


__all__ = [
    "AppSettings",
    "Engine",
    "SearchRequest",
    "SearchResponse",
    "load_settings",
]
