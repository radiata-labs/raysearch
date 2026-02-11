from __future__ import annotations

from pydantic import Field

from serpsage.app.request import SearchRequest
from serpsage.app.response import OverviewResult, ResultItem
from serpsage.core.model_base import MutableModel
from serpsage.models.errors import AppError
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
    dedupe_comparisons: int = 0
    overview: OverviewResult | None = None
    errors: list[AppError] = Field(default_factory=list)


__all__ = ["BaseStepContext", "SearchStepContext"]
