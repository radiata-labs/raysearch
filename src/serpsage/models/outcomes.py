from __future__ import annotations

from pydantic import Field

from serpsage.app.response import ResultItem
from serpsage.core.model_base import FrozenModel
from serpsage.settings.models import ProfileSettings


class FilterOutcome(FrozenModel):
    profile_name: str
    profile: ProfileSettings
    query_tokens: list[str] = Field(default_factory=list)
    results: list[ResultItem] = Field(default_factory=list)


__all__ = ["FilterOutcome"]
