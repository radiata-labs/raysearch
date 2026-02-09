from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Protocol

from serpsage.app.request import SearchRequest
from serpsage.app.response import OverviewResult, ResultItem
from serpsage.contracts.errors import AppError
from serpsage.settings.models import AppSettings, ProfileSettings


@dataclass
class StepContext:
    settings: AppSettings
    request: SearchRequest
    raw_results: list[dict[str, Any]] = field(default_factory=list)
    results: list[ResultItem] = field(default_factory=list)
    profile_name: str = ""
    profile: ProfileSettings | None = None
    overview: OverviewResult | None = None
    errors: list[AppError] = field(default_factory=list)
    scratch: dict[str, Any] = field(default_factory=dict)


class Step(Protocol):
    async def run(self, ctx: StepContext) -> StepContext: ...


__all__ = ["Step", "StepContext"]

