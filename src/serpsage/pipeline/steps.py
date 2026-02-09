from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Protocol

from serpsage.contracts.base import WorkUnit
from serpsage.contracts.errors import AppError

if TYPE_CHECKING:
    from serpsage.app.request import SearchRequest
    from serpsage.app.response import OverviewResult, ResultItem
    from serpsage.app.runtime import CoreRuntime
    from serpsage.contracts.protocols import Span
    from serpsage.settings.models import AppSettings, ProfileSettings


@dataclass(slots=True)
class StepContext:
    settings: AppSettings
    request: SearchRequest
    raw_results: list[dict[str, Any]] = field(default_factory=list)
    results: list[ResultItem] = field(default_factory=list)
    profile_name: str = ""
    profile: ProfileSettings | None = None
    query_tokens: list[str] | None = None
    intent_tokens: list[str] | None = None
    dedupe_comparisons: int = 0
    overview: OverviewResult | None = None
    errors: list[AppError] = field(default_factory=list)


class Step(Protocol):
    async def run(self, ctx: StepContext) -> StepContext: ...


class StepBase(WorkUnit, ABC):
    """Base class for pipeline steps.

    Provides a consistent span name and a generic error boundary. Steps that need
    richer error codes/messages should catch inside `run_inner`.
    """

    span_name: str | None = None

    def __init__(self, *, rt: CoreRuntime) -> None:
        super().__init__(rt=rt)

    async def run(self, ctx: StepContext) -> StepContext:
        name = (
            self.span_name or f"step.{type(self).__name__.replace('Step', '').lower()}"
        )
        with self.span(name) as sp:
            try:
                return await self.run_inner(ctx, span=sp)
            except Exception as exc:  # noqa: BLE001
                ctx.errors.append(
                    AppError(
                        code="step_failed",
                        message=str(exc),
                        details={
                            "step": type(self).__name__,
                            "type": type(exc).__name__,
                        },
                    )
                )
                return ctx

    @abstractmethod
    async def run_inner(
        self, ctx: StepContext, *, span: Span
    ) -> StepContext:  # pragma: no cover
        raise NotImplementedError


__all__ = ["Step", "StepBase", "StepContext"]
