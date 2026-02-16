from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic, TypeVar

from serpsage.core.workunit import WorkUnit
from serpsage.models.errors import AppError
from serpsage.telemetry.base import SpanBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.models.pipeline import BaseStepContext

    TContext = TypeVar("TContext", bound=BaseStepContext)
else:
    TContext = TypeVar("TContext")


class StepBase(WorkUnit, ABC, Generic[TContext]):
    span_name: str | None = None

    async def run(self, ctx: TContext) -> TContext:
        name = (
            self.span_name or f"step.{type(self).__name__.replace('Step', '').lower()}"
        )
        with self.span(name) as sp:
            try:
                return await self.run_inner(ctx, span=sp)
            except Exception as exc:  # noqa: BLE001
                sp.set_attr("error", True)
                sp.add_event(
                    "step_failed",
                    error_type=type(exc).__name__,
                    message=str(exc),
                )
                errors = getattr(ctx, "errors", None)
                if isinstance(errors, list):
                    errors.append(
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
    async def run_inner(self, ctx: TContext, *, span: SpanBase) -> TContext:
        raise NotImplementedError


class RunnerBase(WorkUnit, Generic[TContext]):
    def __init__(self, *, rt: Runtime, steps: list[StepBase[TContext]]) -> None:
        super().__init__(rt=rt)
        self._steps = list(steps)
        self.bind_deps(*steps)

    async def run(self, ctx: TContext) -> TContext:
        for step in self._steps:
            ctx = await step.run(ctx)
        return ctx


__all__ = ["StepBase"]
