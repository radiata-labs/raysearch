from __future__ import annotations

from abc import ABC, abstractmethod

from serpsage.contracts.lifecycle import SpanBase
from serpsage.contracts.services import PipelineStepBase
from serpsage.models.errors import AppError
from serpsage.models.pipeline import SearchStepContext


class StepBase(PipelineStepBase, ABC):
    span_name: str | None = None

    async def run(self, ctx: SearchStepContext) -> SearchStepContext:
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
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        raise NotImplementedError


__all__ = ["StepBase"]
