from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Generic
from typing_extensions import override

from serpsage.contracts.lifecycle import SpanBase
from serpsage.contracts.services import PipelineStepBase, TContext
from serpsage.models.errors import AppError


class PipelineStep(PipelineStepBase[TContext], ABC, Generic[TContext]):
    span_name: str | None = None

    @override
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


__all__ = ["PipelineStep"]
