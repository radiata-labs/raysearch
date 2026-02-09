from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.contracts.base import WorkUnit
from serpsage.contracts.errors import AppError

if TYPE_CHECKING:
    from serpsage.contracts.protocols import LLMClient
    from serpsage.domain.overview import OverviewBuilder
    from serpsage.pipeline.steps import StepContext


class OverviewStep(WorkUnit):
    def __init__(self, *, rt, llm: LLMClient, builder: OverviewBuilder) -> None:  # noqa: ANN001
        super().__init__(rt=rt)
        self._llm = llm
        self._builder = builder

    async def run(self, ctx: StepContext) -> StepContext:
        with self.span("step.overview"):
            enabled = self.settings.overview.enabled
            if ctx.request.overview is not None:
                enabled = bool(ctx.request.overview)
            if not enabled:
                return ctx
            if not ctx.results:
                return ctx
            if not self.settings.overview.llm.api_key:
                ctx.errors.append(
                    AppError(
                        code="overview_skipped",
                        message="LLM api_key not configured; skipping overview",
                        details={},
                    )
                )
                return ctx

            llm_cfg = self.settings.overview.llm
            messages = self._builder.build_messages(
                query=ctx.request.query, results=ctx.results
            )
            schema = self._builder.schema()
            try:
                data = await self._llm.chat_json(
                    model=llm_cfg.model,
                    messages=messages,
                    schema=schema,
                    timeout_s=float(llm_cfg.timeout_s),
                )
                ctx.overview = self._builder.parse(data)
            except Exception as exc:  # noqa: BLE001
                ctx.errors.append(
                    AppError(code="overview_failed", message=str(exc), details={})
                )
            return ctx


__all__ = ["OverviewStep"]
