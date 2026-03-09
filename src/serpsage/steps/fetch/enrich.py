from __future__ import annotations

from typing_extensions import override

import anyio

from serpsage.dependencies import Inject
from serpsage.models.steps.fetch import FetchStepContext
from serpsage.steps.base import StepBase
from serpsage.steps.fetch.overview import FetchOverviewStep
from serpsage.steps.fetch.subpages import FetchSubpageStep


class FetchParallelEnrichStep(StepBase[FetchStepContext]):
    overview_step: FetchOverviewStep = Inject()
    subpages_step: FetchSubpageStep = Inject()

    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        if ctx.error.failed:
            return ctx

        async def _run_overview() -> None:
            await self.overview_step.run(ctx)

        async def _run_subpages() -> None:
            await self.subpages_step.run(ctx)

        async with anyio.create_task_group() as tg:
            tg.start_soon(_run_overview)
            tg.start_soon(_run_subpages)
        return ctx


__all__ = ["FetchParallelEnrichStep"]
