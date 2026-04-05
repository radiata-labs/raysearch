from __future__ import annotations

from typing_extensions import override

import anyio

from raysearch.dependencies import Depends
from raysearch.models.steps.fetch import FetchStepContext
from raysearch.steps.base import StepBase
from raysearch.steps.fetch.overview import FetchOverviewStep
from raysearch.steps.fetch.subpages import FetchSubpageStep


class FetchParallelEnrichStep(StepBase[FetchStepContext]):
    overview_step: FetchOverviewStep = Depends()
    subpages_step: FetchSubpageStep = Depends()

    @override
    async def should_run(self, ctx: FetchStepContext) -> bool:
        """Execute unless failed (runs overview and subpages in parallel)."""
        return not ctx.error.failed

    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        # Pre-condition: should_run() verified no prior error

        async def _run_overview() -> None:
            await self.overview_step.run(ctx)

        async def _run_subpages() -> None:
            await self.subpages_step.run(ctx)

        async with anyio.create_task_group() as tg:
            tg.start_soon(_run_overview)
            tg.start_soon(_run_subpages)
        return ctx


__all__ = ["FetchParallelEnrichStep"]
