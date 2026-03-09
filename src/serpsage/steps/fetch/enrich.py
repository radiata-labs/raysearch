from __future__ import annotations

from typing_extensions import override

import anyio

from serpsage.app.tokens import CHILD_FETCH_RUNNER
from serpsage.components.cache.base import CacheBase
from serpsage.components.llm.base import LLMClientBase
from serpsage.components.rank.base import RankerBase
from serpsage.core.runtime import Runtime
from serpsage.dependencies import Inject
from serpsage.models.steps.fetch import FetchStepContext
from serpsage.steps.base import RunnerBase, StepBase
from serpsage.steps.fetch.overview import FetchOverviewStep
from serpsage.steps.fetch.subpages import FetchSubpageStep


class FetchParallelEnrichStep(StepBase[FetchStepContext]):
    def __init__(
        self,
        *,
        rt: Runtime = Inject(),
        llm: LLMClientBase = Inject(),
        cache: CacheBase = Inject(),
        fetch_runner: RunnerBase[FetchStepContext] = Inject(CHILD_FETCH_RUNNER),
        ranker: RankerBase = Inject(),
    ) -> None:
        super().__init__(rt=rt)
        self._overview_step = FetchOverviewStep(rt=rt, llm=llm, cache=cache)
        self._subpages_step = FetchSubpageStep(
            rt=rt,
            fetch_runner=fetch_runner,
            ranker=ranker,
        )
        self.bind_deps(self._overview_step, self._subpages_step)

    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        if ctx.error.failed:
            return ctx

        async def _run_overview() -> None:
            await self._overview_step.run(ctx)

        async def _run_subpages() -> None:
            await self._subpages_step.run(ctx)

        async with anyio.create_task_group() as tg:
            tg.start_soon(_run_overview)
            tg.start_soon(_run_subpages)
        return ctx


__all__ = ["FetchParallelEnrichStep"]
