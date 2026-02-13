from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.app.response import FetchResponse, SearchResponse
from serpsage.contracts.services import PipelineRunnerBase
from serpsage.core.runtime import Overrides
from serpsage.core.workunit import WorkUnit
from serpsage.models.pipeline import FetchStepContext, SearchStepContext

if TYPE_CHECKING:
    from serpsage.app.request import FetchRequest, SearchRequest
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import AppSettings


class Engine(WorkUnit):
    """Async-only engine with dual paths: search + fetch."""

    def __init__(
        self,
        *,
        rt: Runtime,
        search_runner: PipelineRunnerBase[SearchStepContext],
        fetch_runner: PipelineRunnerBase[FetchStepContext],
    ) -> None:
        super().__init__(rt=rt)
        self._search_runner = search_runner
        self._fetch_runner = fetch_runner
        self.bind_deps(search_runner, fetch_runner)

    @classmethod
    def from_settings(
        cls, settings: AppSettings, *, overrides: Overrides | None = None
    ) -> Engine:
        from serpsage.app.bootstrap import build_engine  # noqa: PLC0415

        return build_engine(settings=settings, overrides=overrides)

    async def search(self, req: SearchRequest) -> SearchResponse:
        await self.ainit()
        with self.span("engine.search"):
            ctx = SearchStepContext(settings=self.settings, request=req)
            ctx = await self._search_runner.run(ctx)

            return SearchResponse(
                query=ctx.request.query,
                depth=ctx.request.depth,
                results=ctx.results,
                overview=ctx.overview,
                errors=ctx.errors,
                telemetry=self.telemetry.summary(),
            )

    async def fetch(self, req: FetchRequest) -> FetchResponse:
        await self.ainit()
        with self.span("engine.fetch"):
            ctx = FetchStepContext(settings=self.settings, request=req)
            ctx = await self._fetch_runner.run(ctx)

            return FetchResponse(
                url=ctx.request.url,
                query=ctx.request.query,
                page=ctx.page,
                overview=ctx.overview,
                errors=ctx.errors,
                telemetry=self.telemetry.summary(),
            )


__all__ = ["Engine"]
