from __future__ import annotations

from typing import TYPE_CHECKING

import anyio

from serpsage.app.response import FetchResponse, SearchResponse
from serpsage.core.runtime import Overrides
from serpsage.core.workunit import WorkUnit
from serpsage.models.pipeline import (
    FetchStepContext,
    FetchStepOthers,
    SearchStepContext,
)
from serpsage.pipeline.base import RunnerBase

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
        search_runner: RunnerBase[SearchStepContext],
        fetch_runner: RunnerBase[FetchStepContext],
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
            contexts: list[FetchStepContext] = [
                FetchStepContext(
                    settings=self.settings,
                    request=req,
                    url=url,
                    url_index=idx,
                    others=FetchStepOthers(
                        crawl_mode=req.crawl_mode,
                        crawl_timeout_s=float(req.crawl_timeout or 0.0),
                        max_links=req.others.max_links,
                        max_image_links=req.others.max_image_links,
                    ),
                )
                for idx, url in enumerate(req.urls)
            ]
            if contexts:
                max_parallel = min(
                    max(1, int(self.settings.fetch.concurrency.global_limit)),
                    max(1, len(contexts)),
                )
                sem = anyio.Semaphore(max_parallel)

                async def run_one(index: int, item: FetchStepContext) -> None:
                    async with sem:
                        contexts[index] = await self._fetch_runner.run(item)

                async with anyio.create_task_group() as tg:
                    for i, item in enumerate(contexts):
                        tg.start_soon(run_one, i, item)

            results = [
                ctx.result
                for ctx in contexts
                if not ctx.fatal and ctx.result is not None
            ]
            errors = [err for ctx in contexts for err in ctx.errors]
            return FetchResponse(
                results=results,
                errors=errors,
                telemetry=self.telemetry.summary(),
            )


__all__ = ["Engine"]
