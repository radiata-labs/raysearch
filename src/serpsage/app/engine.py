from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.response import ResultItem, SearchResponse
from serpsage.contracts.services import PipelineStepBase
from serpsage.core.runtime import ComponentOverrides
from serpsage.core.workunit import WorkUnit
from serpsage.models.pipeline import SearchStepContext
from serpsage.text.normalize import clean_whitespace

if TYPE_CHECKING:
    from collections.abc import Awaitable, Callable

    from serpsage.app.request import SearchRequest
    from serpsage.core.runtime import CoreRuntime
    from serpsage.settings.models import AppSettings


class Engine(WorkUnit):
    """Async-only search engine (pipeline orchestrator)."""

    def __init__(
        self,
        *,
        rt: CoreRuntime,
        steps: list[PipelineStepBase],
        overview_step: PipelineStepBase,
        ainit_hook: Callable[[], Awaitable[None]],
        aclose_hook: Callable[[], Awaitable[None]],
    ) -> None:
        super().__init__(rt=rt)
        self._inited = False
        self._closed = False
        self._steps = steps
        self._overview_step = overview_step
        self._ainit_hook = ainit_hook
        self._aclose_hook = aclose_hook

    @classmethod
    def from_settings(
        cls, settings: AppSettings, *, overrides: ComponentOverrides | None = None
    ) -> Engine:
        # Lazy import to avoid a bootstrap <-> engine import cycle.
        from serpsage.app.bootstrap import build_engine  # noqa: PLC0415

        return build_engine(settings=settings, overrides=overrides)

    @override
    async def ainit(self) -> None:
        if self._inited:
            return
        self._inited = True
        await self._ainit_hook()

    @override
    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._aclose_hook()

    async def run(self, req: SearchRequest) -> SearchResponse:
        await self.ainit()
        with self.span("engine.run") as _:
            query = clean_whitespace(req.query or "")
            depth = req.depth or "simple"
            max_results = (
                int(req.max_results)
                if req.max_results is not None
                else int(self.settings.pipeline.max_results)
            )

            req = req.model_copy(
                update={"query": query, "depth": depth, "max_results": max_results}
            )

            ctx = SearchStepContext(settings=self.settings, request=req)
            for step in self._steps:
                ctx = await step.run(ctx)

            # Global score floor and max_results cap.
            min_score = float(self.settings.pipeline.min_score)
            ctx.results = [
                r
                for r in ctx.results
                if float(r.score) > 0.0 and float(r.score) >= min_score
            ]
            ctx.results = ctx.results[:max_results]

            # Assign stable IDs for citations, then run overview.
            _assign_ids(ctx.results)
            ctx = await self._overview_step.run(ctx)

            telemetry_summary = self.telemetry.summary()

            return SearchResponse(
                query=query,
                depth=depth,
                results=ctx.results,
                overview=ctx.overview,
                errors=ctx.errors,
                telemetry=telemetry_summary,
            )


def _assign_ids(results: list[ResultItem]) -> None:
    for i, r in enumerate(results, 1):
        sid = f"S{i}"
        r.source_id = sid
        if r.page and r.page.chunks:
            for j, ch in enumerate(r.page.chunks, 1):
                ch.chunk_id = ch.chunk_id or f"{sid}:C{j}"


__all__ = ["Engine"]
