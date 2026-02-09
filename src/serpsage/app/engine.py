from __future__ import annotations

from typing import Any

from serpsage.app.container import Container, Overrides
from serpsage.app.request import SearchRequest
from serpsage.app.response import PageChunk, ResultItem, SearchResponse
from serpsage.pipeline.dedupe import DedupeStep
from serpsage.pipeline.enrich import EnrichStep
from serpsage.pipeline.filter import FilterStep
from serpsage.pipeline.normalize import NormalizeStep
from serpsage.pipeline.overview import OverviewStep
from serpsage.pipeline.rank import RankStep
from serpsage.pipeline.rerank import RerankStep
from serpsage.pipeline.search import SearchStep
from serpsage.pipeline.steps import StepContext
from serpsage.settings.models import AppSettings
from serpsage.text.normalize import clean_whitespace


class Engine:
    """Async-only search engine (pipeline orchestrator)."""

    def __init__(self, settings: AppSettings, *, overrides: Overrides | None = None) -> None:
        self.settings = settings
        self._container = Container(settings=settings, overrides=overrides)
        self._closed = False

        self._steps = [
            SearchStep(self._container),
            NormalizeStep(self._container),
            FilterStep(self._container),
            DedupeStep(self._container),
            RankStep(self._container),
            EnrichStep(self._container),
            RerankStep(self._container),
        ]
        self._overview_step = OverviewStep(self._container)

    async def __aenter__(self) -> "Engine":
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:  # noqa: ANN001
        await self.aclose()

    async def aclose(self) -> None:
        if self._closed:
            return
        self._closed = True
        await self._container.aclose()

    async def run(self, req: SearchRequest) -> SearchResponse:
        span = self._container.telemetry.start_span("engine.run")
        try:
            query = clean_whitespace(req.query or "")
            depth = req.depth or "simple"
            max_results = int(req.max_results) if req.max_results is not None else int(self.settings.pipeline.max_results)

            req = req.model_copy(update={"query": query, "depth": depth, "max_results": max_results})

            ctx = StepContext(settings=self.settings, request=req)
            for step in self._steps:
                ctx = await step.run(ctx)

            # Global score floor and max_results cap.
            min_score = float(self.settings.pipeline.min_score)
            ctx.results = [r for r in ctx.results if float(r.score) > 0.0 and float(r.score) >= min_score]
            ctx.results = ctx.results[:max_results]

            # Assign stable IDs for citations, then run overview.
            _assign_ids(ctx.results)
            ctx = await self._overview_step.run(ctx)

            telemetry_summary: dict[str, Any] | None = None
            if hasattr(self._container.telemetry, "summary"):
                telemetry_summary = getattr(self._container.telemetry, "summary")()

            return SearchResponse(
                query=query,
                depth=depth,
                results=ctx.results,
                overview=ctx.overview,
                errors=ctx.errors,
                telemetry=telemetry_summary,
            )
        finally:
            span.end()


def _assign_ids(results: list[ResultItem]) -> None:
    for i, r in enumerate(results, 1):
        sid = f"S{i}"
        r.source_id = sid
        if r.page and r.page.chunks:
            for j, ch in enumerate(r.page.chunks, 1):
                ch.chunk_id = ch.chunk_id or f"{sid}:C{j}"


__all__ = ["Engine"]

