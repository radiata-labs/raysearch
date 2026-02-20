from __future__ import annotations

import uuid
from typing import TYPE_CHECKING

from serpsage.app.response import AnswerResponse, FetchResponse, SearchResponse
from serpsage.core.runtime import Overrides
from serpsage.core.workunit import WorkUnit
from serpsage.models.pipeline import (
    AnswerStepContext,
    FetchRuntimeConfig,
    FetchStepContext,
    SearchStepContext,
)
from serpsage.steps.base import RunnerBase

if TYPE_CHECKING:
    from serpsage.app.request import AnswerRequest, FetchRequest, SearchRequest
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import AppSettings


class Engine(WorkUnit):
    """Async-only engine with search/fetch/answer paths."""

    def __init__(
        self,
        *,
        rt: Runtime,
        search_runner: RunnerBase[SearchStepContext],
        fetch_runner: RunnerBase[FetchStepContext],
        answer_runner: RunnerBase[AnswerStepContext],
    ) -> None:
        super().__init__(rt=rt)
        self._search_runner = search_runner
        self._fetch_runner = fetch_runner
        self._answer_runner = answer_runner
        self.bind_deps(search_runner, fetch_runner, answer_runner)

    @classmethod
    def from_settings(
        cls, settings: AppSettings, *, overrides: Overrides | None = None
    ) -> Engine:
        from serpsage.app.bootstrap import build_engine  # noqa: PLC0415

        return build_engine(settings=settings, overrides=overrides)

    async def search(self, req: SearchRequest) -> SearchResponse:
        with self.span("engine.search"):
            request_id = uuid.uuid4().hex
            ctx = SearchStepContext(
                settings=self.settings,
                request=req,
                request_id=request_id,
            )
            ctx = await self._search_runner.run(ctx)
            return SearchResponse(
                request_id=request_id,
                search_depth=ctx.request.depth,
                results=ctx.output.results,
                errors=ctx.errors,
                telemetry=self.telemetry.summary(),
            )

    async def fetch(self, req: FetchRequest) -> FetchResponse:
        with self.span("engine.fetch"):
            request_id = uuid.uuid4().hex
            contexts: list[FetchStepContext] = [
                FetchStepContext(
                    settings=self.settings,
                    request=req,
                    request_id=request_id,
                    url=url,
                    url_index=idx,
                    enable_others_and_subpages=True,
                    runtime=FetchRuntimeConfig(
                        crawl_mode=req.crawl_mode,
                        crawl_timeout_s=float(req.crawl_timeout or 0.0),
                        max_links=(
                            req.others.max_links if req.others is not None else None
                        ),
                        max_image_links=(
                            req.others.max_image_links
                            if req.others is not None
                            else None
                        ),
                    ),
                )
                for idx, url in enumerate(req.urls)
            ]
            if contexts:
                contexts = await self._fetch_runner.run_batch(contexts)

            results = [
                ctx.output.result
                for ctx in contexts
                if not ctx.fatal and ctx.output.result is not None
            ]
            errors = [err for ctx in contexts for err in ctx.errors]
            return FetchResponse(
                request_id=request_id,
                results=results,
                errors=errors,
                telemetry=self.telemetry.summary(),
            )

    async def answer(self, req: AnswerRequest) -> AnswerResponse:
        with self.span("engine.answer"):
            request_id = uuid.uuid4().hex
            ctx = AnswerStepContext(
                settings=self.settings,
                request=req,
                request_id=request_id,
            )
            ctx = await self._answer_runner.run(ctx)
            return AnswerResponse(
                request_id=request_id,
                answer=ctx.output.answers,
                citations=ctx.output.citations,
                errors=ctx.errors,
                telemetry=self.telemetry.summary(),
            )


__all__ = ["Engine"]
