from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.response import FetchResultItem
from serpsage.components.extract.markdown.postprocess import finalize_markdown
from serpsage.models.errors import AppError
from serpsage.models.pipeline import FetchStepContext
from serpsage.pipeline.step import PipelineStep

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime


class FetchFinalizeStep(PipelineStep[FetchStepContext]):
    span_name = "step.fetch_finalize"

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        if ctx.fatal:
            return ctx
        if ctx.extracted is None:
            ctx.fatal = True
            ctx.errors.append(
                AppError(
                    code="fetch_extract_failed",
                    message="missing extracted content",
                    details={
                        "url": ctx.url,
                        "url_index": ctx.url_index,
                        "stage": "finalize",
                        "fatal": True,
                        "crawl_mode": ctx.runtime.crawl_mode,
                    },
                )
            )
            return ctx

        markdown = ctx.extracted.markdown
        max_chars = ctx.content_request.max_chars
        if max_chars is not None and max_chars > 0:
            markdown = finalize_markdown(markdown=markdown, max_chars=max_chars)
        content = markdown if ctx.return_content else ""

        chunks = [str(item.text) for item in list(ctx.scored_chunks or [])]
        chunk_scores = [float(item.score) for item in list(ctx.scored_chunks or [])]
        result = FetchResultItem(
            url=ctx.url,
            title=str(ctx.extracted.title or ""),
            content=content,
            chunks=chunks,
            chunk_scores=chunk_scores,
            links=list(ctx.links or []),
            overview=ctx.overview,
        )
        ctx.result = result

        span.set_attr("has_result", True)
        span.set_attr("chunks_count", int(len(chunks)))
        span.set_attr("links_count", int(len(ctx.links or [])))
        span.set_attr("has_overview", bool(ctx.overview is not None))
        span.set_attr("has_content_output", bool(ctx.return_content))
        return ctx


__all__ = ["FetchFinalizeStep"]
