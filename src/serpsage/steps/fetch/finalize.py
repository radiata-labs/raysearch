from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.response import FetchResultItem
from serpsage.components.extract.markdown.postprocess import finalize_markdown
from serpsage.models.errors import AppError
from serpsage.models.pipeline import FetchStepContext
from serpsage.steps.base import StepBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class FetchFinalizeStep(StepBase[FetchStepContext]):
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
                        "crawl_mode": ctx.others.crawl_mode,
                    },
                )
            )
            return ctx

        markdown = ctx.extracted.markdown
        max_chars = ctx.content_request.max_chars
        if max_chars is not None and max_chars > 0:
            markdown = finalize_markdown(markdown=markdown, max_chars=max_chars)
        content = markdown if ctx.return_content else ""

        abstracts = [str(item.text) for item in list(ctx.scored_abstracts or [])]
        abstract_scores = [
            float(item.score) for item in list(ctx.scored_abstracts or [])
        ]
        others_result = {}
        if not ctx.enable_others_and_subpages:
            subpages_result = []
        else:
            if ctx.request.others is not None:
                others_result = {"others": ctx.others_result}
            subpages_result = list(ctx.subpages_result)

        ctx.result = FetchResultItem(
            url=ctx.url,
            title=str(ctx.extracted.title or ""),
            content=content,
            abstracts=abstracts,
            abstract_scores=abstract_scores,
            overview="" if ctx.overview_output is None else ctx.overview_output,
            subpages=subpages_result,
            **others_result,
        )

        span.set_attr("has_result", True)
        span.set_attr("abstracts_count", int(len(abstracts)))
        span.set_attr(
            "links_count",
            int(len(others_result["others"].links)) if others_result is not None else 0,
        )
        span.set_attr(
            "image_links_count",
            int(len(others_result["others"].image_links))
            if others_result is not None
            else 0,
        )
        span.set_attr("subpages_count", int(len(subpages_result)))
        span.set_attr("has_overview", bool(ctx.overview_output is not None))
        span.set_attr("has_content_output", bool(ctx.return_content))
        return ctx


__all__ = ["FetchFinalizeStep"]
