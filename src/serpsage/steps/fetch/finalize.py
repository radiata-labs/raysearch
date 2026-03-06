from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.response import FetchResultItem
from serpsage.components.extract.html.postprocess import finalize_markdown
from serpsage.models.pipeline import FetchStepContext
from serpsage.steps.base import StepBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


class FetchFinalizeStep(StepBase[FetchStepContext]):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        if ctx.fatal:
            return ctx
        if ctx.artifacts.extracted is None:
            ctx.fatal = True
            ctx.error_tag = "SOURCE_NOT_AVAILABLE"
            ctx.error_detail = "missing extracted content"
            await self.emit_tracking_event(
                event_name="fetch.finalize.error",
                request_id=ctx.request_id,
                stage="finalize",
                status="error",
                error_code="fetch_extract_failed",
                attrs={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "fatal": True,
                    "crawl_mode": str(ctx.runtime.crawl_mode),
                    "message": "missing extracted content",
                },
            )
            return ctx
        markdown = (
            ctx.artifacts.extracted.markdown
            if ctx.resolved.content_request.include_markdown_links
            else ctx.artifacts.extracted.md_for_abstract
        )
        max_chars = ctx.resolved.content_request.max_chars
        if max_chars is not None and max_chars > 0:
            markdown = finalize_markdown(markdown=markdown, max_chars=max_chars)
        content = markdown if ctx.resolved.return_content else ""
        abstracts = [
            str(item.text) for item in list(ctx.artifacts.scored_abstracts or [])
        ]
        abstract_scores = [
            float(item.score) for item in list(ctx.artifacts.scored_abstracts or [])
        ]
        others_result = {}
        if not ctx.enable_others_and_subpages:
            subpages_result = []
        else:
            if ctx.request.others is not None:
                others_result = {"others": ctx.output.others}
            subpages_result = list(ctx.subpages.results)
        ctx.output.result = FetchResultItem(
            url=ctx.url,
            title=str(ctx.artifacts.extracted.title or ""),
            content=content,
            abstracts=abstracts,
            abstract_scores=abstract_scores,
            overview=(
                ""
                if ctx.artifacts.overview_output is None
                else ctx.artifacts.overview_output
            ),
            subpages=subpages_result,
            **others_result,
        )
        return ctx


__all__ = ["FetchFinalizeStep"]
