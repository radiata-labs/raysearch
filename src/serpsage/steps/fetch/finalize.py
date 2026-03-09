from __future__ import annotations

from typing_extensions import override

from serpsage.models.app.response import FetchResultItem
from serpsage.models.steps.fetch import FetchStepContext
from serpsage.steps.base import StepBase


class FetchFinalizeStep(StepBase[FetchStepContext]):
    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        if ctx.error.failed:
            return ctx
        if ctx.page.doc is None:
            ctx.error.failed = True
            ctx.error.tag = "SOURCE_NOT_AVAILABLE"
            ctx.error.detail = "missing extracted content"
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
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "message": "missing extracted content",
                },
            )
            return ctx
        content = str(ctx.page.doc.content.output_markdown or "")
        abstracts = [
            str(item.text) for item in list(ctx.analysis.abstracts.ranked or [])
        ]
        abstract_scores = [
            float(item.score) for item in list(ctx.analysis.abstracts.ranked or [])
        ]
        others_result = {}
        if not ctx.related.enabled:
            subpages_result = []
        else:
            if ctx.request.others is not None:
                others_result = {"others": ctx.related.others}
            subpages_result = [
                item.result
                for item in list(ctx.related.subpages.items or [])
                if item.result is not None
            ]
        ctx.result = FetchResultItem(
            url=ctx.url,
            title=str(ctx.page.doc.meta.title or ""),
            published_date=str(ctx.page.doc.meta.published_date or ""),
            author=str(ctx.page.doc.meta.author or ""),
            image=str(ctx.page.doc.meta.image or ""),
            favicon=str(ctx.page.doc.meta.favicon or ""),
            content=content,
            abstracts=abstracts,
            abstract_scores=abstract_scores,
            overview=(
                ""
                if ctx.analysis.overview.output is None
                else ctx.analysis.overview.output
            ),
            subpages=subpages_result,
            **others_result,
        )
        return ctx


__all__ = ["FetchFinalizeStep"]
