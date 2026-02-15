from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.request import FetchContentRequest
from serpsage.models.errors import AppError
from serpsage.models.extract import ExtractContentOptions
from serpsage.models.pipeline import FetchStepContext
from serpsage.pipeline.step import PipelineStep
from serpsage.text.normalize import clean_whitespace
from serpsage.text.tokenize import tokenize_for_query

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime


class FetchPrepareStep(PipelineStep[FetchStepContext]):
    span_name = "step.fetch_prepare"

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        url = clean_whitespace(ctx.url or "")
        if not url:
            ctx.fatal = True
            ctx.errors.append(
                AppError(
                    code="fetch_load_failed",
                    message="empty url",
                    details={
                        "url": ctx.url,
                        "url_index": ctx.url_index,
                        "stage": "prepare",
                        "fatal": True,
                        "crawl_mode": ctx.others_runtime.crawl_mode,
                    },
                )
            )
            return ctx

        abstracts_request = ctx.request.abstracts
        if abstracts_request is not None:
            query = clean_whitespace(abstracts_request.query or "")
            if not query:
                ctx.fatal = True
                ctx.errors.append(
                    AppError(
                        code="fetch_abstract_rank_failed",
                        message="abstracts.query must not be empty",
                        details={
                            "url": url,
                            "url_index": ctx.url_index,
                            "stage": "prepare",
                            "fatal": True,
                            "crawl_mode": ctx.others_runtime.crawl_mode,
                        },
                    )
                )
                return ctx
            abstracts_request = abstracts_request.model_copy(update={"query": query})

        overview_request = ctx.request.overview
        if overview_request is not None:
            overview_request = overview_request.model_copy(
                update={"query": clean_whitespace(overview_request.query or "")}
            )
            if not overview_request.query:
                ctx.fatal = True
                ctx.errors.append(
                    AppError(
                        code="fetch_overview_failed",
                        message="overview.query must not be empty",
                        details={
                            "url": url,
                            "url_index": ctx.url_index,
                            "stage": "prepare",
                            "fatal": True,
                            "crawl_mode": ctx.others_runtime.crawl_mode,
                        },
                    )
                )
                return ctx

        raw_content = ctx.request.content
        content_request: FetchContentRequest
        return_content: bool
        if isinstance(raw_content, bool):
            return_content = bool(raw_content)
            content_request = FetchContentRequest()
        else:
            return_content = True
            content_request = raw_content

        content_options = ExtractContentOptions(
            depth=content_request.depth,
            include_html_tags=bool(content_request.include_html_tags),
            include_tags=list(content_request.include_tags),
            exclude_tags=list(content_request.exclude_tags),
        )
        ctx.url = url

        if abstracts_request is not None:
            abstract_query_tokens = tokenize_for_query(abstracts_request.query)
        else:
            abstract_query_tokens = []

        ctx.return_content = bool(return_content)
        ctx.content_request = content_request
        ctx.content_options = content_options
        ctx.abstracts_request = abstracts_request
        ctx.overview_request = overview_request
        ctx.abstract_query_tokens = abstract_query_tokens

        span.set_attr("has_content_output", bool(return_content))
        span.set_attr("has_abstracts", bool(abstracts_request is not None))
        span.set_attr("has_overview", bool(overview_request is not None))
        span.set_attr("content_depth", str(content_request.depth))
        span.set_attr("crawl_mode", str(ctx.others_runtime.crawl_mode))
        span.set_attr("crawl_timeout_s", float(ctx.others_runtime.crawl_timeout_s))
        span.set_attr("url_index", int(ctx.url_index))
        return ctx


__all__ = ["FetchPrepareStep"]
