from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.request import FetchContentRequest
from serpsage.models.extract import ExtractContentOptions
from serpsage.models.pipeline import FetchStepContext
from serpsage.pipeline.step import PipelineStep
from serpsage.text.normalize import clean_whitespace
from serpsage.text.tokenize import tokenize_for_query
from serpsage.text.utils import extract_intent_tokens

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
        url = clean_whitespace(ctx.request.url or "")
        chunks_request = ctx.request.chunks
        if chunks_request is not None:
            query = clean_whitespace(chunks_request.query or "")
            if not query:
                raise ValueError("chunks.query must not be empty")
            chunks_request = chunks_request.model_copy(update={"query": query})

        overview_request = ctx.request.overview
        if overview_request is not None:
            query = clean_whitespace(overview_request.query or "")
            if not query:
                raise ValueError("overview.query must not be empty")
            overview_request = overview_request.model_copy(update={"query": query})

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
        ctx.request = ctx.request.model_copy(
            update={
                "url": url,
                "chunks": chunks_request,
                "overview": overview_request,
            }
        )
        profile_query = (
            chunks_request.query
            if chunks_request is not None
            else (
                overview_request.query if overview_request is not None else url
            )
        )
        profile_name, profile = self.settings.select_profile(
            query=profile_query,
            explicit=ctx.request.profile,
        )
        ctx.profile_name = profile_name
        ctx.profile = profile

        if chunks_request is not None:
            chunk_query_tokens = tokenize_for_query(chunks_request.query)
            chunk_intent_tokens = extract_intent_tokens(
                chunks_request.query,
                profile.intent_terms,
            )
        else:
            chunk_query_tokens = []
            chunk_intent_tokens = []

        ctx.return_content = bool(return_content)
        ctx.content_request = content_request
        ctx.content_options = content_options
        ctx.chunks_request = chunks_request
        ctx.overview_request = overview_request
        ctx.chunk_query_tokens = chunk_query_tokens
        ctx.chunk_intent_tokens = chunk_intent_tokens

        span.set_attr("has_content_output", bool(return_content))
        span.set_attr("has_chunks", bool(chunks_request is not None))
        span.set_attr("has_overview", bool(overview_request is not None))
        span.set_attr("content_depth", str(content_request.depth))
        span.set_attr("profile_name", str(profile_name))
        return ctx


__all__ = ["FetchPrepareStep"]
