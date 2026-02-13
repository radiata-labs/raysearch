from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

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
        query = clean_whitespace(ctx.request.query or "")
        include_chunks = (
            bool(ctx.request.include_chunks)
            if ctx.request.include_chunks is not None
            else bool(query)
        )
        top_k = (
            int(ctx.request.top_k_chunks)
            if ctx.request.top_k_chunks is not None
            else int(self.settings.fetch.chunk.default_top_k)
        )
        ctx.request = ctx.request.model_copy(
            update={
                "url": url,
                "query": query or None,
                "include_chunks": include_chunks,
                "top_k_chunks": top_k,
            }
        )
        profile_query = query or url
        profile_name, profile = self.settings.select_profile(
            query=profile_query,
            explicit=ctx.request.profile,
        )
        ctx.profile_name = profile_name
        ctx.profile = profile
        if query:
            ctx.query_tokens = tokenize_for_query(query)
            ctx.intent_tokens = extract_intent_tokens(query, profile.intent_terms)
        else:
            ctx.query_tokens = []
            ctx.intent_tokens = []
        span.set_attr("include_chunks", bool(include_chunks))
        span.set_attr("top_k_chunks", int(top_k))
        span.set_attr("profile_name", str(profile_name))
        return ctx


__all__ = ["FetchPrepareStep"]
