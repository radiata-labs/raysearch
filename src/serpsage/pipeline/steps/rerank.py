from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.pipeline.base import StepBase
from serpsage.pipeline.context import SearchStepContext
from serpsage.text.tokenize import tokenize
from serpsage.text.utils import extract_intent_tokens

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime
    from serpsage.domain.rerank import Reranker


class RerankStep(StepBase):
    span_name = "step.rerank"

    def __init__(self, *, rt: Runtime, reranker: Reranker) -> None:
        super().__init__(rt=rt)
        self._reranker = reranker
        self.bind_deps(reranker)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        if not ctx.results:
            span.set_attr("items_count", 0)
            return ctx

        query = ctx.request.query
        query_tokens = ctx.query_tokens or tokenize(query)
        ctx.query_tokens = list(query_tokens)

        profile = ctx.profile or self.settings.get_profile(
            self.settings.pipeline.default_profile
        )
        intent_tokens = ctx.intent_tokens or extract_intent_tokens(
            query, profile.intent_terms
        )
        ctx.intent_tokens = list(intent_tokens)

        ctx.results = await self._reranker.rerank(
            results=ctx.results,
            query=query,
            query_tokens=list(query_tokens),
            intent_tokens=list(intent_tokens),
        )
        span.set_attr("items_count", int(len(ctx.results or [])))
        return ctx


__all__ = ["RerankStep"]
