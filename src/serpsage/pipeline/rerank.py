from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.contracts.base import WorkUnit
from serpsage.text.tokenize import tokenize
from serpsage.text.utils import extract_intent_tokens

if TYPE_CHECKING:
    from serpsage.domain.rerank import Reranker
    from serpsage.pipeline.steps import StepContext


class RerankStep(WorkUnit):
    def __init__(self, *, rt, reranker: Reranker) -> None:  # noqa: ANN001
        super().__init__(rt=rt)
        self._reranker = reranker

    async def run(self, ctx: StepContext) -> StepContext:
        with self.span("step.rerank"):
            if not ctx.results:
                return ctx
            query = ctx.request.query
            query_tokens = list(ctx.scratch.get("query_tokens") or tokenize(query))
            profile = ctx.profile or self.settings.get_profile(
                self.settings.pipeline.default_profile
            )
            intent_tokens = extract_intent_tokens(query, profile.intent_terms)

            ctx.results = self._reranker.rerank(
                results=ctx.results,
                query=query,
                query_tokens=query_tokens,
                intent_tokens=intent_tokens,
            )
            return ctx


__all__ = ["RerankStep"]
