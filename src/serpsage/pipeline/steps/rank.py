from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.pipeline.base import StepBase
from serpsage.pipeline.context import SearchStepContext
from serpsage.util.collections import uniq_preserve_order

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.contracts.services import RankerBase
    from serpsage.core.runtime import Runtime


class RankStep(StepBase):
    span_name = "step.rank"

    def __init__(self, *, rt: Runtime, ranker: RankerBase) -> None:
        super().__init__(rt=rt)
        self._ranker = ranker
        self.bind_deps(ranker)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        if not ctx.results:
            span.set_attr("items_count", 0)
            return ctx

        query = ctx.request.query
        query_tokens = ctx.query_tokens or []
        intent_tokens = ctx.intent_tokens or []

        docs = [f"{r.title} {r.snippet}".strip() for r in ctx.results]
        span.set_attr("items_count", int(len(docs)))
        backend = str(self.settings.rank.backend or "blend").lower()
        if backend == "blend":
            weights = {
                k: float(v)
                for k, v in (self.settings.rank.blend.providers or {}).items()
                if float(v) > 0
            }
        elif backend == "heuristic":
            weights = {"heuristic": 1.0}
        elif backend == "bm25":
            weights = {"bm25": 1.0}
        else:
            weights = {}
        span.set_attr("providers_used", sorted(weights.keys()))
        span.set_attr("weights", weights)
        scores = await self._ranker.score_texts(
            texts=docs,
            query=query,
            query_tokens=list(query_tokens),
            intent_tokens=list(intent_tokens),
        )

        for i, r in enumerate(ctx.results):
            r.score = float(scores[i]) if i < len(scores) else 0.0
            title_l = (r.title or "").lower()
            snippet_l = (r.snippet or "").lower()
            hits = [t for t in query_tokens if t in title_l or t in snippet_l]
            r.hit_keywords = uniq_preserve_order(hits)

        ctx.results.sort(key=lambda r: float(r.score), reverse=True)
        return ctx


__all__ = ["RankStep"]
