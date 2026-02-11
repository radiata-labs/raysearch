from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.pipeline.base import StepBase
from serpsage.pipeline.context import SearchStepContext
from serpsage.text.tokenize import tokenize
from serpsage.text.utils import extract_intent_tokens
from serpsage.util.collections import uniq_preserve_order

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.contracts.services import RankerBase
    from serpsage.core.runtime import CoreRuntime


class RankStep(StepBase):
    span_name = "step.rank"

    def __init__(self, *, rt: CoreRuntime, ranker: RankerBase) -> None:
        super().__init__(rt=rt)
        self._ranker = ranker

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
        intent_tokens = extract_intent_tokens(query, profile.intent_terms)
        ctx.intent_tokens = list(intent_tokens)

        docs = [f"{r.title} {r.snippet}".strip() for r in ctx.results]
        span.set_attr("items_count", int(len(docs)))
        weights = {
            k: float(v)
            for k, v in (self.settings.rank.providers or {}).items()
            if float(v) > 0
        }
        span.set_attr("providers_used", sorted(weights.keys()))
        span.set_attr("weights", weights)
        raw_scores = self._ranker.score_texts(
            texts=docs,
            query=query,
            query_tokens=list(query_tokens),
            intent_tokens=list(intent_tokens),
        )
        norm = self._ranker.normalize(scores=raw_scores)
        if norm and max(norm) <= 0.0 and max(raw_scores) > 0.0:
            norm = [0.5 for _ in norm]

        for i, r in enumerate(ctx.results):
            r.score = float(norm[i]) if i < len(norm) else 0.0
            title_l = (r.title or "").lower()
            snippet_l = (r.snippet or "").lower()
            hits = [t for t in query_tokens if t in title_l or t in snippet_l]
            r.hit_keywords = uniq_preserve_order(hits)

        ctx.results.sort(key=lambda r: float(r.score), reverse=True)
        return ctx


__all__ = ["RankStep"]
