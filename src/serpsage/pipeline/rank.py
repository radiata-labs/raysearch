from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.contracts.base import WorkUnit
from serpsage.text.tokenize import tokenize
from serpsage.text.utils import extract_intent_tokens
from serpsage.util.collections import uniq_preserve_order

if TYPE_CHECKING:
    from serpsage.contracts.protocols import Ranker
    from serpsage.pipeline.steps import StepContext


class RankStep(WorkUnit):
    def __init__(self, *, rt, ranker: Ranker) -> None:  # noqa: ANN001
        super().__init__(rt=rt)
        self._ranker = ranker

    async def run(self, ctx: StepContext) -> StepContext:
        with self.span("step.rank"):
            if not ctx.results:
                return ctx

            query = ctx.request.query
            query_tokens = list(ctx.scratch.get("query_tokens") or tokenize(query))
            profile = ctx.profile or self.settings.get_profile(
                self.settings.pipeline.default_profile
            )
            intent_tokens = extract_intent_tokens(query, profile.intent_terms)

            docs = [f"{r.title} {r.snippet}".strip() for r in ctx.results]
            raw_scores = self._ranker.score_texts(
                texts=docs,
                query=query,
                query_tokens=query_tokens,
                intent_tokens=intent_tokens,
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
