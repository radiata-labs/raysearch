from __future__ import annotations

from serpsage.app.container import Container
from serpsage.pipeline.steps import StepContext
from serpsage.text.tokenize import tokenize
from serpsage.text.utils import extract_intent_tokens
from serpsage.contracts.protocols import uniq_preserve_order


class RankStep:
    def __init__(self, container: Container) -> None:
        self._c = container

    async def run(self, ctx: StepContext) -> StepContext:
        span = self._c.telemetry.start_span("step.rank")
        try:
            if not ctx.results:
                return ctx

            query = ctx.request.query
            query_tokens = list(ctx.scratch.get("query_tokens") or tokenize(query))
            profile = ctx.profile or ctx.settings.get_profile(ctx.settings.pipeline.default_profile)
            intent_tokens = extract_intent_tokens(query, profile.intent_terms)

            docs = [f"{r.title} {r.snippet}".strip() for r in ctx.results]
            raw_scores = self._c.ranker.score_texts(
                texts=docs,
                query=query,
                query_tokens=query_tokens,
                intent_tokens=intent_tokens,
            )
            norm = self._c.ranker.normalize(scores=raw_scores)
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
        finally:
            span.end()


__all__ = ["RankStep"]

