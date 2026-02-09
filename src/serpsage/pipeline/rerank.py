from __future__ import annotations

from serpsage.app.container import Container
from serpsage.pipeline.steps import StepContext
from serpsage.text.normalize import clean_whitespace
from serpsage.text.tokenize import tokenize
from serpsage.text.utils import extract_intent_tokens


class RerankStep:
    """Blend snippet score with page score and rerank."""

    def __init__(self, container: Container) -> None:
        self._c = container

    async def run(self, ctx: StepContext) -> StepContext:
        span = self._c.telemetry.start_span("step.rerank")
        try:
            if not ctx.results:
                return ctx

            query = ctx.request.query
            query_tokens = list(ctx.scratch.get("query_tokens") or tokenize(query))
            profile = ctx.profile or ctx.settings.get_profile(ctx.settings.pipeline.default_profile)
            intent_tokens = extract_intent_tokens(query, profile.intent_terms)

            page_docs: list[str] = []
            has_any_page = False
            for r in ctx.results:
                if r.page and r.page.chunks:
                    doc = clean_whitespace(" ".join(c.text for c in r.page.chunks))
                    page_docs.append(doc)
                    if doc:
                        has_any_page = True
                else:
                    page_docs.append("")
            if not has_any_page:
                return ctx

            raw = self._c.ranker.score_texts(
                texts=page_docs,
                query=query,
                query_tokens=query_tokens,
                intent_tokens=intent_tokens,
            )
            page_scores = self._c.ranker.normalize(scores=raw)
            if page_scores and max(page_scores) <= 0.0 and max(raw) > 0.0:
                page_scores = [0.5 for _ in page_scores]

            # Weights: page higher than snippet by default.
            sn_w = 0.4
            pg_w = 0.6
            combined_raw: list[float] = []
            for i, r in enumerate(ctx.results):
                snippet_s = float(r.score)
                page_s = float(page_scores[i]) if i < len(page_scores) else 0.0
                combined_raw.append(sn_w * snippet_s + pg_w * page_s)

            combined = self._c.ranker.normalize(scores=combined_raw)
            if combined and max(combined) <= 0.0 and max(combined_raw) > 0.0:
                combined = [0.5 for _ in combined]

            for i, r in enumerate(ctx.results):
                r.score = float(combined[i]) if i < len(combined) else 0.0

            ctx.results.sort(key=lambda r: float(r.score), reverse=True)
            return ctx
        finally:
            span.end()


__all__ = ["RerankStep"]

