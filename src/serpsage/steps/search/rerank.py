from __future__ import annotations

import math
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import SearchStepContext
from serpsage.steps.base import StepBase

if TYPE_CHECKING:
    from serpsage.app.response import ResultItem
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class RerankStep(StepBase[SearchStepContext]):
    span_name = "step.rerank"

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        if not ctx.results:
            span.set_attr("items_count", 0)
            return ctx

        ctx.results = self._rerank(results=ctx.results)
        span.set_attr("items_count", int(len(ctx.results or [])))
        return ctx

    def _rerank(self, *, results: list[ResultItem]) -> list[ResultItem]:
        if not results:
            return []

        page_scores: list[float] = []
        has_any_page = False
        for r in results:
            if r.page and r.page.abstracts:
                scores = [float(c.score or 0.0) for c in r.page.abstracts]
                page_scores.append(max(scores) if scores else 0.0)
                has_any_page = True
            else:
                page_scores.append(0.0)

        result_scores = [float(r.score) if r.score else 0.0 for r in results]

        if not has_any_page or max(page_scores) <= min(result_scores):
            return results

        combined = self._blend_scores(result_scores, page_scores, k=0.5)
        combine_calibration = (
            max(result_scores) / max(combined) if max(combined) > 0 else 1.0
        )
        combined = [s * combine_calibration for s in combined]

        for i, r in enumerate(results):
            r.score = float(combined[i]) if i < len(combined) else r.score
        return sorted(results, key=lambda r: float(r.score), reverse=True)

    def _blend_scores(
        self,
        result_scores: list[float],
        page_scores: list[float],
        *,
        k: float = 0.6,
    ) -> list[float]:
        eps = 1e-6
        out: list[float] = []
        for x, y in zip(result_scores, page_scores, strict=False):
            x = min(1.0 - eps, max(eps, x))
            g = max(0, min(1, y))
            t = math.log(x / (1.0 - x)) + k * g
            out.append(1.0 / (1.0 + math.exp(-t)))
        return out


__all__ = ["RerankStep"]
