from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING

from serpsage.core.workunit import WorkUnit

if TYPE_CHECKING:
    from serpsage.app.response import ResultItem
    from serpsage.core.runtime import Runtime


class Reranker(WorkUnit):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    async def rerank(
        self,
        *,
        results: list[ResultItem],
    ) -> list[ResultItem]:
        if not results:
            return []

        page_scores: list[float] = []
        has_any_page = False
        for r in results:
            if r.page and r.page.chunks:
                scores = [float(c.score or 0.0) for c in r.page.chunks]
                page_scores.append(max(scores) if scores else 0.0)
                has_any_page = True
            else:
                page_scores.append(0)

        result_scores = [float(r.score) if r.score else 0.0 for r in results]

        if not has_any_page or max(page_scores) <= min(result_scores):
            return results

        print(result_scores, "\n", page_scores)

        sn_w = 0.7
        pg_w = 0.3
        combined: list[float] = self._blend_scores(
            result_scores, page_scores, k=sn_w / pg_w
        )

        combine_calibration = (
            max(result_scores) / max(combined) if max(combined) > 0 else 1.0
        )
        combined = [s * combine_calibration for s in combined]

        print(combined)

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
            g = max(-0.5, min(0.5, y - 0.5))
            t = math.log(x / (1.0 - x)) + k * g
            out.append(1.0 / (1.0 + math.exp(-t)))
        return out

    def _get_calibration(
        self, result_scores: list[float], page_scores: list[float]
    ) -> float:
        result_scores = [float(s) for s in result_scores if s > 0.01]
        page_scores = [float(s) for s in page_scores if s > 0.01]
        if not result_scores or not page_scores:
            return 1.0
        return (
            statistics.median(result_scores) / statistics.median(page_scores)
            if statistics.median(page_scores) > 0
            else 1.0
        )


__all__ = ["Reranker"]
