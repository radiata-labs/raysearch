from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.core.workunit import WorkUnit
from serpsage.text.normalize import clean_whitespace

if TYPE_CHECKING:
    from serpsage.app.response import ResultItem
    from serpsage.contracts.services import RankerBase
    from serpsage.core.runtime import Runtime


class Reranker(WorkUnit):
    def __init__(self, *, rt: Runtime, ranker: RankerBase) -> None:
        super().__init__(rt=rt)
        self._ranker = ranker
        self.bind_deps(ranker)

    async def rerank(
        self,
        *,
        results: list[ResultItem],
        query: str,
        query_tokens: list[str],
        intent_tokens: list[str],
    ) -> list[ResultItem]:
        if not results:
            return []

        page_docs: list[str] = []
        has_any_page = False
        for r in results:
            if r.page and r.page.chunks:
                doc = clean_whitespace(" ".join(c.text for c in r.page.chunks))
                page_docs.append(doc)
                if doc:
                    has_any_page = True
            else:
                page_docs.append("")
        if not has_any_page:
            return results

        raw = await self._ranker.score_texts(
            texts=page_docs,
            query=query,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
        )
        page_scores = self._ranker.normalize(scores=raw)
        if page_scores and max(page_scores) <= 0.0 and max(raw) > 0.0:
            page_scores = [0.5 for _ in page_scores]

        sn_w = 0.4
        pg_w = 0.6
        combined_raw: list[float] = []
        for i, r in enumerate(results):
            snippet_s = float(r.score)
            page_s = float(page_scores[i]) if i < len(page_scores) else 0.0
            combined_raw.append(sn_w * snippet_s + pg_w * page_s)

        combined = self._ranker.normalize(scores=combined_raw)
        if combined and max(combined) <= 0.0 and max(combined_raw) > 0.0:
            combined = [0.5 for _ in combined]

        for i, r in enumerate(results):
            r.score = float(combined[i]) if i < len(combined) else 0.0

        return sorted(results, key=lambda r: float(r.score), reverse=True)


__all__ = ["Reranker"]
