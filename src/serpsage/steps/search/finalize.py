from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import SearchStepContext
from serpsage.steps.base import StepBase
from serpsage.utils.normalize import clean_whitespace
from serpsage.utils.tokenize import tokenize_for_query

if TYPE_CHECKING:
    from serpsage.app.response import FetchResultItem
    from serpsage.components.rank.base import RankerBase
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class SearchFinalizeStep(StepBase[SearchStepContext]):
    span_name = "step.search_finalize"

    def __init__(self, *, rt: Runtime, ranker: RankerBase) -> None:
        super().__init__(rt=rt)
        self._ranker = ranker
        self.bind_deps(ranker)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        if not ctx.results and not ctx.fetched_candidates:
            span.set_attr("before_count", 0)
            span.set_attr("after_count", 0)
            return ctx

        include_text = [
            clean_whitespace(x).casefold() for x in (ctx.request.include_text or [])
        ]
        exclude_text = [
            clean_whitespace(x).casefold() for x in (ctx.request.exclude_text or [])
        ]
        query_tokens = tokenize_for_query(ctx.request.query)

        candidates = list(ctx.fetched_candidates or [])
        if not candidates and ctx.results:
            from serpsage.models.pipeline import SearchFetchedCandidate  # noqa: PLC0415

            candidates = [
                SearchFetchedCandidate(
                    result=item,
                    main_md_for_abstract=clean_whitespace(str(item.content or "")),
                    subpages_md_for_abstract=[
                        clean_whitespace(str(sub.content or "")) for sub in item.subpages
                    ],
                )
                for item in ctx.results
            ]

        kept: list[tuple[float, int, FetchResultItem]] = []
        filtered_count = 0
        for idx, candidate in enumerate(candidates):
            main_text = clean_whitespace(str(candidate.main_md_for_abstract or ""))
            if not self._passes_text_filters(
                text=main_text,
                include_text=include_text,
                exclude_text=exclude_text,
            ):
                filtered_count += 1
                continue
            docs = [main_text]
            docs.extend(
                clean_whitespace(str(text or ""))
                for text in candidate.subpages_md_for_abstract
            )
            docs = [doc for doc in docs if doc]
            if not docs:
                best_score = 0.0
            else:
                scores = await self._ranker.score_texts(
                    texts=docs,
                    query=ctx.request.query,
                    query_tokens=query_tokens,
                )
                best_score = max((float(score) for score in scores), default=0.0)
            kept.append((best_score, idx, candidate.result))

        kept.sort(key=lambda x: (-x[0], x[1]))
        max_results = int(ctx.request.max_results or self.settings.search.max_results)
        ctx.results = [item for _, _, item in kept[: max(1, max_results)]]

        span.set_attr("before_count", int(len(kept) + filtered_count))
        span.set_attr("filtered_by_text", int(filtered_count))
        span.set_attr("after_count", int(len(ctx.results)))
        span.set_attr("max_results", int(max_results))
        return ctx

    def _passes_text_filters(
        self,
        *,
        text: str,
        include_text: list[str],
        exclude_text: list[str],
    ) -> bool:
        haystack = clean_whitespace(text).casefold()
        if include_text and not all(needle in haystack for needle in include_text):
            return False
        return not (exclude_text and any(needle in haystack for needle in exclude_text))


__all__ = ["SearchFinalizeStep"]
