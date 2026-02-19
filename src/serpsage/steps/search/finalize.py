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
        if not ctx.output.results and not ctx.fetch.candidates:
            span.set_attr("before_count", 0)
            span.set_attr("after_count", 0)
            return ctx

        content_enabled = self._is_enabled(ctx.request.fetchs.content)
        abstracts_enabled = self._is_enabled(ctx.request.fetchs.abstracts)
        overview_enabled = self._is_enabled(ctx.request.fetchs.overview)
        has_sort_feature = bool(
            content_enabled or abstracts_enabled or overview_enabled
        )

        include_text = [
            clean_whitespace(x).casefold() for x in (ctx.request.include_text or [])
        ]
        exclude_text = [
            clean_whitespace(x).casefold() for x in (ctx.request.exclude_text or [])
        ]
        query_tokens = tokenize_for_query(ctx.request.query) if content_enabled else []

        candidates = list(ctx.fetch.candidates or [])
        if not candidates and ctx.output.results:
            from serpsage.models.pipeline import SearchFetchedCandidate  # noqa: PLC0415

            candidates = [
                SearchFetchedCandidate(
                    result=item,
                    main_md_for_abstract=clean_whitespace(str(item.content or "")),
                    subpages_md_for_abstract=[
                        clean_whitespace(str(sub.content or ""))
                        for sub in item.subpages
                    ],
                )
                for item in ctx.output.results
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
            if not has_sort_feature:
                kept.append((0.0, idx, candidate.result))
                continue

            main_page_score = await self._score_page(
                content_text=main_text,
                abstract_scores=[
                    float(x) for x in list(candidate.result.abstract_scores)
                ],
                overview_scores=[
                    float(x) for x in list(candidate.main_overview_scores)
                ],
                content_enabled=content_enabled,
                abstracts_enabled=abstracts_enabled,
                overview_enabled=overview_enabled,
                query=ctx.request.query,
                query_tokens=query_tokens,
            )
            page_scores = [main_page_score]
            subpage_count = max(
                len(candidate.result.subpages),
                len(candidate.subpages_md_for_abstract),
                len(candidate.subpages_overview_scores),
            )
            for sub_idx in range(subpage_count):
                subpage = (
                    candidate.result.subpages[sub_idx]
                    if sub_idx < len(candidate.result.subpages)
                    else None
                )
                subpage_text = clean_whitespace(
                    str(
                        candidate.subpages_md_for_abstract[sub_idx]
                        if sub_idx < len(candidate.subpages_md_for_abstract)
                        else ""
                    )
                )
                subpage_abstract_scores = (
                    [float(x) for x in list(subpage.abstract_scores)]
                    if subpage is not None
                    else []
                )
                subpage_overview_scores = (
                    [
                        float(x)
                        for x in list(candidate.subpages_overview_scores[sub_idx] or [])
                    ]
                    if sub_idx < len(candidate.subpages_overview_scores)
                    else []
                )
                subpage_score = await self._score_page(
                    content_text=subpage_text,
                    abstract_scores=subpage_abstract_scores,
                    overview_scores=subpage_overview_scores,
                    content_enabled=content_enabled,
                    abstracts_enabled=abstracts_enabled,
                    overview_enabled=overview_enabled,
                    query=ctx.request.query,
                    query_tokens=query_tokens,
                )
                page_scores.append(subpage_score)
            kept.append((max(page_scores, default=0.0), idx, candidate.result))

        if has_sort_feature:
            kept.sort(key=lambda x: (-x[0], x[1]))
        else:
            kept.sort(key=lambda x: x[1])
        max_results = int(ctx.request.max_results or self.settings.search.max_results)
        ctx.output.results = [item for _, _, item in kept[: max(1, max_results)]]

        span.set_attr("before_count", int(len(kept) + filtered_count))
        span.set_attr("filtered_by_text", int(filtered_count))
        span.set_attr("after_count", int(len(ctx.output.results)))
        span.set_attr("max_results", int(max_results))
        span.set_attr("content_enabled", bool(content_enabled))
        span.set_attr("abstracts_enabled", bool(abstracts_enabled))
        span.set_attr("overview_enabled", bool(overview_enabled))
        return ctx

    async def _score_page(
        self,
        *,
        content_text: str,
        abstract_scores: list[float],
        overview_scores: list[float],
        content_enabled: bool,
        abstracts_enabled: bool,
        overview_enabled: bool,
        query: str,
        query_tokens: list[str],
    ) -> float:
        if content_enabled and not content_text:
            return 0.0
        if abstracts_enabled and not abstract_scores:
            return 0.0
        if overview_enabled and not overview_scores:
            return 0.0

        parts: list[float] = []
        if content_enabled:
            content_part = await self._score_content(
                text=content_text,
                query=query,
                query_tokens=query_tokens,
            )
            parts.append(content_part)
        if abstracts_enabled:
            parts.append(self._avg_top3(abstract_scores))
        if overview_enabled:
            parts.append(self._avg_top3(overview_scores))
        if not parts:
            return 0.0
        return float(sum(parts) / len(parts))

    async def _score_content(
        self,
        *,
        text: str,
        query: str,
        query_tokens: list[str],
    ) -> float:
        scores = await self._ranker.score_texts(
            texts=[text],
            query=query,
            query_tokens=query_tokens,
        )
        if not scores:
            return 0.0
        return float(scores[0])

    def _avg_top3(self, scores: list[float]) -> float:
        top3 = [float(x) for x in list(scores)[:3]]
        if not top3:
            return 0.0
        return float(sum(top3) / len(top3))

    def _is_enabled(self, value: bool | object) -> bool:
        return bool(value) if isinstance(value, bool) else True

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
