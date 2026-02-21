from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import (
    SearchRankedCandidate,
    SearchRankState,
    SearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.utils.normalize import clean_whitespace
from serpsage.utils.tokenize import tokenize_for_query

if TYPE_CHECKING:
    from serpsage.components.rank.base import RankerBase
    from serpsage.core.runtime import Runtime
    from serpsage.models.pipeline import SearchFetchedCandidate
    from serpsage.telemetry.base import SpanBase


_MAX_CONTEXT_DOCS = 12
_TOP_K_SCORES = 3


@dataclass(slots=True)
class _RankOptions:
    content_enabled: bool
    abstracts_enabled: bool
    overview_enabled: bool
    has_sort_feature: bool
    include_text: list[str]
    exclude_text: list[str]
    query_tokens: list[str]
    context_query_tokens: list[str]
    deep_enabled: bool
    max_results: int
    page_weight: float
    context_weight: float
    prefetch_weight: float


@dataclass(slots=True)
class _RankStats:
    filtered_count: int = 0
    sum_page_score: float = 0.0
    sum_context_score: float = 0.0
    sum_prefetch_score: float = 0.0


class SearchRankStep(StepBase[SearchStepContext]):
    span_name = "step.search_rank"

    def __init__(self, *, rt: Runtime, ranker: RankerBase) -> None:
        super().__init__(rt=rt)
        self._ranker = ranker
        self.bind_deps(ranker)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        """Score fetched search candidates and persist ranked metadata for finalize.

        Args:
            ctx: Search context containing fetched candidates and deep artifacts.
            span: Telemetry span for ranking metrics.

        Returns:
            Search context with `ctx.rank` populated for `SearchFinalizeStep`.
        """
        if bool(ctx.deep.aborted):
            ctx.rank = SearchRankState()
            ctx.output.results = []
            span.set_attr("aborted", True)
            span.set_attr("before_count", 0)
            span.set_attr("after_count", 0)
            return ctx

        if not ctx.output.results and not ctx.fetch.candidates:
            ctx.rank = SearchRankState()
            span.set_attr("aborted", False)
            span.set_attr("before_count", 0)
            span.set_attr("after_count", 0)
            return ctx

        options = self._build_rank_options(ctx)
        candidates = self._resolve_candidates(ctx)
        ranked, stats = await self._rank_candidates(
            ctx=ctx,
            candidates=candidates,
            options=options,
        )
        ctx.rank = SearchRankState(
            candidates=ranked,
            filtered_count=int(stats.filtered_count),
            sum_page_score=float(stats.sum_page_score),
            sum_context_score=float(stats.sum_context_score),
            sum_prefetch_score=float(stats.sum_prefetch_score),
            deep_enabled=bool(options.deep_enabled),
            has_sort_feature=bool(options.has_sort_feature),
            max_results=int(options.max_results),
            page_weight=float(options.page_weight),
            context_weight=float(options.context_weight),
            prefetch_weight=float(options.prefetch_weight),
        )
        ranked_count = len(ranked)
        span.set_attr("aborted", False)
        span.set_attr("before_count", int(ranked_count + stats.filtered_count))
        span.set_attr("filtered_by_text", int(stats.filtered_count))
        span.set_attr("after_count", int(ranked_count))
        span.set_attr("max_results", int(options.max_results))
        span.set_attr("content_enabled", bool(options.content_enabled))
        span.set_attr("abstracts_enabled", bool(options.abstracts_enabled))
        span.set_attr("overview_enabled", bool(options.overview_enabled))
        span.set_attr("deep_enabled", bool(options.deep_enabled))
        span.set_attr(
            "avg_page_score", float(stats.sum_page_score / max(1, ranked_count))
        )
        span.set_attr(
            "avg_context_score", float(stats.sum_context_score / max(1, ranked_count))
        )
        span.set_attr(
            "avg_prefetch_score",
            float(stats.sum_prefetch_score / max(1, ranked_count)),
        )
        span.set_attr(
            "final_page_weight",
            float(options.page_weight if options.deep_enabled else 1.0),
        )
        span.set_attr(
            "final_context_weight",
            float(options.context_weight if options.deep_enabled else 0.0),
        )
        span.set_attr(
            "final_prefetch_weight",
            float(options.prefetch_weight if options.deep_enabled else 0.0),
        )
        return ctx

    def _build_rank_options(self, ctx: SearchStepContext) -> _RankOptions:
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
        context_query_tokens = tokenize_for_query(ctx.request.query)

        deep_enabled = str(ctx.request.mode or "auto") == "deep" and bool(
            ctx.settings.search.deep.enabled
        )
        deep_cfg = ctx.settings.search.deep

        return _RankOptions(
            content_enabled=bool(content_enabled),
            abstracts_enabled=bool(abstracts_enabled),
            overview_enabled=bool(overview_enabled),
            has_sort_feature=bool(has_sort_feature),
            include_text=include_text,
            exclude_text=exclude_text,
            query_tokens=query_tokens,
            context_query_tokens=context_query_tokens,
            deep_enabled=bool(deep_enabled),
            max_results=max(
                1, int(ctx.request.max_results or self.settings.search.max_results)
            ),
            page_weight=float(deep_cfg.final_page_weight),
            context_weight=float(deep_cfg.final_context_weight),
            prefetch_weight=float(deep_cfg.final_prefetch_weight),
        )

    def _resolve_candidates(
        self, ctx: SearchStepContext
    ) -> list[SearchFetchedCandidate]:
        candidates = list(ctx.fetch.candidates or [])
        if candidates or not ctx.output.results:
            return candidates

        from serpsage.models.pipeline import SearchFetchedCandidate  # noqa: PLC0415

        return [
            SearchFetchedCandidate(
                result=item,
                main_md_for_abstract=clean_whitespace(str(item.content or "")),
                subpages_md_for_abstract=[
                    clean_whitespace(str(sub.content or "")) for sub in item.subpages
                ],
            )
            for item in ctx.output.results
        ]

    async def _rank_candidates(
        self,
        *,
        ctx: SearchStepContext,
        candidates: list[SearchFetchedCandidate],
        options: _RankOptions,
    ) -> tuple[list[SearchRankedCandidate], _RankStats]:
        ranked: list[SearchRankedCandidate] = []
        stats = _RankStats()
        ctx.deep.context_scores = {}

        for idx, candidate in enumerate(candidates):
            main_text = clean_whitespace(str(candidate.main_md_for_abstract or ""))
            if not self._passes_text_filters(
                text=main_text,
                include_text=options.include_text,
                exclude_text=options.exclude_text,
            ):
                stats.filtered_count += 1
                continue

            item = await self._score_candidate(
                ctx=ctx,
                candidate=candidate,
                order=idx,
                main_text=main_text,
                options=options,
            )
            ranked.append(item)
            stats.sum_page_score += float(item.page_score)
            stats.sum_context_score += float(item.context_score)
            stats.sum_prefetch_score += float(item.prefetch_score)
        return ranked, stats

    async def _score_candidate(
        self,
        *,
        ctx: SearchStepContext,
        candidate: SearchFetchedCandidate,
        order: int,
        main_text: str,
        options: _RankOptions,
    ) -> SearchRankedCandidate:
        page_scores = [
            await self._score_page(
                content_text=main_text,
                abstract_scores=[
                    float(x) for x in list(candidate.result.abstract_scores)
                ],
                overview_scores=[
                    float(x) for x in list(candidate.main_overview_scores)
                ],
                content_enabled=options.content_enabled,
                abstracts_enabled=options.abstracts_enabled,
                overview_enabled=options.overview_enabled,
                query=ctx.request.query,
                query_tokens=options.query_tokens,
            )
        ]
        page_scores.extend(
            await self._score_subpages(
                candidate=candidate,
                options=options,
                query=ctx.request.query,
            )
        )

        page_score = max(page_scores, default=0.0) if options.has_sort_feature else 0.0
        context_score = 0.0
        prefetch_score = float(ctx.prefetch.scores.get(candidate.result.url, 0.0))
        final_score = float(page_score)

        if options.deep_enabled:
            context_score = await self._score_context(
                ctx=ctx,
                candidate=candidate,
                query=ctx.request.query,
                query_tokens=options.context_query_tokens,
            )
            ctx.deep.context_scores[candidate.result.url] = float(context_score)
            final_score = float(
                options.page_weight * page_score
                + options.context_weight * context_score
                + options.prefetch_weight * prefetch_score
            )
        elif not options.has_sort_feature:
            final_score = 0.0

        return SearchRankedCandidate(
            final_score=float(final_score),
            order=int(order),
            result=candidate.result,
            page_score=float(page_score),
            context_score=float(context_score),
            prefetch_score=float(prefetch_score),
        )

    async def _score_subpages(
        self,
        *,
        candidate: SearchFetchedCandidate,
        options: _RankOptions,
        query: str,
    ) -> list[float]:
        scores: list[float] = []
        for (
            content_text,
            abstract_scores,
            overview_scores,
        ) in self._collect_subpage_inputs(candidate):
            value = await self._score_page(
                content_text=content_text,
                abstract_scores=abstract_scores,
                overview_scores=overview_scores,
                content_enabled=options.content_enabled,
                abstracts_enabled=options.abstracts_enabled,
                overview_enabled=options.overview_enabled,
                query=query,
                query_tokens=options.query_tokens,
            )
            scores.append(float(value))
        return scores

    def _collect_subpage_inputs(
        self, candidate: SearchFetchedCandidate
    ) -> list[tuple[str, list[float], list[float]]]:
        subpage_count = max(
            len(candidate.result.subpages),
            len(candidate.subpages_md_for_abstract),
            len(candidate.subpages_overview_scores),
        )
        out: list[tuple[str, list[float], list[float]]] = []
        for sub_idx in range(subpage_count):
            subpage = (
                candidate.result.subpages[sub_idx]
                if sub_idx < len(candidate.result.subpages)
                else None
            )
            content_text = clean_whitespace(
                str(
                    candidate.subpages_md_for_abstract[sub_idx]
                    if sub_idx < len(candidate.subpages_md_for_abstract)
                    else ""
                )
            )
            abstract_scores = (
                [float(x) for x in list(subpage.abstract_scores)]
                if subpage is not None
                else []
            )
            overview_scores = (
                [
                    float(x)
                    for x in list(candidate.subpages_overview_scores[sub_idx] or [])
                ]
                if sub_idx < len(candidate.subpages_overview_scores)
                else []
            )
            out.append((content_text, abstract_scores, overview_scores))
        return out

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

    async def _score_context(
        self,
        *,
        ctx: SearchStepContext,
        candidate: SearchFetchedCandidate,
        query: str,
        query_tokens: list[str],
    ) -> float:
        result = candidate.result
        docs: list[str] = []
        snippets = list(ctx.deep.snippet_context.get(str(result.url), []))
        docs.extend(
            clean_whitespace(str(item.snippet or ""))
            for item in snippets
            if clean_whitespace(str(item.snippet or ""))
        )
        docs.extend(
            clean_whitespace(str(item))
            for item in list(result.abstracts or [])
            if clean_whitespace(str(item))
        )
        for subpage in list(result.subpages or []):
            docs.extend(
                clean_whitespace(str(item))
                for item in list(subpage.abstracts or [])
                if clean_whitespace(str(item))
            )
        docs = docs[:_MAX_CONTEXT_DOCS]
        if not docs:
            return 0.0
        scores = await self._ranker.score_texts(
            texts=docs,
            query=query,
            query_tokens=query_tokens,
        )
        return self._avg_top_k(scores, k=_TOP_K_SCORES)

    def _avg_top3(self, scores: list[float]) -> float:
        top3 = [float(x) for x in list(scores)[:_TOP_K_SCORES]]
        if not top3:
            return 0.0
        return float(sum(top3) / len(top3))

    def _avg_top_k(self, scores: list[float], *, k: int) -> float:
        if not scores or k <= 0:
            return 0.0
        top_k = sorted((float(x) for x in scores), reverse=True)[:k]
        if not top_k:
            return 0.0
        return float(sum(top_k) / len(top_k))

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


__all__ = ["SearchRankStep"]
