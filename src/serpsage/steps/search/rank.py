from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.steps.search import (
    SearchCandidateForScoring,
    SearchRankedCandidate,
    SearchRankOptions,
    SearchRankState,
    SearchRankStats,
    SearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.tokenize import tokenize_for_query
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.components.rank.base import RankerBase
    from serpsage.core.runtime import Runtime
    from serpsage.models.steps.search import SearchFetchedCandidate
_DEFAULT_MAX_CONTEXT_DOCS = 12
_DEEP_MAX_CONTEXT_DOCS = 18
_DEEP_MIN_CONTEXT_DOC_CHARS = 16
_TOP_K_SCORES = 3
_MAIN_CONTENT_SUBPAGE_INDEX = -1  # Sentinel value for main content vs subpage indices


class SearchRankStep(StepBase[SearchStepContext]):
    def __init__(self, *, rt: Runtime, ranker: RankerBase) -> None:
        super().__init__(rt=rt)
        self._ranker = ranker
        self.bind_deps(ranker)

    @override
    async def run_inner(self, ctx: SearchStepContext) -> SearchStepContext:
        """Score fetched search candidates and persist ranked metadata for finalize.
        Args:
            ctx: Search context containing fetched candidates and deep artifacts.
        Returns:
            Search context with `ctx.rank` populated for `SearchFinalizeStep`.
        """
        if bool(ctx.deep.aborted):
            ctx.rank = SearchRankState()
            ctx.output.results = []
            return ctx
        if not ctx.output.results and not ctx.fetch.candidates:
            ctx.rank = SearchRankState()
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
        return ctx

    def _build_rank_options(self, ctx: SearchStepContext) -> SearchRankOptions:
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
        context_docs_limit = (
            int(_DEEP_MAX_CONTEXT_DOCS)
            if deep_enabled
            else int(_DEFAULT_MAX_CONTEXT_DOCS)
        )
        context_doc_min_chars = int(_DEEP_MIN_CONTEXT_DOC_CHARS) if deep_enabled else 0
        return SearchRankOptions(
            content_enabled=bool(content_enabled),
            abstracts_enabled=bool(abstracts_enabled),
            overview_enabled=bool(overview_enabled),
            has_sort_feature=bool(has_sort_feature),
            include_text=include_text,
            exclude_text=exclude_text,
            query_tokens=query_tokens,
            context_query_tokens=context_query_tokens,
            deep_enabled=bool(deep_enabled),
            context_docs_limit=int(context_docs_limit),
            context_doc_min_chars=int(context_doc_min_chars),
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
        from serpsage.models.steps.search import SearchFetchedCandidate  # noqa: PLC0415

        return [
            SearchFetchedCandidate(
                result=item,
                main_abstract_text=clean_whitespace(str(item.content or "")),
                subpage_abstract_texts=[
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
        options: SearchRankOptions,
    ) -> tuple[list[SearchRankedCandidate], SearchRankStats]:
        ranked: list[SearchRankedCandidate] = []
        stats = SearchRankStats()
        ctx.deep.context_scores = {}
        ready: list[SearchCandidateForScoring] = []
        for idx, candidate in enumerate(candidates):
            main_text = clean_whitespace(str(candidate.main_abstract_text or ""))
            if not self._passes_text_filters(
                text=main_text,
                include_text=options.include_text,
                exclude_text=options.exclude_text,
            ):
                stats.filtered_count += 1
                continue
            ready.append(
                SearchCandidateForScoring(
                    order=idx,
                    candidate=candidate,
                    main_text=main_text,
                    subpage_inputs=self._collect_subpage_inputs(candidate),
                )
            )
        if not ready:
            return ranked, stats
        content_scores = await self._batch_score_content(
            candidates=ready,
            query=ctx.request.query,
            query_tokens=options.query_tokens,
            enabled=bool(options.content_enabled),
        )
        context_scores = await self._batch_score_context(
            ctx=ctx,
            candidates=ready,
            query=ctx.request.query,
            query_tokens=options.context_query_tokens,
            enabled=bool(options.deep_enabled),
            options=options,
        )
        for idx, scoped in enumerate(ready):
            candidate = scoped.candidate
            page_scores = [
                self._score_page_with_prefetched_content(
                    content_text=scoped.main_text,
                    content_score=float(content_scores.get((idx, -1), 0.0)),
                    abstract_scores=[
                        float(x) for x in list(candidate.result.abstract_scores)
                    ],
                    overview_scores=[
                        float(x) for x in list(candidate.main_overview_scores)
                    ],
                    content_enabled=bool(options.content_enabled),
                    abstracts_enabled=bool(options.abstracts_enabled),
                    overview_enabled=bool(options.overview_enabled),
                )
            ]
            for sub_idx, (
                content_text,
                abstract_scores,
                overview_scores,
            ) in enumerate(scoped.subpage_inputs):
                page_scores.append(
                    self._score_page_with_prefetched_content(
                        content_text=content_text,
                        content_score=float(content_scores.get((idx, sub_idx), 0.0)),
                        abstract_scores=abstract_scores,
                        overview_scores=overview_scores,
                        content_enabled=bool(options.content_enabled),
                        abstracts_enabled=bool(options.abstracts_enabled),
                        overview_enabled=bool(options.overview_enabled),
                    )
                )
            page_score = (
                max(page_scores, default=0.0) if options.has_sort_feature else 0.0
            )
            context_score = float(context_scores.get(idx, 0.0))
            prefetch_score = float(ctx.prefetch.scores.get(candidate.result.url, 0.0))
            final_score = float(page_score)
            if options.deep_enabled:
                ctx.deep.context_scores[candidate.result.url] = float(context_score)
                final_score = float(
                    options.page_weight * page_score
                    + options.context_weight * context_score
                    + options.prefetch_weight * prefetch_score
                )
            elif not options.has_sort_feature:
                final_score = 0.0
            item = SearchRankedCandidate(
                final_score=float(final_score),
                order=int(scoped.order),
                result=candidate.result,
                page_score=float(page_score),
                context_score=float(context_score),
                prefetch_score=float(prefetch_score),
            )
            ranked.append(item)
            stats.sum_page_score += float(item.page_score)
            stats.sum_context_score += float(item.context_score)
            stats.sum_prefetch_score += float(item.prefetch_score)
        return ranked, stats

    async def _batch_score_content(
        self,
        *,
        candidates: list[SearchCandidateForScoring],
        query: str,
        query_tokens: list[str],
        enabled: bool,
    ) -> dict[tuple[int, int], float]:
        if not enabled:
            return {}
        texts: list[str] = []
        keys: list[tuple[int, int]] = []
        for idx, scoped in enumerate(candidates):
            if scoped.main_text:
                keys.append((idx, _MAIN_CONTENT_SUBPAGE_INDEX))
                texts.append(scoped.main_text)
            for sub_idx, (content_text, _, _) in enumerate(scoped.subpage_inputs):
                if not content_text:
                    continue
                keys.append((idx, sub_idx))
                texts.append(content_text)
        if not texts:
            return {}
        scores = await self._ranker.score_texts(
            texts,
            query=query,
            query_tokens=query_tokens,
        )
        return {
            key: (float(scores[i]) if i < len(scores) else 0.0)
            for i, key in enumerate(keys)
        }

    async def _batch_score_context(
        self,
        *,
        ctx: SearchStepContext,
        candidates: list[SearchCandidateForScoring],
        query: str,
        query_tokens: list[str],
        enabled: bool,
        options: SearchRankOptions,
    ) -> dict[int, float]:
        if not enabled:
            return {}
        texts: list[str] = []
        spans: dict[int, tuple[int, int]] = {}
        for idx, scoped in enumerate(candidates):
            docs = self._collect_context_docs(
                ctx=ctx,
                candidate=scoped.candidate,
                options=options,
            )
            if not docs:
                continue
            start = len(texts)
            texts.extend(docs)
            spans[idx] = (start, len(texts))
        if not texts:
            return {}
        scores = await self._ranker.score_texts(
            texts,
            query=query,
            query_tokens=query_tokens,
        )
        return {
            idx: self._avg_top_k(
                [float(x) for x in list(scores[start:end])],
                k=_TOP_K_SCORES,
            )
            for idx, (start, end) in spans.items()
        }

    def _collect_subpage_inputs(
        self, candidate: SearchFetchedCandidate
    ) -> list[tuple[str, list[float], list[float]]]:
        subpage_count = max(
            len(candidate.result.subpages),
            len(candidate.subpage_abstract_texts),
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
                    candidate.subpage_abstract_texts[sub_idx]
                    if sub_idx < len(candidate.subpage_abstract_texts)
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

    def _collect_context_docs(
        self,
        *,
        ctx: SearchStepContext,
        candidate: SearchFetchedCandidate,
        options: SearchRankOptions,
    ) -> list[str]:
        result = candidate.result
        docs: list[str] = []
        seen: set[str] = set()
        max_docs = max(1, int(options.context_docs_limit))
        min_chars = max(0, int(options.context_doc_min_chars))
        snippets = list(ctx.deep.snippet_context.get(str(result.url), []))
        for item in snippets:
            snippet_text = clean_whitespace(str(item.snippet or ""))
            self._append_context_doc(
                docs=docs,
                seen=seen,
                text=snippet_text,
                min_chars=0,
                max_docs=max_docs,
            )
        for abstract_item in list(result.abstracts or []):
            abstract_text = clean_whitespace(str(abstract_item))
            self._append_context_doc(
                docs=docs,
                seen=seen,
                text=abstract_text,
                min_chars=min_chars,
                max_docs=max_docs,
            )
            if len(docs) >= max_docs:
                break
        for subpage in list(result.subpages or []):
            for subpage_abstract in list(subpage.abstracts or []):
                abstract_text = clean_whitespace(str(subpage_abstract))
                self._append_context_doc(
                    docs=docs,
                    seen=seen,
                    text=abstract_text,
                    min_chars=min_chars,
                    max_docs=max_docs,
                )
                if len(docs) >= max_docs:
                    break
            if len(docs) >= max_docs:
                break
        return docs[:max_docs]

    def _append_context_doc(
        self,
        *,
        docs: list[str],
        seen: set[str],
        text: str,
        min_chars: int,
        max_docs: int,
    ) -> None:
        normalized = clean_whitespace(text)
        if not normalized:
            return
        if min_chars > 0 and len(normalized) < min_chars:
            return
        key = normalized.casefold()
        if key in seen:
            return
        if len(docs) >= max_docs:
            return
        seen.add(key)
        docs.append(normalized)

    def _score_page_with_prefetched_content(
        self,
        *,
        content_text: str,
        content_score: float,
        abstract_scores: list[float],
        overview_scores: list[float],
        content_enabled: bool,
        abstracts_enabled: bool,
        overview_enabled: bool,
    ) -> float:
        if content_enabled and not content_text:
            return 0.0
        if abstracts_enabled and not abstract_scores:
            return 0.0
        if overview_enabled and not overview_scores:
            return 0.0
        parts: list[float] = []
        if content_enabled:
            parts.append(float(content_score))
        if abstracts_enabled:
            parts.append(self._avg_top3(abstract_scores))
        if overview_enabled:
            parts.append(self._avg_top3(overview_scores))
        if not parts:
            return 0.0
        return float(sum(parts) / len(parts))

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
