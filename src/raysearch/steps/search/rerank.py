from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from raysearch.components.rank.base import RankerBase
from raysearch.dependencies import Depends
from raysearch.models.steps.search import (
    SearchCandidateForScoring,
    SearchRankedCandidate,
    SearchRankOptions,
    SearchRankState,
    SearchRankStats,
    SearchStepContext,
)
from raysearch.steps.base import StepBase
from raysearch.tokenize import tokenize_for_query
from raysearch.utils import clean_whitespace

if TYPE_CHECKING:
    from raysearch.models.steps.search import SearchFetchedCandidate


class SearchRerankStep(StepBase[SearchStepContext]):
    ranker: RankerBase = Depends()

    @override
    async def run_inner(self, ctx: SearchStepContext) -> SearchStepContext:
        if bool(ctx.plan.aborted):
            ctx.rank = SearchRankState()
            ctx.output.results = []
            return ctx
        if not ctx.output.results and not ctx.fetch.candidates:
            ctx.rank = SearchRankState()
            return ctx
        options = self._build_rank_options(ctx)
        candidates = self._resolve_candidates(ctx)
        ranked, stats, context_scores = await self._rank_candidates(
            ctx=ctx,
            candidates=candidates,
            options=options,
        )
        ctx.rank = SearchRankState(
            candidates=ranked,
            filtered_count=int(stats.filtered_count),
            sum_page_score=float(stats.sum_page_score),
            sum_context_score=float(stats.sum_context_score),
            has_sort_feature=bool(options.has_sort_feature),
            use_context_score=bool(options.use_context_score),
            max_results=int(options.max_results),
            context_scores=dict(context_scores),
        )
        await self.tracker.info(
            name="search.rerank.completed",
            request_id=ctx.request_id,
            step="search.rerank",
            data={
                "ranked_count": len(ranked),
                "filtered_count": int(stats.filtered_count),
                "has_sort_feature": bool(options.has_sort_feature),
            },
        )
        await self.tracker.debug(
            name="search.rerank.detail",
            request_id=ctx.request_id,
            step="search.rerank",
            data={
                "top_candidates": [
                    {
                        "url": c.result.url,
                        "final_score": round(float(c.final_score), 4),
                        "page_score": round(float(c.page_score), 4),
                        "context_score": round(float(c.context_score), 4),
                    }
                    for c in ranked[:10]
                ],
                "content_enabled": options.content_enabled,
                "abstracts_enabled": options.abstracts_enabled,
                "overview_enabled": options.overview_enabled,
            },
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
        context_query_tokens = (
            tokenize_for_query(ctx.request.query) if ctx.plan.rank_by_context else []
        )
        return SearchRankOptions(
            content_enabled=bool(content_enabled),
            abstracts_enabled=bool(abstracts_enabled),
            overview_enabled=bool(overview_enabled),
            has_sort_feature=bool(has_sort_feature),
            include_text=include_text,
            exclude_text=exclude_text,
            query_tokens=query_tokens,
            context_query_tokens=context_query_tokens,
            use_context_score=bool(ctx.plan.rank_by_context),
            context_docs_limit=int(ctx.plan.context_docs_limit),
            context_doc_min_chars=int(ctx.plan.context_doc_min_chars),
            max_results=int(ctx.plan.max_results),
        )

    def _resolve_candidates(
        self, ctx: SearchStepContext
    ) -> list[SearchFetchedCandidate]:
        candidates = list(ctx.fetch.candidates or [])
        if candidates or not ctx.output.results:
            return candidates
        from raysearch.models.steps.search import (
            SearchFetchedCandidate,  # noqa: PLC0415
        )

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
    ) -> tuple[list[SearchRankedCandidate], SearchRankStats, dict[str, float]]:
        ranked: list[SearchRankedCandidate] = []
        stats = SearchRankStats()
        context_scores_by_url: dict[str, float] = {}
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
            return ranked, stats, context_scores_by_url
        content_scores = await self._batch_score_content(
            candidates=ready,
            query=ctx.request.query,
            query_tokens=options.query_tokens,
            enabled=bool(options.content_enabled),
        )
        abstract_scores = await self._batch_score_abstracts(
            candidates=ready,
            query=ctx.request.query,
            query_tokens=options.context_query_tokens or options.query_tokens,
            enabled=bool(options.abstracts_enabled),
        )
        context_scores = await self._batch_score_context(
            ctx=ctx,
            candidates=ready,
            query=ctx.request.query,
            query_tokens=options.context_query_tokens,
            enabled=bool(options.use_context_score),
            options=options,
        )
        for idx, scoped in enumerate(ready):
            candidate = scoped.candidate
            page_scores = [
                self._score_page_signals(
                    content_text=scoped.main_text,
                    content_score=float(content_scores.get((idx, -1), 0.0)),
                    abstract_text=self._merge_abstracts_text(
                        candidate.result.abstracts
                    ),
                    abstract_score=float(abstract_scores.get((idx, -1), 0.0)),
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
                abstracts,
                overview_scores,
            ) in enumerate(scoped.subpage_inputs):
                page_scores.append(
                    self._score_page_signals(
                        content_text=content_text,
                        content_score=float(content_scores.get((idx, sub_idx), 0.0)),
                        abstract_text=self._merge_abstracts_text(abstracts),
                        abstract_score=float(abstract_scores.get((idx, sub_idx), 0.0)),
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
            final_score = self._combine_scores(
                page_score=page_score,
                context_score=context_score,
                has_sort_feature=bool(options.has_sort_feature),
                use_context_score=bool(options.use_context_score),
            )
            if options.use_context_score:
                context_scores_by_url[candidate.result.url] = float(context_score)
            item = SearchRankedCandidate(
                final_score=float(final_score),
                order=int(scoped.order),
                result=candidate.result,
                page_score=float(page_score),
                context_score=float(context_score),
            )
            ranked.append(item)
            stats.sum_page_score += float(item.page_score)
            stats.sum_context_score += float(item.context_score)
        return ranked, stats, context_scores_by_url

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
                keys.append((idx, -1))
                texts.append(scoped.main_text)
            for sub_idx, (content_text, _, _) in enumerate(scoped.subpage_inputs):
                if not content_text:
                    continue
                keys.append((idx, sub_idx))
                texts.append(content_text)
        if not texts:
            return {}
        scores = await self.ranker.score_texts(
            texts,
            query=query,
            query_tokens=query_tokens,
        )
        return {
            key: (float(scores[i]) if i < len(scores) else 0.0)
            for i, key in enumerate(keys)
        }

    async def _batch_score_abstracts(
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
            main_abstract_text = self._merge_abstracts_text(
                scoped.candidate.result.abstracts
            )
            if main_abstract_text:
                keys.append((idx, -1))
                texts.append(main_abstract_text)
            for sub_idx, (_, abstracts, _) in enumerate(scoped.subpage_inputs):
                abstract_text = self._merge_abstracts_text(abstracts)
                if not abstract_text:
                    continue
                keys.append((idx, sub_idx))
                texts.append(abstract_text)
        if not texts:
            return {}
        scores = await self.ranker.score_texts(
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
        scores = await self.ranker.score_texts(
            texts,
            query=query,
            query_tokens=query_tokens,
        )
        out: dict[int, float] = {}
        for idx, (start, end) in spans.items():
            values = [float(x) for x in list(scores[start:end])]
            if not values:
                continue
            out[idx] = float(sum(values) / len(values))
        return out

    def _collect_subpage_inputs(
        self, candidate: SearchFetchedCandidate
    ) -> list[tuple[str, list[str], list[float]]]:
        subpage_count = max(
            len(candidate.result.subpages),
            len(candidate.subpage_abstract_texts),
            len(candidate.subpages_overview_scores),
        )
        out: list[tuple[str, list[str], list[float]]] = []
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
            abstracts = (
                [str(x) for x in list(subpage.abstracts)] if subpage is not None else []
            )
            overview_scores = (
                [
                    float(x)
                    for x in list(candidate.subpages_overview_scores[sub_idx] or [])
                ]
                if sub_idx < len(candidate.subpages_overview_scores)
                else []
            )
            out.append((content_text, abstracts, overview_scores))
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
        max_docs = max(0, int(options.context_docs_limit))
        if max_docs <= 0:
            return []
        min_chars = max(0, int(options.context_doc_min_chars))
        snippets = list(ctx.retrieval.snippet_context.get(str(result.url), []))
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

    def _score_page_signals(
        self,
        *,
        content_text: str,
        content_score: float,
        abstract_text: str,
        abstract_score: float,
        overview_scores: list[float],
        content_enabled: bool,
        abstracts_enabled: bool,
        overview_enabled: bool,
    ) -> float:
        if content_enabled and not content_text:
            return 0.0
        if abstracts_enabled and not abstract_text:
            return 0.0
        if overview_enabled and not overview_scores:
            return 0.0
        parts: list[float] = []
        if content_enabled:
            parts.append(float(content_score))
        if abstracts_enabled:
            parts.append(float(abstract_score))
        if overview_enabled:
            parts.append(self._average_scores(overview_scores))
        if not parts:
            return 0.0
        return float(sum(parts) / len(parts))

    def _merge_abstracts_text(self, abstracts: list[str]) -> str:
        parts: list[str] = []
        seen: set[str] = set()
        for raw in abstracts:
            text = clean_whitespace(str(raw or ""))
            if not text:
                continue
            key = text.casefold()
            if key in seen:
                continue
            seen.add(key)
            parts.append(text)
        return "\n".join(parts)

    def _average_scores(self, scores: list[float]) -> float:
        values = [float(x) for x in scores]
        if not values:
            return 0.0
        return float(sum(values) / len(values))

    def _combine_scores(
        self,
        *,
        page_score: float,
        context_score: float,
        has_sort_feature: bool,
        use_context_score: bool,
    ) -> float:
        parts: list[float] = []
        if has_sort_feature:
            parts.append(float(page_score))
        if use_context_score:
            parts.append(float(context_score))
        if not parts:
            return 0.0
        return float(sum(parts) / len(parts))

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


__all__ = ["SearchRerankStep"]
