from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any
from typing_extensions import override
from urllib.parse import urlparse

import anyio

from serpsage.models.errors import AppError
from serpsage.models.pipeline import (
    SearchQueryJob,
    SearchSnippetContext,
    SearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.utils.normalize import clean_whitespace, strip_html
from serpsage.utils.tokenize import tokenize_for_query

if TYPE_CHECKING:
    from serpsage.components.provider.base import SearchProviderBase
    from serpsage.components.rank.base import RankerBase
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class SearchStep(StepBase[SearchStepContext]):
    span_name = "step.search"

    def __init__(
        self,
        *,
        rt: Runtime,
        provider: SearchProviderBase,
        ranker: RankerBase,
    ) -> None:
        super().__init__(rt=rt)
        self._provider = provider
        self._ranker = ranker
        self.bind_deps(provider, ranker)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        if bool(ctx.deep.aborted):
            ctx.prefetch.urls = []
            ctx.prefetch.scores = {}
            ctx.fetch.candidates = []
            ctx.output.results = []
            span.set_attr("aborted", True)
            span.set_attr("query_count", 0)
            span.set_attr("candidate_count", 0)
            return ctx

        req = ctx.request
        query_jobs = self._resolve_query_jobs(ctx=ctx)
        if not query_jobs:
            span.set_attr("aborted", False)
            span.set_attr("query_count", 0)
            return ctx

        raw_results: list[list[dict[str, Any]]] = [[] for _ in query_jobs]
        async with anyio.create_task_group() as tg:
            for idx, job in enumerate(query_jobs):
                tg.start_soon(self._run_query, idx, job.query, raw_results, ctx)

        query_tokens = tokenize_for_query(req.query)
        include_domains = list(req.include_domains or [])
        exclude_domains = list(req.exclude_domains or [])

        scored_by_url: dict[str, tuple[float, int]] = {}
        hit_indexes_by_url: dict[str, set[int]] = {}
        snippets_by_url: dict[str, dict[str, SearchSnippetContext]] = {}
        next_order = 0
        total_filtered_items = 0
        for idx, job in enumerate(query_jobs):
            normalized = self._normalize_results(
                raw_results[idx],
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )
            total_filtered_items += len(normalized)
            if not normalized:
                continue

            # Duplicate title to boost its weight in relevance scoring
            docs = [
                f"{item.title} {item.title} {item.snippet}".strip()
                for item in normalized
            ]
            base_scores = await self._ranker.score_texts(
                texts=docs,
                query=req.query,
                query_tokens=query_tokens,
            )
            for item_idx, item in enumerate(normalized):
                score = (
                    float(base_scores[item_idx]) if item_idx < len(base_scores) else 0.0
                )
                weighted_score = float(score * float(job.weight))
                prev = scored_by_url.get(item.url)
                if prev is None:
                    scored_by_url[item.url] = (weighted_score, next_order)
                else:
                    prev_score, prev_order = prev
                    if weighted_score > prev_score:
                        scored_by_url[item.url] = (weighted_score, prev_order)
                hit_indexes_by_url.setdefault(item.url, set()).add(idx)
                snippet_text = self._pick_snippet(item)
                if snippet_text:
                    source_context = snippets_by_url.setdefault(item.url, {})
                    current = source_context.get(job.source)
                    if current is None or weighted_score > float(current.score):
                        source_context[job.source] = SearchSnippetContext(
                            snippet=snippet_text,
                            source_query=job.query,
                            source_type=job.source,
                            score=weighted_score,
                            order=next_order,
                        )
                next_order += 1

        ctx.deep.snippet_context = self._finalize_snippet_context(snippets_by_url)
        ctx.deep.query_hit_stats = {
            url: int(len(indexes)) for url, indexes in hit_indexes_by_url.items()
        }

        coverage_bonus_weight = (
            float(self.settings.search.deep.coverage_bonus_weight)
            if str(req.depth or "auto") == "deep" and bool(self.settings.search.deep.enabled)
            else 0.0
        )
        ranked_with_prefetch: list[tuple[str, float, int]] = []
        for url, (base_score, order) in scored_by_url.items():
            query_hit_count = int(len(hit_indexes_by_url.get(url, set())))
            bonus = (
                float(coverage_bonus_weight * math.log1p(float(query_hit_count)))
                if coverage_bonus_weight > 0
                else 0.0
            )
            ranked_with_prefetch.append((url, float(base_score + bonus), int(order)))

        max_results = int(req.max_results or self.settings.search.max_results)
        prefetch_limit = self._resolve_prefetch_limit(
            depth=str(req.depth or "auto"),
            max_results=max_results,
        )
        ranked = sorted(
            ranked_with_prefetch, key=lambda item: (-item[1], item[2])
        )
        ctx.prefetch.urls = [url for url, _, _ in ranked[:prefetch_limit]]
        ctx.prefetch.scores = {url: float(score) for url, score, _ in ranked}

        average_query_hits = (
            float(
                sum(len(hits) for hits in hit_indexes_by_url.values())
                / max(1, len(hit_indexes_by_url))
            )
            if hit_indexes_by_url
            else 0.0
        )
        span.set_attr("aborted", False)
        span.set_attr("query_count", int(len(query_jobs)))
        span.set_attr("raw_result_count", int(sum(len(x) for x in raw_results)))
        span.set_attr("filtered_result_count", int(total_filtered_items))
        span.set_attr("deduped_count", int(len(scored_by_url)))
        span.set_attr("prefetch_limit", int(prefetch_limit))
        span.set_attr("candidate_count", int(len(ctx.prefetch.urls)))
        span.set_attr("query_coverage_urls", int(len(hit_indexes_by_url)))
        span.set_attr("query_coverage_avg_hits", float(average_query_hits))
        span.set_attr("coverage_bonus_weight", float(coverage_bonus_weight))
        return ctx

    async def _run_query(
        self,
        idx: int,
        query: str,
        out: list[list[dict[str, Any]]],
        ctx: SearchStepContext,
    ) -> None:
        try:
            out[idx] = await self._provider.asearch(query=query, params=None)
        except Exception as exc:  # noqa: BLE001
            ctx.errors.append(
                AppError(
                    code="search_failed",
                    message=str(exc),
                    details={"query": query, "stage": "search"},
                )
            )

    def _resolve_query_jobs(self, *, ctx: SearchStepContext) -> list[SearchQueryJob]:
        req = ctx.request
        if str(req.depth or "auto") == "deep" and list(ctx.deep.query_jobs or []):
            return list(ctx.deep.query_jobs)
        return self._build_query_jobs(
            query=req.query,
            depth=str(req.depth or "auto"),
            additional_queries=list(req.additional_queries or []),
        )

    def _build_query_jobs(
        self,
        *,
        query: str,
        depth: str,
        additional_queries: list[str],
    ) -> list[SearchQueryJob]:
        jobs: list[SearchQueryJob] = [
            SearchQueryJob(query=clean_whitespace(query), weight=1.0, source="primary")
        ]
        if depth != "deep":
            return jobs
        weight = float(self.settings.search.additional_query_score_weight)
        seen = {clean_whitespace(query).casefold()}
        for raw in additional_queries:
            item = clean_whitespace(str(raw or ""))
            if not item:
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            jobs.append(SearchQueryJob(query=item, weight=weight, source="manual"))
        return jobs

    def _resolve_prefetch_limit(self, *, depth: str, max_results: int) -> int:
        if depth != "deep" or not bool(self.settings.search.deep.enabled):
            return max(1, int(max_results) * 2)
        cfg = self.settings.search.deep
        desired = int(math.ceil(float(max_results) * float(cfg.prefetch_multiplier)))
        return max(1, min(int(cfg.prefetch_max_urls), desired))

    def _pick_snippet(self, item: _NormalizedResult) -> str:
        snippet = clean_whitespace(item.snippet)
        if snippet:
            return snippet
        return clean_whitespace(item.title)

    def _finalize_snippet_context(
        self, values: dict[str, dict[str, SearchSnippetContext]]
    ) -> dict[str, list[SearchSnippetContext]]:
        out: dict[str, list[SearchSnippetContext]] = {}
        for url, grouped in values.items():
            selected = list(grouped.values())
            selected.sort(key=lambda item: (-float(item.score), int(item.order)))
            out[url] = selected[:3]
        return out

    def _normalize_results(
        self,
        raw_results: list[dict[str, Any]],
        *,
        include_domains: list[str],
        exclude_domains: list[str],
    ) -> list[_NormalizedResult]:
        out: list[_NormalizedResult] = []
        for raw in raw_results:
            url = clean_whitespace(str(raw.get("url") or ""))
            if not url:
                continue
            domain = self._extract_domain(url)
            if not self._domain_allowed(
                domain=domain,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            ):
                continue
            title = clean_whitespace(strip_html(str(raw.get("title") or "")))
            snippet_raw = raw.get("snippet")
            if snippet_raw is None:
                snippet_raw = raw.get("content")
            if snippet_raw is None:
                snippet_raw = raw.get("description")
            snippet = clean_whitespace(strip_html(str(snippet_raw or "")))
            out.append(
                _NormalizedResult(
                    url=url,
                    title=title,
                    snippet=snippet,
                )
            )
        return out

    def _extract_domain(self, url: str) -> str:
        parsed = urlparse(url)
        host = parsed.netloc or parsed.path.split("/", 1)[0]
        host = host.split("@")[-1].split(":", 1)[0].strip().lower()
        return host.removeprefix("www.")

    def _domain_allowed(
        self,
        *,
        domain: str,
        include_domains: list[str],
        exclude_domains: list[str],
    ) -> bool:
        if include_domains:
            return any(token in domain for token in include_domains)
        return not (
            exclude_domains and any(token in domain for token in exclude_domains)
        )


class _NormalizedResult:
    def __init__(self, *, url: str, title: str, snippet: str) -> None:
        self.url = url
        self.title = title
        self.snippet = snippet


__all__ = ["SearchStep"]
