from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override
from urllib.parse import urlparse

import anyio

from serpsage.models.errors import AppError
from serpsage.models.pipeline import SearchStepContext
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
        req = ctx.request
        query_jobs = self._build_query_jobs(
            query=req.query,
            depth=req.depth,
            additional_queries=req.additional_queries or [],
        )
        if not query_jobs:
            return ctx

        raw_results: list[list[dict[str, Any]]] = [[] for _ in query_jobs]
        async with anyio.create_task_group() as tg:
            for idx, (query, _) in enumerate(query_jobs):
                tg.start_soon(self._run_query, idx, query, raw_results, ctx)

        query_tokens = tokenize_for_query(req.query)
        include_domains = list(req.include_domains or [])
        exclude_domains = list(req.exclude_domains or [])

        scored_by_url: dict[str, tuple[float, int]] = {}
        next_order = 0
        total_filtered_items = 0
        for idx, (_, source_weight) in enumerate(query_jobs):
            normalized = self._normalize_results(
                raw_results[idx],
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            )
            total_filtered_items += len(normalized)
            if not normalized:
                continue

            docs = [f"{item.title} {item.snippet}".strip() for item in normalized]
            base_scores = await self._ranker.score_texts(
                texts=docs,
                query=req.query,
                query_tokens=query_tokens,
            )
            for item_idx, item in enumerate(normalized):
                score = float(base_scores[item_idx]) if item_idx < len(base_scores) else 0.0
                weighted_score = float(score * source_weight)
                prev = scored_by_url.get(item.url)
                if prev is None:
                    scored_by_url[item.url] = (weighted_score, next_order)
                else:
                    prev_score, prev_order = prev
                    if weighted_score > prev_score:
                        scored_by_url[item.url] = (weighted_score, prev_order)
                next_order += 1

        max_results = int(req.max_results or self.settings.search.max_results)
        prefetch_limit = max(1, max_results * 2)
        ranked = sorted(scored_by_url.items(), key=lambda item: (-item[1][0], item[1][1]))
        ctx.candidate_urls = [url for url, _ in ranked[:prefetch_limit]]
        ctx.candidate_scores = {url: float(meta[0]) for url, meta in ranked}

        span.set_attr("query_count", int(len(query_jobs)))
        span.set_attr("raw_result_count", int(sum(len(x) for x in raw_results)))
        span.set_attr("filtered_result_count", int(total_filtered_items))
        span.set_attr("deduped_count", int(len(scored_by_url)))
        span.set_attr("prefetch_limit", int(prefetch_limit))
        span.set_attr("candidate_count", int(len(ctx.candidate_urls)))
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

    def _build_query_jobs(
        self,
        *,
        query: str,
        depth: str,
        additional_queries: list[str],
    ) -> list[tuple[str, float]]:
        jobs: list[tuple[str, float]] = [(query, 1.0)]
        if depth != "deep":
            return jobs
        weight = float(self.settings.search.additional_query_score_weight)
        seen = {query.casefold()}
        for raw in additional_queries:
            item = clean_whitespace(str(raw or ""))
            if not item:
                continue
            key = item.casefold()
            if key in seen:
                continue
            seen.add(key)
            jobs.append((item, weight))
        return jobs

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
        if host.startswith("www."):
            host = host[4:]
        return host

    def _domain_allowed(
        self,
        *,
        domain: str,
        include_domains: list[str],
        exclude_domains: list[str],
    ) -> bool:
        if include_domains:
            return any(token in domain for token in include_domains)
        if exclude_domains and any(token in domain for token in exclude_domains):
            return False
        return True


class _NormalizedResult:
    def __init__(self, *, url: str, title: str, snippet: str) -> None:
        self.url = url
        self.title = title
        self.snippet = snippet


__all__ = ["SearchStep"]
