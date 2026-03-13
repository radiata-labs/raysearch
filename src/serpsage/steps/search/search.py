from __future__ import annotations

import math
from contextlib import suppress
from typing_extensions import override
from urllib.parse import urlparse

import anyio

from serpsage.components.provider.base import SearchProviderBase
from serpsage.components.rank.base import RankerBase
from serpsage.dependencies import Depends
from serpsage.models.components.provider import SearchProviderResult
from serpsage.models.components.telemetry import MeterPayload
from serpsage.models.steps.search import (
    SearchCanonicalBucket,
    SearchNormalizedResult,
    SearchQueryJob,
    SearchScoredHit,
    SearchSnippetContext,
    SearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.tokenize import tokenize_for_query
from serpsage.utils import canonicalize_url, clean_whitespace, strip_html


class SearchStep(StepBase[SearchStepContext]):
    provider: SearchProviderBase = Depends()
    ranker: RankerBase = Depends()

    @override
    async def run_inner(self, ctx: SearchStepContext) -> SearchStepContext:
        if bool(ctx.plan.aborted):
            ctx.retrieval.urls = []
            ctx.retrieval.scores = {}
            ctx.retrieval.snippet_context = {}
            ctx.retrieval.query_hit_stats = {}
            ctx.fetch.candidates = []
            ctx.output.results = []
            return ctx
        query_jobs: list[SearchQueryJob] = list(ctx.plan.query_jobs or [])
        if not query_jobs:
            return ctx
        provider_responses: list[list[SearchProviderResult] | None] = [
            None for _ in query_jobs
        ]
        async with anyio.create_task_group() as tg:
            for idx, job in enumerate(query_jobs):
                tg.start_soon(self._run_query, idx, job.query, provider_responses, ctx)
        query_tokens = tokenize_for_query(ctx.request.query)
        include_domains = list(ctx.request.include_domains or [])
        exclude_domains = list(ctx.request.exclude_domains or [])
        normalized_by_job: list[list[SearchNormalizedResult]] = []
        for response in provider_responses:
            provider_results = response if response is not None else []
            normalized_by_job.append(
                self._normalize_results(
                    provider_results,
                    include_domains=include_domains,
                    exclude_domains=exclude_domains,
                )
            )
        scored_hits: list[SearchScoredHit] = []
        docs: list[str] = []
        next_order = 0
        for idx, normalized in enumerate(normalized_by_job):
            for item in normalized:
                docs.append(f"{item.title} {item.title} {item.snippet}".strip())
                scored_hits.append(
                    SearchScoredHit(
                        job_index=idx,
                        order=next_order,
                        item=item,
                    )
                )
                next_order += 1
        base_scores: list[float] = []
        if docs:
            base_scores = await self.ranker.score_texts(
                docs,
                query=ctx.request.query,
                query_tokens=query_tokens,
            )
        canonical_buckets = self._collect_canonical_buckets(
            query_jobs=query_jobs,
            scored_hits=scored_hits,
            base_scores=base_scores,
        )
        snippets_by_url: dict[str, dict[str, SearchSnippetContext]] = {}
        query_hit_stats: dict[str, int] = {}
        ranked_with_prefetch: list[tuple[str, float, int]] = []
        for bucket in canonical_buckets.values():
            url = bucket.representative_url
            order = int(bucket.representative_order)
            query_hit_count = int(len(bucket.hit_indexes))
            snippets_by_url[url] = dict(bucket.snippets_by_source)
            query_hit_stats[url] = query_hit_count
            base_score = self._fuse_prefetch_score(
                hit_scores=list(bucket.hit_scores),
                hit_query_count=query_hit_count,
                total_query_jobs=len(query_jobs),
            )
            ranked_with_prefetch.append((url, float(base_score), order))
        ctx.retrieval.snippet_context = self._finalize_snippet_context(snippets_by_url)
        ctx.retrieval.query_hit_stats = dict(query_hit_stats)
        ranked = sorted(ranked_with_prefetch, key=lambda item: (-item[1], item[2]))
        ctx.retrieval.urls = [
            url for url, _, _ in ranked[: int(ctx.plan.prefetch_limit)]
        ]
        ctx.retrieval.scores = {url: float(score) for url, score, _ in ranked}
        return ctx

    def _collect_canonical_buckets(
        self,
        *,
        query_jobs: list[SearchQueryJob],
        scored_hits: list[SearchScoredHit],
        base_scores: list[float],
    ) -> dict[str, SearchCanonicalBucket]:
        buckets: dict[str, SearchCanonicalBucket] = {}
        for score_idx, hit in enumerate(scored_hits):
            if hit.job_index >= len(query_jobs):
                continue
            job = query_jobs[hit.job_index]
            score = (
                float(base_scores[score_idx]) if score_idx < len(base_scores) else 0.0
            )
            canonical_url = str(hit.item.canonical_url or hit.item.url)
            bucket = buckets.get(canonical_url)
            if bucket is None:
                bucket = SearchCanonicalBucket(
                    representative_url=str(hit.item.url),
                    representative_order=int(hit.order),
                    representative_score=float(score),
                )
                buckets[canonical_url] = bucket
            bucket.hit_indexes.add(int(hit.job_index))
            bucket.hit_scores.append(float(score))
            if float(score) > float(bucket.representative_score) or (
                float(score) == float(bucket.representative_score)
                and int(hit.order) < int(bucket.representative_order)
            ):
                bucket.representative_score = float(score)
                bucket.representative_url = str(hit.item.url)
                bucket.representative_order = int(hit.order)
            snippet_text = self._pick_snippet(hit.item)
            if not snippet_text:
                continue
            source_key = str(job.source)
            current = bucket.snippets_by_source.get(source_key)
            if (
                current is None
                or float(score) > float(current.score)
                or (
                    float(score) == float(current.score)
                    and int(hit.order) < int(current.order)
                )
            ):
                bucket.snippets_by_source[source_key] = SearchSnippetContext(
                    snippet=snippet_text,
                    source_query=job.query,
                    source_type=job.source,
                    score=float(score),
                    order=int(hit.order),
                )
        return buckets

    async def _run_query(
        self,
        idx: int,
        query: str,
        out: list[list[SearchProviderResult] | None],
        ctx: SearchStepContext,
    ) -> None:
        try:
            out[idx] = await self.provider.asearch(
                query=query,
                limit=ctx.runtime.provider_limit,
                locale=str(ctx.runtime.provider_locale or ""),
                **dict(ctx.runtime.provider_extra_kwargs or {}),
            )
            await self._emit_search_meter(
                ctx=ctx,
                query=query,
                query_index=idx,
                status="ok",
            )
        except Exception as exc:  # noqa: BLE001
            await self._emit_search_meter(
                ctx=ctx,
                query=query,
                query_index=idx,
                status="error",
                error_type=type(exc).__name__,
            )
            await self.emit_tracking_event(
                event_name="search.query.error",
                request_id=ctx.request_id,
                stage="search",
                status="error",
                error_code="search_failed",
                error_type=type(exc).__name__,
                attrs={
                    "query": query,
                    "message": str(exc),
                },
            )

    async def _emit_search_meter(
        self,
        *,
        ctx: SearchStepContext,
        query: str,
        query_index: int,
        status: str,
        error_type: str = "",
    ) -> None:
        telemetry = self.telemetry
        if telemetry is None:
            return
        provider_backend = self.components.family_name("provider")
        with suppress(Exception):
            await telemetry.emit(
                event_name="meter.usage.search_call",
                status="error" if status == "error" else "ok",
                request_id=ctx.request_id,
                component="search_step",
                stage="search",
                error_type=error_type,
                idempotency_key=(
                    f"{ctx.request_id}:meter.usage.search_call:{int(query_index)}"
                ),
                attrs={
                    "query": query,
                    "mode": str(ctx.plan.mode),
                    "provider_backend": provider_backend,
                },
                meter=MeterPayload(
                    meter_type="search_call",
                    unit="call",
                    quantity=1.0,
                    provider=provider_backend,
                ),
            )

    def _fuse_prefetch_score(
        self,
        *,
        hit_scores: list[float],
        hit_query_count: int,
        total_query_jobs: int,
    ) -> float:
        ordered_scores = sorted((float(x) for x in hit_scores), reverse=True)
        if not ordered_scores:
            return 0.0
        if int(total_query_jobs) <= 1:
            return float(ordered_scores[0])
        mean_score = float(sum(ordered_scores) / len(ordered_scores))
        coverage = self._coverage_signal(
            hit_query_count=hit_query_count,
            total_query_jobs=total_query_jobs,
        )
        signals = [mean_score, coverage]
        return float(sum(signals) / len(signals))

    def _coverage_signal(self, *, hit_query_count: int, total_query_jobs: int) -> float:
        safe_hit = max(0, int(hit_query_count))
        safe_total = max(1, int(total_query_jobs))
        if safe_total <= 1:
            return 1.0
        return float(math.log1p(float(safe_hit)) / math.log1p(float(safe_total)))

    def _pick_snippet(self, item: SearchNormalizedResult) -> str:
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
            out[url] = selected
        return out

    def _normalize_results(
        self,
        raw_results: list[SearchProviderResult],
        *,
        include_domains: list[str],
        exclude_domains: list[str],
    ) -> list[SearchNormalizedResult]:
        out: list[SearchNormalizedResult] = []
        for raw in raw_results:
            url = clean_whitespace(raw.url)
            if not url:
                continue
            domain = self._extract_domain(url)
            if not self._domain_allowed(
                domain=domain,
                include_domains=include_domains,
                exclude_domains=exclude_domains,
            ):
                continue
            title = clean_whitespace(strip_html(raw.title))
            snippet = clean_whitespace(strip_html(raw.snippet))
            canonical_url = canonicalize_url(url)
            out.append(
                SearchNormalizedResult(
                    url=url,
                    canonical_url=canonical_url,
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
            return any(
                self._domain_token_matches(domain=domain, token=token)
                for token in include_domains
            )
        return not (
            exclude_domains
            and any(
                self._domain_token_matches(domain=domain, token=token)
                for token in exclude_domains
            )
        )

    def _domain_token_matches(self, *, domain: str, token: str) -> bool:
        normalized_domain = clean_whitespace(domain).lower().removeprefix("www.")
        normalized_token = (
            clean_whitespace(token).lower().strip(".").removeprefix("www.")
        )
        if not normalized_domain or not normalized_token:
            return False
        return normalized_domain == normalized_token or normalized_domain.endswith(
            f".{normalized_token}"
        )


__all__ = ["SearchStep"]
