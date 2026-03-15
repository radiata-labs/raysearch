from __future__ import annotations

from typing_extensions import override
from urllib.parse import urlparse

import anyio

from serpsage.components.provider.base import SearchProviderBase
from serpsage.components.provider.blend import (
    resolve_engine_selection_routes,
)
from serpsage.dependencies import Depends
from serpsage.models.components.provider import SearchProviderResult
from serpsage.models.steps.search import (
    SearchCanonicalBucket,
    SearchNormalizedResult,
    SearchQueryJob,
    SearchSnippetContext,
    SearchStepContext,
)
from serpsage.steps.base import StepBase
from serpsage.utils import (
    canonicalize_url,
    clean_whitespace,
    pick_earliest_published_date,
    strip_html,
)


class SearchStep(StepBase[SearchStepContext]):
    provider: SearchProviderBase = Depends()

    @override
    async def run_inner(self, ctx: SearchStepContext) -> SearchStepContext:
        if bool(ctx.plan.aborted):
            ctx.retrieval.urls = []
            ctx.retrieval.published_dates = {}
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
                tg.start_soon(
                    self._run_query,
                    idx,
                    job.query.query,
                    provider_responses,
                    ctx,
                )
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
        canonical_buckets = self._collect_canonical_buckets(
            query_jobs=query_jobs,
            normalized_by_job=normalized_by_job,
        )
        snippets_by_url: dict[str, dict[str, SearchSnippetContext]] = {}
        query_hit_stats: dict[str, int] = {}
        for bucket in canonical_buckets.values():
            url = bucket.representative_url
            snippets_by_url[url] = dict(bucket.snippets_by_source)
            query_hit_stats[url] = int(len(bucket.hit_indexes))
        ctx.retrieval.snippet_context = self._finalize_snippet_context(snippets_by_url)
        ctx.retrieval.query_hit_stats = dict(query_hit_stats)
        ctx.retrieval.published_dates = {
            bucket.representative_url: bucket.published_date
            for bucket in canonical_buckets.values()
            if bucket.published_date
        }
        ctx.retrieval.urls = [
            bucket.representative_url
            for bucket in list(canonical_buckets.values())[
                : int(ctx.plan.prefetch_limit)
            ]
        ]
        return ctx

    def _collect_canonical_buckets(
        self,
        *,
        query_jobs: list[SearchQueryJob],
        normalized_by_job: list[list[SearchNormalizedResult]],
    ) -> dict[str, SearchCanonicalBucket]:
        buckets: dict[str, SearchCanonicalBucket] = {}
        next_order = 0
        for job_index, normalized in enumerate(normalized_by_job):
            if job_index >= len(query_jobs):
                continue
            job = query_jobs[job_index]
            for item in normalized:
                current_order = int(next_order)
                next_order += 1
                canonical_url = str(item.canonical_url or item.url)
                bucket = buckets.get(canonical_url)
                if bucket is None:
                    bucket = SearchCanonicalBucket(
                        representative_url=str(item.url),
                        representative_order=current_order,
                        published_date=str(item.published_date or ""),
                    )
                    buckets[canonical_url] = bucket
                bucket.hit_indexes.add(job_index)
                if current_order < int(bucket.representative_order):
                    bucket.representative_url = str(item.url)
                    bucket.representative_order = current_order
                bucket.published_date = pick_earliest_published_date(
                    bucket.published_date,
                    item.published_date,
                )
                snippet_text = self._pick_snippet(item)
                if not snippet_text:
                    continue
                source_key = str(job.source)
                current = bucket.snippets_by_source.get(source_key)
                if current is None or current_order < int(current.order):
                    bucket.snippets_by_source[source_key] = SearchSnippetContext(
                        snippet=snippet_text,
                        source_query=job.query.query,
                        source_type=job.source,
                        order=current_order,
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
            kwargs = dict(ctx.runtime.provider_extra_kwargs or {})
            subsystem = ctx.runtime.engine_selection_subsystem or "search"
            if resolve_engine_selection_routes(
                settings=self.settings,
                subsystem=subsystem,
                provider=self.provider,
            ):
                if idx < len(ctx.plan.query_jobs):
                    kwargs["include_sources"] = list(
                        ctx.plan.query_jobs[idx].query.include_sources
                    )
            out[idx] = await self.provider.asearch(
                query=query,
                limit=ctx.runtime.provider_limit,
                language=ctx.runtime.provider_language,
                location=ctx.request.user_location,
                moderation=ctx.request.moderation,
                start_published_date=ctx.request.start_published_date,
                end_published_date=ctx.request.end_published_date,
                **kwargs,
            )
            await self.meter.record(
                name="search.query",
                request_id=ctx.request_id,
                key=f"{ctx.request_id}:search.query:{int(idx)}",
                unit="call",
            )
        except Exception as exc:  # noqa: BLE001
            await self.meter.record(
                name="search.query",
                request_id=ctx.request_id,
                key=f"{ctx.request_id}:search.query:{int(idx)}",
                unit="call",
            )
            await self.tracker.error(
                name="search.query.failed",
                request_id=ctx.request_id,
                step="search.search",
                error_code="search_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "query": query,
                },
            )

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
            selected.sort(key=lambda item: int(item.order))
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
                    published_date=clean_whitespace(raw.published_date),
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
