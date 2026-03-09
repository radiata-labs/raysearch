from __future__ import annotations

import math
import re
from contextlib import suppress
from typing import Any, Literal
from typing_extensions import override
from urllib.parse import parse_qsl, urlencode, urlparse, urlsplit, urlunsplit

import anyio

from serpsage.components.base import Depends
from serpsage.components.provider.base import SearchProviderBase
from serpsage.components.rank.base import RankerBase
from serpsage.core.runtime import Runtime
from serpsage.models.components.provider import (
    SearchProviderResponse,
    SearchProviderResult,
)
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
from serpsage.utils import clean_whitespace, strip_html

_TRACKING_QUERY_KEYS = {"gclid", "fbclid", "msclkid"}
_TRACKING_QUERY_PREFIXES = ("utm_",)
_AUTO_RULE_QUERY_WEIGHT = 0.35
_AUTO_PREFETCH_MULTIPLIER = 1.6
_AUTO_PREFETCH_EXTRA_CAP = 8
_MAX_SNIPPET_CONTEXT_PER_URL = 3
_RE_CJK_TOKEN = re.compile(r"[\u4e00-\u9fff\u3400-\u4dbf\u3040-\u30ff]")


class SearchStep(StepBase[SearchStepContext]):
    def __init__(
        self,
        *,
        rt: Runtime,
        provider: SearchProviderBase = Depends(),
        ranker: RankerBase = Depends(),
    ) -> None:
        super().__init__(rt=rt)
        self._provider = provider
        self._ranker = ranker
        self.bind_deps(provider, ranker)

    @override
    async def run_inner(self, ctx: SearchStepContext) -> SearchStepContext:
        if bool(ctx.deep.aborted):
            ctx.prefetch.urls = []
            ctx.prefetch.scores = {}
            ctx.fetch.candidates = []
            ctx.output.results = []
            return ctx
        req = ctx.request
        mode = self._normalize_mode(req.mode)
        query_jobs = self._resolve_query_jobs(ctx=ctx, mode=mode)
        if not query_jobs:
            return ctx
        provider_responses: list[SearchProviderResponse | None] = [
            None for _ in query_jobs
        ]
        async with anyio.create_task_group() as tg:
            for idx, job in enumerate(query_jobs):
                tg.start_soon(self._run_query, idx, job.query, provider_responses, ctx)
        query_tokens = tokenize_for_query(req.query)
        include_domains = list(req.include_domains or [])
        exclude_domains = list(req.exclude_domains or [])
        normalized_by_job: list[list[SearchNormalizedResult]] = []
        for response in provider_responses:
            provider_results = response.results if response is not None else []
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
            base_scores = await self._ranker.score_texts(
                docs,
                query=req.query,
                query_tokens=query_tokens,
            )
        canonical_buckets = self._collect_canonical_buckets(
            query_jobs=query_jobs,
            scored_hits=scored_hits,
            base_scores=base_scores,
        )
        snippets_by_url: dict[str, dict[str, SearchSnippetContext]] = {}
        query_hit_stats: dict[str, int] = {}
        coverage_bonus_weight = (
            float(self.settings.search.deep.coverage_bonus_weight)
            if mode == "deep" and bool(self.settings.search.deep.enabled)
            else 0.0
        )
        ranked_with_prefetch: list[tuple[str, float, int]] = []
        for bucket in canonical_buckets.values():
            url = bucket.representative_url
            order = int(bucket.representative_order)
            query_hit_count = int(len(bucket.hit_indexes))
            snippets_by_url[url] = dict(bucket.snippets_by_source)
            query_hit_stats[url] = query_hit_count
            base_score = self._fuse_prefetch_score(
                mode=mode,
                hit_scores=list(bucket.hit_scores),
                hit_query_count=query_hit_count,
                total_query_jobs=len(query_jobs),
            )
            bonus = (
                float(coverage_bonus_weight * math.log1p(float(query_hit_count)))
                if coverage_bonus_weight > 0
                else 0.0
            )
            ranked_with_prefetch.append((url, float(base_score + bonus), order))
        ctx.deep.snippet_context = self._finalize_snippet_context(snippets_by_url)
        ctx.deep.query_hit_stats = dict(query_hit_stats)
        max_results = int(req.max_results or self.settings.search.max_results)
        prefetch_limit = self._resolve_prefetch_limit(
            mode=mode,
            max_results=max_results,
        )
        ranked = sorted(ranked_with_prefetch, key=lambda item: (-item[1], item[2]))
        ctx.prefetch.urls = [url for url, _, _ in ranked[:prefetch_limit]]
        ctx.prefetch.scores = {url: float(score) for url, score, _ in ranked}
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
            weighted_score = float(score * float(job.weight))
            canonical_url = str(hit.item.canonical_url or hit.item.url)
            bucket = buckets.get(canonical_url)
            if bucket is None:
                bucket = SearchCanonicalBucket(
                    representative_url=str(hit.item.url),
                    representative_order=int(hit.order),
                    representative_score=float(weighted_score),
                )
                buckets[canonical_url] = bucket
            bucket.hit_indexes.add(int(hit.job_index))
            bucket.hit_scores.append(float(weighted_score))
            if weighted_score > float(bucket.representative_score) or (
                weighted_score == float(bucket.representative_score)
                and int(hit.order) < int(bucket.representative_order)
            ):
                bucket.representative_score = float(weighted_score)
                bucket.representative_url = str(hit.item.url)
                bucket.representative_order = int(hit.order)
            snippet_text = self._pick_snippet(hit.item)
            if not snippet_text:
                continue
            source_key = str(job.source)
            current = bucket.snippets_by_source.get(source_key)
            if (
                current is None
                or weighted_score > float(current.score)
                or (
                    weighted_score == float(current.score)
                    and int(hit.order) < int(current.order)
                )
            ):
                bucket.snippets_by_source[source_key] = SearchSnippetContext(
                    snippet=snippet_text,
                    source_query=job.query,
                    source_type=job.source,
                    score=float(weighted_score),
                    order=int(hit.order),
                )
        return buckets

    async def _run_query(
        self,
        idx: int,
        query: str,
        out: list[SearchProviderResponse | None],
        ctx: SearchStepContext,
    ) -> None:
        try:
            out[idx] = await self._provider.asearch(
                query=query,
                page=int(ctx.provider_page),
                language=str(ctx.provider_language or ""),
                **dict(ctx.provider_extra_kwargs),
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
        provider_name = self.components.family_name("provider")
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
                    "mode": str(ctx.request.mode),
                    "provider_backend": provider_name,
                },
                meter=MeterPayload(
                    meter_type="search_call",
                    unit="call",
                    quantity=1.0,
                    provider=provider_name,
                ),
            )

    def _resolve_query_jobs(
        self,
        *,
        ctx: SearchStepContext,
        mode: Literal["fast", "auto", "deep"],
    ) -> list[SearchQueryJob]:
        req = ctx.request
        if mode == "deep" and list(ctx.deep.query_jobs or []):
            return list(ctx.deep.query_jobs)
        return self._build_query_jobs(
            query=req.query,
            mode=mode,
            additional_queries=list(req.additional_queries or []),
        )

    def _build_query_jobs(
        self,
        *,
        query: str,
        mode: Literal["fast", "auto", "deep"],
        additional_queries: list[str],
    ) -> list[SearchQueryJob]:
        normalized_query = clean_whitespace(query)
        jobs: list[SearchQueryJob] = [
            SearchQueryJob(query=normalized_query, weight=1.0, source="primary")
        ]
        seen = {normalized_query.casefold()}
        if mode == "auto":
            rule_query = self._build_auto_rule_query(query=normalized_query)
            if rule_query and rule_query.casefold() not in seen:
                seen.add(rule_query.casefold())
                jobs.append(
                    SearchQueryJob(
                        query=rule_query,
                        weight=float(_AUTO_RULE_QUERY_WEIGHT),
                        source="rule",
                    )
                )
            return jobs
        if mode != "deep":
            return jobs
        weight = float(self.settings.search.additional_query_score_weight)
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

    def _build_auto_rule_query(self, *, query: str) -> str:
        tokens = tokenize_for_query(query)
        seen: set[str] = set()
        compact_tokens: list[str] = []
        has_cjk = self._contains_cjk(text=query)
        for raw in tokens:
            token = clean_whitespace(str(raw or ""))
            if not token:
                continue
            key = token.casefold()
            if key in seen:
                continue
            seen.add(key)
            if has_cjk and self._is_cjk_token(token):
                if len(token) < 2:
                    continue
            else:
                if token.isdigit() and len(token) != 4:
                    continue
                if not token.isdigit() and len(token) < 3:
                    continue
            compact_tokens.append(token)
            if len(compact_tokens) >= 5:
                break
        candidate = clean_whitespace(" ".join(compact_tokens))
        if not candidate:
            return ""
        if len(compact_tokens) < 2:
            return ""
        if has_cjk:
            if sum(len(tok) for tok in compact_tokens) < 4:
                return ""
        elif sum(len(tok) for tok in compact_tokens) < 8:
            return ""
        if candidate.casefold() == clean_whitespace(query).casefold():
            return ""
        return candidate

    def _resolve_prefetch_limit(
        self,
        *,
        mode: Literal["fast", "auto", "deep"],
        max_results: int,
    ) -> int:
        if mode == "fast":
            return max(1, int(max_results))
        if mode == "auto":
            desired = int(
                math.ceil(float(max_results) * float(_AUTO_PREFETCH_MULTIPLIER))
            )
            capped = min(desired, int(max_results) + int(_AUTO_PREFETCH_EXTRA_CAP))
            return max(1, capped)
        if not bool(self.settings.search.deep.enabled):
            return max(1, int(max_results))
        cfg = self.settings.search.deep
        desired = int(math.ceil(float(max_results) * float(cfg.prefetch_multiplier)))
        return max(1, min(int(cfg.prefetch_max_urls), desired))

    def _fuse_prefetch_score(
        self,
        *,
        mode: Literal["fast", "auto", "deep"],
        hit_scores: list[float],
        hit_query_count: int,
        total_query_jobs: int,
    ) -> float:
        ordered_scores = sorted((float(x) for x in hit_scores), reverse=True)
        if not ordered_scores:
            return 0.0
        max_score = float(ordered_scores[0])
        coverage = self._coverage_signal(
            hit_query_count=hit_query_count,
            total_query_jobs=total_query_jobs,
        )
        if mode == "fast":
            return max_score
        if mode == "auto":
            top2_mean = self._mean_top_k(ordered_scores, k=2)
            return float(0.65 * max_score + 0.20 * top2_mean + 0.15 * coverage)
        top3_mean = self._mean_top_k(ordered_scores, k=3)
        return float(0.50 * max_score + 0.20 * top3_mean + 0.30 * coverage)

    def _coverage_signal(self, *, hit_query_count: int, total_query_jobs: int) -> float:
        safe_hit = max(0, int(hit_query_count))
        safe_total = max(1, int(total_query_jobs))
        return float(math.log1p(float(safe_hit)) / math.log1p(float(safe_total)))

    def _mean_top_k(self, values: list[float], *, k: int) -> float:
        top_k = list(values[: max(1, int(k))])
        if not top_k:
            return 0.0
        return float(sum(top_k) / len(top_k))

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
            out[url] = selected[: int(_MAX_SNIPPET_CONTEXT_PER_URL)]
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
            canonical_url = self._canonicalize_url(url)
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

    def _canonicalize_url(self, url: str) -> str:
        token = clean_whitespace(url)
        if not token:
            return ""
        try:
            parsed = urlsplit(token)
        except Exception:  # noqa: BLE001
            return token
        scheme = clean_whitespace(parsed.scheme).lower() or "https"
        host = clean_whitespace(str(parsed.hostname or "")).lower()
        if not host:
            return token
        port = self._resolve_port(parsed)
        netloc = self._compose_netloc(scheme=scheme, host=host, port=port)
        path = clean_whitespace(parsed.path) or "/"
        while "//" in path:
            path = path.replace("//", "/")
        if path != "/":
            path = path.rstrip("/") or "/"
        pairs: list[tuple[str, str]] = []
        for key, value in parse_qsl(parsed.query, keep_blank_values=False):
            normalized_key = clean_whitespace(key)
            if not normalized_key:
                continue
            key_lc = normalized_key.casefold()
            if key_lc in _TRACKING_QUERY_KEYS:
                continue
            if any(key_lc.startswith(prefix) for prefix in _TRACKING_QUERY_PREFIXES):
                continue
            pairs.append((normalized_key, clean_whitespace(value)))
        pairs.sort(key=lambda item: (item[0].casefold(), item[1]))
        query = urlencode(pairs, doseq=True)
        return urlunsplit((scheme, netloc, path, query, ""))

    def _is_cjk_token(self, token: str) -> bool:
        return bool(_RE_CJK_TOKEN.search(token))

    def _contains_cjk(self, *, text: str) -> bool:
        return bool(_RE_CJK_TOKEN.search(text))

    def _resolve_port(self, parsed: Any) -> int | None:
        try:
            value = parsed.port
        except Exception:  # noqa: BLE001
            return None
        return int(value) if value is not None else None

    def _compose_netloc(self, *, scheme: str, host: str, port: int | None) -> str:
        if port is None:
            return host
        if (scheme == "http" and port == 80) or (scheme == "https" and port == 443):
            return host
        return f"{host}:{int(port)}"

    def _normalize_mode(self, value: object) -> Literal["fast", "auto", "deep"]:
        token = clean_whitespace(str(value or "")).casefold()
        if token in {"fast", "auto", "deep"}:
            return token  # type: ignore[return-value]
        return "auto"


__all__ = ["SearchStep"]
