from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING, Any
from typing_extensions import override

import anyio
import httpx
import openai

from serpsage.app.response import OverviewResult  # noqa: TC001
from serpsage.contracts.errors import AppError
from serpsage.contracts.protocols import (  # noqa: TC001
    Cache,
    Ranker,
    SearchProvider,
)
from serpsage.pipeline.steps import StepBase, StepContext
from serpsage.text.tokenize import tokenize
from serpsage.text.utils import extract_intent_tokens
from serpsage.util.collections import uniq_preserve_order
from serpsage.util.json import stable_json

if TYPE_CHECKING:
    from serpsage.app.response import ResultItem
    from serpsage.app.runtime import CoreRuntime
    from serpsage.contracts.protocols import Span
    from serpsage.domain.dedupe import ResultDeduper
    from serpsage.domain.enrich import Enricher
    from serpsage.domain.filter import ResultFilterer
    from serpsage.domain.normalize import ResultNormalizer
    from serpsage.domain.overview import OverviewBuilder
    from serpsage.domain.rerank import Reranker


class SearchStep(StepBase):
    span_name = "step.search"

    def __init__(
        self, *, rt: CoreRuntime, provider: SearchProvider, cache: Cache
    ) -> None:
        super().__init__(rt=rt)
        self._provider = provider
        self._cache = cache

    @override
    async def run_inner(self, ctx: StepContext, *, span: Span) -> StepContext:
        try:
            req = ctx.request
            params = dict(req.params or {})
            span.set_attr("provider", "searxng")
            cache_key = hashlib.sha256(
                stable_json(
                    {
                        "provider": "searxng",
                        "q": req.query,
                        "params": params,
                    }
                ).encode("utf-8")
            ).hexdigest()

            cached = await self._cache.aget(namespace="search", key=cache_key)
            if cached:
                payload = json.loads(cached.decode("utf-8"))
                ctx.raw_results = list(payload.get("results") or [])
                span.set_attr("cache_hit", True)
                span.set_attr("raw_results_count", int(len(ctx.raw_results)))
                return ctx

            raw = await self._provider.asearch(query=req.query, params=params)
            ctx.raw_results = raw
            span.set_attr("cache_hit", False)
            span.set_attr("raw_results_count", int(len(ctx.raw_results)))

            await self._cache.aset(
                namespace="search",
                key=cache_key,
                value=json.dumps({"results": raw}, ensure_ascii=False).encode("utf-8"),
                ttl_s=int(self.settings.cache.search_ttl_s),
            )
        except Exception as exc:  # noqa: BLE001
            span.set_attr("cache_hit", False)
            ctx.errors.append(
                AppError(code="search_failed", message=str(exc), details={})
            )
        return ctx


class NormalizeStep(StepBase):
    span_name = "step.normalize"

    def __init__(self, *, rt: CoreRuntime, normalizer: ResultNormalizer) -> None:
        super().__init__(rt=rt)
        self._normalizer = normalizer

    @override
    async def run_inner(self, ctx: StepContext, *, span: Span) -> StepContext:
        span.set_attr("raw_results_count", int(len(ctx.raw_results or [])))
        ctx.results = self._normalizer.normalize_many(ctx.raw_results)
        span.set_attr("results_count", int(len(ctx.results or [])))
        return ctx


class FilterStep(StepBase):
    span_name = "step.filter"

    def __init__(self, *, rt: CoreRuntime, filterer: ResultFilterer) -> None:
        super().__init__(rt=rt)
        self._filterer = filterer

    @override
    async def run_inner(self, ctx: StepContext, *, span: Span) -> StepContext:
        span.set_attr("before_count", int(len(ctx.results or [])))
        outcome = self._filterer.filter(
            query=ctx.request.query,
            explicit_profile=ctx.request.profile,
            results=ctx.results,
        )
        ctx.profile_name = outcome.profile_name
        ctx.profile = outcome.profile
        ctx.query_tokens = list(outcome.query_tokens)
        ctx.results = outcome.results
        span.set_attr("profile_name", str(ctx.profile_name or ""))
        span.set_attr("after_count", int(len(ctx.results or [])))
        return ctx


class DedupeStep(StepBase):
    span_name = "step.dedupe"

    def __init__(self, *, rt: CoreRuntime, deduper: ResultDeduper) -> None:
        super().__init__(rt=rt)
        self._deduper = deduper

    @override
    async def run_inner(self, ctx: StepContext, *, span: Span) -> StepContext:
        span.set_attr("before_count", int(len(ctx.results or [])))
        profile = ctx.profile or self.settings.get_profile(
            self.settings.pipeline.default_profile
        )
        kept, comparisons = self._deduper.dedupe(results=ctx.results, profile=profile)
        ctx.results = kept
        ctx.dedupe_comparisons = int(comparisons)
        span.set_attr("after_count", int(len(ctx.results or [])))
        span.set_attr("comparisons", int(ctx.dedupe_comparisons))
        return ctx


class RankStep(StepBase):
    span_name = "step.rank"

    def __init__(self, *, rt: CoreRuntime, ranker: Ranker) -> None:
        super().__init__(rt=rt)
        self._ranker = ranker

    @override
    async def run_inner(self, ctx: StepContext, *, span: Span) -> StepContext:
        if not ctx.results:
            span.set_attr("items_count", 0)
            return ctx

        query = ctx.request.query
        query_tokens = ctx.query_tokens or tokenize(query)
        ctx.query_tokens = list(query_tokens)

        profile = ctx.profile or self.settings.get_profile(
            self.settings.pipeline.default_profile
        )
        intent_tokens = extract_intent_tokens(query, profile.intent_terms)
        ctx.intent_tokens = list(intent_tokens)

        docs = [f"{r.title} {r.snippet}".strip() for r in ctx.results]
        span.set_attr("items_count", int(len(docs)))
        weights = {
            k: float(v)
            for k, v in (self.settings.rank.providers or {}).items()
            if float(v) > 0
        }
        span.set_attr("providers_used", sorted(weights.keys()))
        span.set_attr("weights", weights)
        raw_scores = self._ranker.score_texts(
            texts=docs,
            query=query,
            query_tokens=list(query_tokens),
            intent_tokens=list(intent_tokens),
        )
        norm = self._ranker.normalize(scores=raw_scores)
        if norm and max(norm) <= 0.0 and max(raw_scores) > 0.0:
            norm = [0.5 for _ in norm]

        for i, r in enumerate(ctx.results):
            r.score = float(norm[i]) if i < len(norm) else 0.0
            title_l = (r.title or "").lower()
            snippet_l = (r.snippet or "").lower()
            hits = [t for t in query_tokens if t in title_l or t in snippet_l]
            r.hit_keywords = uniq_preserve_order(hits)

        ctx.results.sort(key=lambda r: float(r.score), reverse=True)
        return ctx


class EnrichStep(StepBase):
    span_name = "step.enrich"

    def __init__(self, *, rt: CoreRuntime, enricher: Enricher) -> None:
        super().__init__(rt=rt)
        self._enricher = enricher

    @override
    async def run_inner(self, ctx: StepContext, *, span: Span) -> StepContext:
        depth = ctx.request.depth
        span.set_attr("depth", str(depth))
        if depth == "simple":
            return ctx
        if not self.settings.enrich.enabled:
            return ctx
        if not ctx.results:
            return ctx

        preset = self.settings.enrich.depth_presets.get(depth)  # type: ignore[index]
        if preset is None:
            return ctx

        n = len(ctx.results)
        target = int(math.ceil(n * float(preset.pages_ratio)))
        m = max(int(preset.min_pages), min(int(preset.max_pages), target))
        m = min(m, n)
        if m <= 0:
            return ctx

        work = ctx.results[:m]
        span.set_attr("items_considered", int(m))
        query = ctx.request.query
        query_tokens = ctx.query_tokens or tokenize(query)
        ctx.query_tokens = list(query_tokens)
        profile = ctx.profile or self.settings.get_profile(
            self.settings.pipeline.default_profile
        )
        top_k = int(preset.top_chunks_per_page)

        async def enrich_one(r: ResultItem) -> None:
            r.page = await self._enricher.enrich_one(
                result=r,
                query=query,
                query_tokens=list(query_tokens),
                profile=profile,
                top_k=top_k,
            )

        async with anyio.create_task_group() as tg:
            for r in work:
                tg.start_soon(enrich_one, r)
        span.set_attr("pages_enriched", int(m))
        return ctx


class RerankStep(StepBase):
    span_name = "step.rerank"

    def __init__(self, *, rt: CoreRuntime, reranker: Reranker) -> None:
        super().__init__(rt=rt)
        self._reranker = reranker

    @override
    async def run_inner(self, ctx: StepContext, *, span: Span) -> StepContext:
        if not ctx.results:
            span.set_attr("items_count", 0)
            return ctx

        query = ctx.request.query
        query_tokens = ctx.query_tokens or tokenize(query)
        ctx.query_tokens = list(query_tokens)

        profile = ctx.profile or self.settings.get_profile(
            self.settings.pipeline.default_profile
        )
        intent_tokens = ctx.intent_tokens or extract_intent_tokens(
            query, profile.intent_terms
        )
        ctx.intent_tokens = list(intent_tokens)

        ctx.results = self._reranker.rerank(
            results=ctx.results,
            query=query,
            query_tokens=list(query_tokens),
            intent_tokens=list(intent_tokens),
        )
        span.set_attr("items_count", int(len(ctx.results or [])))
        return ctx


class OverviewStep(StepBase):
    span_name = "step.overview"

    def __init__(
        self, *, rt: CoreRuntime, builder: OverviewBuilder, cache: Cache
    ) -> None:
        super().__init__(rt=rt)
        self._builder = builder
        self._cache = cache

    @override
    async def run_inner(self, ctx: StepContext, *, span: Span) -> StepContext:
        enabled = self.settings.overview.enabled
        if ctx.request.overview is not None:
            enabled = bool(ctx.request.overview)
        if not enabled:
            return ctx
        if not ctx.results:
            return ctx
        if not self.settings.overview.llm.api_key:
            ctx.errors.append(
                AppError(
                    code="overview_skipped",
                    message="LLM api_key not configured; skipping overview",
                    details={},
                )
            )
            return ctx

        llm_cfg = self.settings.overview.llm
        model = llm_cfg.model
        messages = self._builder.build_messages(
            query=ctx.request.query, results=ctx.results
        )
        schema = self._builder.schema()

        prompt_chars = sum(len(str(m.get("content") or "")) for m in messages)
        span.set_attr("model", model)
        span.set_attr("schema_strict", bool(self.settings.overview.schema_strict))
        span.set_attr("prompt_chars", int(prompt_chars))
        span.set_attr(
            "max_summary_tokens", int(self.settings.overview.max_output_tokens)
        )

        cache_ttl_s = int(self.settings.overview.cache_ttl_s)
        cache_key: str | None = None
        if cache_ttl_s > 0:
            cache_key = self._overview_cache_key(
                model=model,
                messages=messages,
                schema=schema,
                schema_strict=bool(self.settings.overview.schema_strict),
            )
            cached = await self._cache.aget(namespace="overview", key=cache_key)
            if cached:
                span.set_attr("cache_hit", True)
                try:
                    ctx.overview = OverviewResult.model_validate_json(cached)
                except Exception:  # noqa: BLE001
                    # Corrupted/old cache; ignore and recompute.
                    span.add_event("overview.cache_corrupt")
                else:
                    return ctx
        span.set_attr("cache_hit", False)
        try:
            overview = await self._builder.build_overview(
                query=ctx.request.query, results=ctx.results
            )
            ctx.overview = overview

            if cache_ttl_s > 0 and cache_key:
                await self._cache.aset(
                    namespace="overview",
                    key=cache_key,
                    value=overview.model_dump_json().encode("utf-8"),
                    ttl_s=cache_ttl_s,
                )
        except Exception as exc:  # noqa: BLE001
            retries = max(0, int(self.settings.overview.self_heal_retries))
            code, details = self._map_overview_error(
                exc if isinstance(exc, Exception) else Exception(str(exc)),
                model=model,
                base_url=str(llm_cfg.base_url),
                attempt=retries,
            )
            ctx.errors.append(AppError(code=code, message=str(exc), details=details))
        return ctx

    def _overview_cache_key(
        self,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        schema_strict: bool,
    ) -> str:
        payload = {
            "model": model,
            "messages": messages,
            "schema": schema,
            "schema_strict": bool(schema_strict),
        }
        return hashlib.sha256(stable_json(payload).encode("utf-8")).hexdigest()

    def _map_overview_error(
        self, exc: Exception, *, model: str, base_url: str, attempt: int
    ) -> tuple[str, dict[str, Any]]:
        details: dict[str, Any] = {
            "model": model,
            "base_url": base_url,
            "attempt": int(attempt),
            "type": type(exc).__name__,
        }

        request_id = getattr(exc, "request_id", None)
        if request_id:
            details["request_id"] = str(request_id)

        status = getattr(exc, "status_code", None)
        if status is not None:
            details["status_code"] = int(status)

        code = "overview_failed"
        if isinstance(exc, openai.RateLimitError):
            code = "overview_rate_limited"
        elif isinstance(exc, openai.APITimeoutError):
            code = "overview_timeout"
        elif isinstance(exc, openai.AuthenticationError):
            code = "overview_auth_failed"
        elif isinstance(exc, openai.BadRequestError):
            code = "overview_bad_request"
        elif isinstance(exc, openai.APIStatusError):
            sc = getattr(exc, "status_code", None)
            if sc is not None and 500 <= int(sc) < 600:
                code = "overview_server_error"
            else:
                code = "overview_failed"
        elif isinstance(exc, (openai.APIConnectionError, httpx.TimeoutException)):
            code = "overview_timeout"
        return code, details


__all__ = [
    "DedupeStep",
    "EnrichStep",
    "FilterStep",
    "NormalizeStep",
    "OverviewStep",
    "RankStep",
    "RerankStep",
    "SearchStep",
]
