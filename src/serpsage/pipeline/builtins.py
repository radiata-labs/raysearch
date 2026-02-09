from __future__ import annotations

import hashlib
import json
import math
from typing import TYPE_CHECKING
from typing_extensions import override

import anyio

from serpsage.contracts.errors import AppError
from serpsage.contracts.protocols import (  # noqa: TC001
    Cache,
    LLMClient,
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
    async def run_inner(self, ctx: StepContext) -> StepContext:
        try:
            req = ctx.request
            params = dict(req.params or {})
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
                return ctx

            raw = await self._provider.asearch(query=req.query, params=params)
            ctx.raw_results = raw

            await self._cache.aset(
                namespace="search",
                key=cache_key,
                value=json.dumps({"results": raw}, ensure_ascii=False).encode("utf-8"),
                ttl_s=int(self.settings.cache.search_ttl_s),
            )
        except Exception as exc:  # noqa: BLE001
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
    async def run_inner(self, ctx: StepContext) -> StepContext:
        ctx.results = self._normalizer.normalize_many(ctx.raw_results)
        return ctx


class FilterStep(StepBase):
    span_name = "step.filter"

    def __init__(self, *, rt: CoreRuntime, filterer: ResultFilterer) -> None:
        super().__init__(rt=rt)
        self._filterer = filterer

    @override
    async def run_inner(self, ctx: StepContext) -> StepContext:
        outcome = self._filterer.filter(
            query=ctx.request.query,
            explicit_profile=ctx.request.profile,
            results=ctx.results,
        )
        ctx.profile_name = outcome.profile_name
        ctx.profile = outcome.profile
        ctx.query_tokens = list(outcome.query_tokens)
        ctx.results = outcome.results
        return ctx


class DedupeStep(StepBase):
    span_name = "step.dedupe"

    def __init__(self, *, rt: CoreRuntime, deduper: ResultDeduper) -> None:
        super().__init__(rt=rt)
        self._deduper = deduper

    @override
    async def run_inner(self, ctx: StepContext) -> StepContext:
        profile = ctx.profile or self.settings.get_profile(
            self.settings.pipeline.default_profile
        )
        kept, comparisons = self._deduper.dedupe(results=ctx.results, profile=profile)
        ctx.results = kept
        ctx.dedupe_comparisons = int(comparisons)
        return ctx


class RankStep(StepBase):
    span_name = "step.rank"

    def __init__(self, *, rt: CoreRuntime, ranker: Ranker) -> None:
        super().__init__(rt=rt)
        self._ranker = ranker

    @override
    async def run_inner(self, ctx: StepContext) -> StepContext:
        if not ctx.results:
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
    async def run_inner(self, ctx: StepContext) -> StepContext:
        depth = ctx.request.depth
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
        return ctx


class RerankStep(StepBase):
    span_name = "step.rerank"

    def __init__(self, *, rt: CoreRuntime, reranker: Reranker) -> None:
        super().__init__(rt=rt)
        self._reranker = reranker

    @override
    async def run_inner(self, ctx: StepContext) -> StepContext:
        if not ctx.results:
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
        return ctx


class OverviewStep(StepBase):
    span_name = "step.overview"

    def __init__(
        self, *, rt: CoreRuntime, llm: LLMClient, builder: OverviewBuilder
    ) -> None:
        super().__init__(rt=rt)
        self._llm = llm
        self._builder = builder

    @override
    async def run_inner(self, ctx: StepContext) -> StepContext:
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
        messages = self._builder.build_messages(
            query=ctx.request.query, results=ctx.results
        )
        schema = self._builder.schema()
        try:
            data = await self._llm.chat_json(
                model=llm_cfg.model,
                messages=messages,
                schema=schema,
                timeout_s=float(llm_cfg.timeout_s),
            )
            ctx.overview = self._builder.parse(data)
        except Exception as exc:  # noqa: BLE001
            ctx.errors.append(
                AppError(code="overview_failed", message=str(exc), details={})
            )
        return ctx


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
