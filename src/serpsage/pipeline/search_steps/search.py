from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.errors import AppError
from serpsage.models.pipeline import SearchStepContext
from serpsage.pipeline.step import PipelineStep
from serpsage.utils import stable_json

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.contracts.services import CacheBase, SearchProviderBase
    from serpsage.core.runtime import Runtime


class SearchStep(PipelineStep[SearchStepContext]):
    span_name = "step.search"

    def __init__(
        self, *, rt: Runtime, provider: SearchProviderBase, cache: CacheBase
    ) -> None:
        super().__init__(rt=rt)
        self._provider = provider
        self._cache = cache
        self.bind_deps(provider, cache)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
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


__all__ = ["SearchStep"]
