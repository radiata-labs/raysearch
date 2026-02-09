from __future__ import annotations

import hashlib
import json
from typing import TYPE_CHECKING

from serpsage.contracts.base import WorkUnit
from serpsage.contracts.errors import AppError
from serpsage.util.json import stable_json

if TYPE_CHECKING:
    from serpsage.contracts.protocols import Cache, SearchProvider
    from serpsage.pipeline.steps import StepContext


class SearchStep(WorkUnit):
    def __init__(self, *, rt, provider: SearchProvider, cache: Cache) -> None:  # noqa: ANN001
        super().__init__(rt=rt)
        self._provider = provider
        self._cache = cache

    async def run(self, ctx: StepContext) -> StepContext:
        with self.span("step.search"):
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
                    value=json.dumps({"results": raw}, ensure_ascii=False).encode(
                        "utf-8"
                    ),
                    ttl_s=int(self.settings.cache.search_ttl_s),
                )
            except Exception as exc:  # noqa: BLE001
                ctx.errors.append(
                    AppError(code="search_failed", message=str(exc), details={})
                )
            else:
                return ctx
            return ctx


__all__ = ["SearchStep"]
