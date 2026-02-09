from __future__ import annotations

import hashlib
import json
from typing import Any

from serpsage.app.container import Container
from serpsage.contracts.errors import AppError
from serpsage.contracts.protocols import stable_json
from serpsage.pipeline.steps import StepContext


class SearchStep:
    def __init__(self, container: Container) -> None:
        self._c = container

    async def run(self, ctx: StepContext) -> StepContext:
        span = self._c.telemetry.start_span("step.search")
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

            cached = await self._c.cache.aget(namespace="search", key=cache_key)
            if cached:
                payload = json.loads(cached.decode("utf-8"))
                ctx.raw_results = list(payload.get("results") or [])
                span.set_attr("cache_hit", True)
                return ctx

            raw = await self._c.provider.asearch(query=req.query, params=params)
            ctx.raw_results = raw

            await self._c.cache.aset(
                namespace="search",
                key=cache_key,
                value=json.dumps({"results": raw}, ensure_ascii=False).encode("utf-8"),
                ttl_s=int(ctx.settings.cache.search_ttl_s),
            )
            span.set_attr("cache_hit", False)
            span.set_attr("n_raw", len(raw))
            return ctx
        except Exception as exc:  # noqa: BLE001
            ctx.errors.append(
                AppError(code="search_failed", message=str(exc), details={})
            )
            return ctx
        finally:
            span.end()


__all__ = ["SearchStep"]

