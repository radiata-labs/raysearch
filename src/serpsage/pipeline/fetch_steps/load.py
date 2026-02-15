from __future__ import annotations

import hashlib
import json
import time
from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.models.errors import AppError
from serpsage.models.fetch import FetchResult
from serpsage.models.pipeline import FetchStepContext
from serpsage.pipeline.step import PipelineStep

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.contracts.services import CacheBase, FetcherBase
    from serpsage.core.runtime import Runtime

_CACHE_NAMESPACE = "fetch:v3"


class FetchLoadStep(PipelineStep[FetchStepContext]):
    span_name = "step.fetch_load"

    def __init__(
        self,
        *,
        rt: Runtime,
        fetcher: FetcherBase,
        cache: CacheBase,
    ) -> None:
        super().__init__(rt=rt)
        self._fetcher = fetcher
        self._cache = cache
        self.bind_deps(fetcher, cache)

    @override
    async def run_inner(
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        if ctx.fatal:
            return ctx
        url = (ctx.url or "").strip()
        if not url:
            self._fail(
                ctx,
                code="fetch_load_failed",
                message="empty url",
                stage="load",
                source=None,
            )
            return ctx

        mode = str(ctx.others_runtime.crawl_mode or "fallback")
        cache_key = _cache_key(
            url=url,
            backend=str(self.settings.fetch.backend or "auto").lower(),
            allow_render=bool(ctx.others_runtime.allow_render),
        )
        timeout_s = float(ctx.others_runtime.crawl_timeout_s or 0.0) or float(
            self.settings.fetch.timeout_s
        )
        span.set_attr("crawl_mode", mode)
        span.set_attr("crawl_timeout_s", float(timeout_s))
        span.set_attr("allow_render", bool(ctx.others_runtime.allow_render))
        span.set_attr("rank_index", int(ctx.others_runtime.rank_index))

        cache_fetch_ms = 0
        crawl_fetch_ms = 0

        async def get_cached() -> FetchResult | None:
            nonlocal cache_fetch_ms
            t0 = time.monotonic()
            payload = await self._cache.aget(namespace=_CACHE_NAMESPACE, key=cache_key)
            cache_fetch_ms = int((time.monotonic() - t0) * 1000)
            if not payload:
                return None
            try:
                return _decode_fetch_cache(payload, url=url)
            except Exception:
                return None

        async def crawl_once() -> FetchResult:
            nonlocal crawl_fetch_ms
            t0 = time.monotonic()
            result = await self._fetcher.afetch(
                url=url,
                timeout_s=float(timeout_s),
                allow_render=bool(ctx.others_runtime.allow_render),
                rank_index=int(ctx.others_runtime.rank_index),
            )
            crawl_fetch_ms = int((time.monotonic() - t0) * 1000)
            return result

        async def write_cache(result: FetchResult) -> None:
            ttl_s = int(self.settings.cache.fetch_ttl_s)
            if ttl_s <= 0:
                return
            await self._cache.aset(
                namespace=_CACHE_NAMESPACE,
                key=cache_key,
                value=_encode_fetch_cache(result),
                ttl_s=ttl_s,
            )

        source: str | None = None
        fetched: FetchResult | None = None
        crawl_exc: Exception | None = None

        if mode == "never":
            fetched = await get_cached()
            if fetched is None:
                self._fail(
                    ctx,
                    code="fetch_cache_miss",
                    message="cache miss in crawl_mode=never",
                    stage="load",
                    source="cache",
                )
                span.set_attr("cache_hit", False)
                return ctx
            source = "cache"
            span.set_attr("cache_hit", True)
        elif mode == "always":
            try:
                fetched = await crawl_once()
            except Exception as exc:  # noqa: BLE001
                crawl_exc = exc if isinstance(exc, Exception) else Exception(str(exc))
            if fetched is None:
                self._fail(
                    ctx,
                    code="fetch_crawl_failed",
                    message=str(crawl_exc or "crawl failed"),
                    stage="load",
                    source="crawl",
                )
                span.set_attr("cache_hit", False)
                return ctx
            await write_cache(fetched)
            source = "crawl"
            span.set_attr("cache_hit", False)
        elif mode == "preferred":
            try:
                fetched = await crawl_once()
            except Exception as exc:  # noqa: BLE001
                crawl_exc = exc if isinstance(exc, Exception) else Exception(str(exc))
            if fetched is None:
                cached = await get_cached()
                if cached is None:
                    self._fail(
                        ctx,
                        code="fetch_crawl_failed",
                        message=str(crawl_exc or "crawl failed and cache miss"),
                        stage="load",
                        source="crawl",
                    )
                    span.set_attr("cache_hit", False)
                    return ctx
                fetched = cached
                source = "cache"
                span.set_attr("cache_hit", True)
            else:
                await write_cache(fetched)
                source = "crawl"
                span.set_attr("cache_hit", False)
        else:
            cached = await get_cached()
            if cached is not None:
                fetched = cached
                source = "cache"
                span.set_attr("cache_hit", True)
            else:
                try:
                    fetched = await crawl_once()
                except Exception as exc:  # noqa: BLE001
                    crawl_exc = (
                        exc if isinstance(exc, Exception) else Exception(str(exc))
                    )
                if fetched is None:
                    self._fail(
                        ctx,
                        code="fetch_crawl_failed",
                        message=str(crawl_exc or "crawl failed"),
                        stage="load",
                        source="crawl",
                    )
                    span.set_attr("cache_hit", False)
                    return ctx
                await write_cache(fetched)
                source = "crawl"
                span.set_attr("cache_hit", False)

        assert fetched is not None
        ctx.fetch_result = fetched
        span.set_attr("source", str(source or "unknown"))
        span.set_attr("status_code", int(fetched.status_code))
        span.set_attr("fetch_mode", str(fetched.fetch_mode))
        span.set_attr("content_kind", str(fetched.content_kind))
        span.set_attr("cache_fetch_ms", int(cache_fetch_ms))
        span.set_attr("crawl_fetch_ms", int(crawl_fetch_ms))
        return ctx

    def _fail(
        self,
        ctx: FetchStepContext,
        *,
        code: str,
        message: str,
        stage: str,
        source: str | None,
    ) -> None:
        ctx.fatal = True
        details: dict[str, Any] = {
            "url": ctx.url,
            "url_index": ctx.url_index,
            "stage": stage,
            "fatal": True,
            "crawl_mode": ctx.others_runtime.crawl_mode,
        }
        if source:
            details["source"] = source
        ctx.errors.append(
            AppError(
                code=code,
                message=message,
                details=details,
            )
        )


def _cache_key(*, url: str, backend: str, allow_render: bool) -> str:
    payload = json.dumps(
        {
            "url": str(url),
            "backend": str(backend),
            "allow_render": bool(allow_render),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _encode_fetch_cache(result: FetchResult) -> bytes:
    return json.dumps(
        {
            "url": str(result.url),
            "status_code": int(result.status_code),
            "content_type": result.content_type,
            "content_hex": bytes(result.content or b"").hex(),
            "fetch_mode": str(result.fetch_mode),
            "rendered": bool(result.rendered),
            "content_kind": str(result.content_kind),
            "headers": {str(k): str(v) for k, v in dict(result.headers or {}).items()},
            "attempt_chain": [str(x) for x in list(result.attempt_chain or [])],
            "quality_score": float(result.quality_score or 0.0),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _decode_fetch_cache(payload: bytes, *, url: str) -> FetchResult:
    obj = json.loads(payload.decode("utf-8"))
    return FetchResult(
        url=str(obj.get("url") or url),
        status_code=int(obj.get("status_code") or 0),
        content_type=obj.get("content_type"),
        content=bytes.fromhex(str(obj.get("content_hex") or "")),
        fetch_mode=str(obj.get("fetch_mode") or "httpx"),  # type: ignore[arg-type]
        rendered=bool(obj.get("rendered", False)),
        content_kind=str(obj.get("content_kind") or "unknown"),  # type: ignore[arg-type]
        headers={str(k): str(v) for k, v in dict(obj.get("headers") or {}).items()},
        attempt_chain=[str(x) for x in list(obj.get("attempt_chain") or [])],
        quality_score=float(obj.get("quality_score") or 0.0),
    )


__all__ = ["FetchLoadStep"]
