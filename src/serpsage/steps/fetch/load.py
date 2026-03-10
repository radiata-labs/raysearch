from __future__ import annotations

import hashlib
import json
import time
from contextlib import suppress
from typing import Any
from typing_extensions import override

from serpsage.components.cache import CacheBase
from serpsage.components.cache.base import CacheConfigBase
from serpsage.components.crawl import CrawlerBase
from serpsage.components.crawl.base import CrawlConfigBase
from serpsage.dependencies import Inject
from serpsage.models.app.response import FetchErrorTag
from serpsage.models.components.crawl import CrawlResult
from serpsage.models.components.telemetry import MeterPayload
from serpsage.models.steps.fetch import FetchStepContext
from serpsage.steps.base import StepBase

_CACHE_NAMESPACE = "fetch:v4"


class FetchLoadStep(StepBase[FetchStepContext]):
    crawler: CrawlerBase = Inject()
    cache: CacheBase = Inject()

    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        if ctx.error.failed:
            return ctx
        url = (ctx.url or "").strip()
        if not url:
            await self._fail(
                ctx,
                code="fetch_load_failed",
                message="empty url",
                stage="load",
                source=None,
            )
            return ctx
        mode = str(ctx.page.crawl_mode or "fallback")
        crawl_cfg = self.components.resolve_default_config(
            "crawl", expected_type=CrawlConfigBase
        )
        cache_cfg = self.components.resolve_default_config(
            "cache", expected_type=CacheConfigBase
        )
        cache_key = _cache_key(
            url=url,
            backend=self.components.family_name("crawl"),
        )
        timeout_s = float(ctx.page.crawl_timeout_s or 0.0) or float(crawl_cfg.timeout_s)
        cache_fetch_ms = 0
        crawl_fetch_ms = 0

        async def get_cached() -> CrawlResult | None:
            nonlocal cache_fetch_ms
            t0 = time.monotonic()
            payload = await self.cache.aget(namespace=_CACHE_NAMESPACE, key=cache_key)
            cache_fetch_ms = int((time.monotonic() - t0) * 1000)
            if not payload:
                return None
            try:
                return _decode_fetch_cache(payload, url=url)
            except Exception:
                return None

        async def crawl_once() -> CrawlResult:
            nonlocal crawl_fetch_ms
            t0 = time.monotonic()
            result = await self.crawler.acrawl(
                url=url,
                timeout_s=float(timeout_s),
            )
            crawl_fetch_ms = int((time.monotonic() - t0) * 1000)
            return result

        async def write_cache(result: CrawlResult) -> None:
            ttl_s = int(cache_cfg.fetch_ttl_s)
            if ttl_s <= 0:
                return
            await self.cache.aset(
                namespace=_CACHE_NAMESPACE,
                key=cache_key,
                value=_encode_fetch_cache(result),
                ttl_s=ttl_s,
            )

        source: str | None = None
        fetched: CrawlResult | None = None
        crawl_exc: Exception | None = None
        if mode == "never":
            fetched = await get_cached()
            if fetched is None:
                await self._emit_load_event(
                    ctx=ctx,
                    event_name="fetch.load.cache_miss",
                    status="error",
                    attrs={"mode": mode, "cache_key": cache_key},
                )
                await self._fail(
                    ctx,
                    code="fetch_cache_miss",
                    message="cache miss in crawl_mode=never",
                    stage="load",
                    source="cache",
                )
                return ctx
            await self._emit_load_event(
                ctx=ctx,
                event_name="fetch.load.cache_hit",
                attrs={
                    "mode": mode,
                    "cache_key": cache_key,
                    "latency_ms": cache_fetch_ms,
                },
            )
            source = "cache"
        elif mode == "always":
            try:
                fetched = await crawl_once()
            except Exception as exc:  # noqa: BLE001
                crawl_exc = exc if isinstance(exc, Exception) else Exception(str(exc))
                await self._emit_load_event(
                    ctx=ctx,
                    event_name="fetch.load.crawl_error",
                    status="error",
                    error_type=type(crawl_exc).__name__,
                    attrs={"mode": mode, "cache_key": cache_key},
                )
            if fetched is None:
                await self._fail(
                    ctx,
                    code="fetch_crawl_failed",
                    message=str(crawl_exc or "crawl failed"),
                    stage="load",
                    source="crawl",
                )
                return ctx
            await write_cache(fetched)
            await self._emit_load_event(
                ctx=ctx,
                event_name="fetch.load.crawl_success",
                attrs={
                    "mode": mode,
                    "cache_key": cache_key,
                    "latency_ms": crawl_fetch_ms,
                },
            )
            source = "crawl"
        elif mode == "preferred":
            try:
                fetched = await crawl_once()
            except Exception as exc:  # noqa: BLE001
                crawl_exc = exc if isinstance(exc, Exception) else Exception(str(exc))
                await self._emit_load_event(
                    ctx=ctx,
                    event_name="fetch.load.crawl_error",
                    status="error",
                    error_type=type(crawl_exc).__name__,
                    attrs={"mode": mode, "cache_key": cache_key},
                )
            if fetched is None:
                cached = await get_cached()
                if cached is None:
                    await self._emit_load_event(
                        ctx=ctx,
                        event_name="fetch.load.cache_miss",
                        status="error",
                        attrs={"mode": mode, "cache_key": cache_key},
                    )
                    await self._fail(
                        ctx,
                        code="fetch_crawl_failed",
                        message=str(crawl_exc or "crawl failed and cache miss"),
                        stage="load",
                        source="crawl",
                    )
                    return ctx
                fetched = cached
                await self._emit_load_event(
                    ctx=ctx,
                    event_name="fetch.load.cache_hit",
                    attrs={
                        "mode": mode,
                        "cache_key": cache_key,
                        "latency_ms": cache_fetch_ms,
                    },
                )
                source = "cache"
            else:
                await write_cache(fetched)
                await self._emit_load_event(
                    ctx=ctx,
                    event_name="fetch.load.crawl_success",
                    attrs={
                        "mode": mode,
                        "cache_key": cache_key,
                        "latency_ms": crawl_fetch_ms,
                    },
                )
                source = "crawl"
        else:
            cached = await get_cached()
            if cached is not None:
                fetched = cached
                await self._emit_load_event(
                    ctx=ctx,
                    event_name="fetch.load.cache_hit",
                    attrs={
                        "mode": mode,
                        "cache_key": cache_key,
                        "latency_ms": cache_fetch_ms,
                    },
                )
                source = "cache"
            else:
                await self._emit_load_event(
                    ctx=ctx,
                    event_name="fetch.load.cache_miss",
                    attrs={
                        "mode": mode,
                        "cache_key": cache_key,
                        "latency_ms": cache_fetch_ms,
                    },
                )
                try:
                    fetched = await crawl_once()
                except Exception as exc:  # noqa: BLE001
                    crawl_exc = (
                        exc if isinstance(exc, Exception) else Exception(str(exc))
                    )
                    await self._emit_load_event(
                        ctx=ctx,
                        event_name="fetch.load.crawl_error",
                        status="error",
                        error_type=type(crawl_exc).__name__,
                        attrs={"mode": mode, "cache_key": cache_key},
                    )
                if fetched is None:
                    await self._fail(
                        ctx,
                        code="fetch_crawl_failed",
                        message=str(crawl_exc or "crawl failed"),
                        stage="load",
                        source="crawl",
                    )
                    return ctx
                await write_cache(fetched)
                await self._emit_load_event(
                    ctx=ctx,
                    event_name="fetch.load.crawl_success",
                    attrs={
                        "mode": mode,
                        "cache_key": cache_key,
                        "latency_ms": crawl_fetch_ms,
                    },
                )
                source = "crawl"
        assert fetched is not None
        await self._emit_fetch_meter(ctx=ctx, source=str(source or "unknown"))
        ctx.page.raw = fetched
        return ctx

    async def _fail(
        self,
        ctx: FetchStepContext,
        *,
        code: str,
        message: str,
        stage: str,
        source: str | None,
        tag: FetchErrorTag | None = None,
    ) -> None:
        ctx.error.failed = True
        ctx.error.tag = tag or _resolve_error_tag(code=code, message=message)
        ctx.error.detail = str(message or "").strip() or None
        details: dict[str, Any] = {
            "url": ctx.url,
            "url_index": ctx.url_index,
            "stage": stage,
            "fatal": True,
            "crawl_mode": ctx.page.crawl_mode,
        }
        if source:
            details["source"] = source
        await self.emit_tracking_event(
            event_name="fetch.load.error",
            request_id=ctx.request_id,
            stage=stage,
            status="error",
            error_code=code,
            attrs={
                **details,
                "message": message,
                "error_tag": ctx.error.tag,
            },
        )

    async def _emit_load_event(
        self,
        *,
        ctx: FetchStepContext,
        event_name: str,
        status: str = "ok",
        error_type: str = "",
        attrs: dict[str, Any] | None = None,
    ) -> None:
        telemetry = self.telemetry
        if telemetry is None:
            return
        payload = {
            "url": ctx.url,
            "url_index": int(ctx.url_index),
            "crawl_mode": str(ctx.page.crawl_mode),
        }
        if attrs:
            payload.update(attrs)
        with suppress(Exception):
            await telemetry.emit(
                event_name=event_name,
                status="error" if status == "error" else "ok",
                request_id=ctx.request_id,
                component="fetch_load",
                stage="load",
                error_type=error_type,
                attrs=payload,
            )

    async def _emit_fetch_meter(self, *, ctx: FetchStepContext, source: str) -> None:
        telemetry = self.telemetry
        if telemetry is None:
            return
        with suppress(Exception):
            await telemetry.emit(
                event_name="meter.usage.fetch_page",
                status="ok",
                request_id=ctx.request_id,
                component="fetch_load",
                stage="load",
                idempotency_key=f"{ctx.request_id}:meter.usage.fetch_page:{ctx.url_index}",
                attrs={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "source": source,
                    "crawl_mode": str(ctx.page.crawl_mode),
                },
                meter=MeterPayload(
                    meter_type="fetch_page",
                    unit="page",
                    quantity=1.0,
                ),
            )


def _cache_key(*, url: str, backend: str) -> str:
    payload = json.dumps(
        {
            "url": str(url),
            "backend": str(backend),
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _encode_fetch_cache(result: CrawlResult) -> bytes:
    return json.dumps(
        {
            "url": str(result.url),
            "status_code": int(result.status_code),
            "content_type": result.content_type,
            "content_hex": bytes(result.content or b"").hex(),
            "crawl_backend": str(result.crawl_backend),
            "rendered": bool(result.rendered),
            "content_kind": str(result.content_kind),
            "headers": {str(k): str(v) for k, v in dict(result.headers or {}).items()},
            "attempt_chain": [str(x) for x in list(result.attempt_chain or [])],
        },
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


def _decode_fetch_cache(payload: bytes, *, url: str) -> CrawlResult:
    obj = json.loads(payload.decode("utf-8"))
    return CrawlResult(
        url=str(obj.get("url") or url),
        status_code=int(obj.get("status_code") or 0),
        content_type=obj.get("content_type"),
        content=bytes.fromhex(str(obj.get("content_hex") or "")),
        crawl_backend=str(obj.get("crawl_backend") or "curl_cffi"),  # type: ignore[arg-type]
        rendered=bool(obj.get("rendered", False)),
        content_kind=str(obj.get("content_kind") or "unknown"),  # type: ignore[arg-type]
        headers={str(k): str(v) for k, v in dict(obj.get("headers") or {}).items()},
        attempt_chain=[str(x) for x in list(obj.get("attempt_chain") or [])],
    )


def _resolve_error_tag(*, code: str, message: str) -> FetchErrorTag:
    if code in {"fetch_load_failed", "fetch_cache_miss"}:
        return "SOURCE_NOT_AVAILABLE"
    if code != "fetch_crawl_failed":
        return "CRAWL_UNKNOWN_ERROR"
    lowered = str(message or "").casefold()
    if "livecrawl" in lowered and "timeout" in lowered:
        return "CRAWL_LIVECRAWL_TIMEOUT"
    if "timeout" in lowered or "timed out" in lowered or "deadline" in lowered:
        return "CRAWL_TIMEOUT"
    if "404" in lowered or "not found" in lowered:
        return "CRAWL_NOT_FOUND"
    return "CRAWL_UNKNOWN_ERROR"


__all__ = ["FetchLoadStep"]
