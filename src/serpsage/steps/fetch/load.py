from __future__ import annotations

import hashlib
import json
import time
from typing_extensions import override

from serpsage.components.cache import CacheBase
from serpsage.components.cache.base import CacheConfigBase
from serpsage.components.crawl import CrawlerBase
from serpsage.components.crawl.base import CrawlerConfigBase
from serpsage.dependencies import Depends
from serpsage.models.app.response import FetchErrorTag
from serpsage.models.components.crawl import CrawlResult
from serpsage.models.steps.fetch import FetchStepContext
from serpsage.steps.base import StepBase

_CACHE_NAMESPACE = "fetch:v4"


class FetchLoadStep(StepBase[FetchStepContext]):
    crawler: CrawlerBase = Depends()
    cache: CacheBase = Depends()

    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        if ctx.error.failed:
            return ctx
        url = (ctx.url or "").strip()
        if not url:
            ctx.error.failed = True
            ctx.error.tag = "SOURCE_NOT_AVAILABLE"
            ctx.error.detail = "empty url"
            await self.tracker.error(
                name="fetch.load.failed",
                request_id=ctx.request_id,
                step="fetch.load",
                error_code="fetch_load_failed",
                error_message="empty url",
                data={
                    "error_tag": ctx.error.tag,
                    "url": ctx.url,
                    "url_index": ctx.url_index,
                    "crawl_mode": ctx.page.crawl_mode,
                    "source": "",
                    "fatal": True,
                },
            )
            return ctx
        mode = str(ctx.page.crawl_mode or "fallback")
        registry = self.require_components()
        default_crawl_cfg = registry.resolve_default_config(
            "crawl", expected_type=CrawlerConfigBase
        )
        default_cache_cfg = registry.resolve_default_config(
            "cache", expected_type=CacheConfigBase
        )
        cache_key = _cache_key(
            url=url,
            backend=registry.family_name("crawl"),
        )
        timeout_s = float(ctx.page.crawl_timeout_s or 0.0) or float(
            default_crawl_cfg.timeout_s
        )
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
            ttl_s = int(default_cache_cfg.fetch_ttl_s)
            if ttl_s <= 0:
                return
            await self.cache.aset(
                namespace=_CACHE_NAMESPACE,
                key=cache_key,
                value=_encode_fetch_cache(result),
                ttl_s=ttl_s,
            )

        fetched: CrawlResult | None = None
        crawl_exc: Exception | None = None
        pre_fetched_content = ctx.page.pre_fetched_content or ""
        if mode == "never":
            fetched = await get_cached()
            if fetched is None:
                await self.tracker.warning(
                    name="fetch.load.cache_miss",
                    request_id=ctx.request_id,
                    step="fetch.load",
                    data={"mode": mode, "cache_key": cache_key},
                )
                ctx.error.failed = True
                ctx.error.tag = "SOURCE_NOT_AVAILABLE"
                ctx.error.detail = "cache miss in crawl_mode=never"
                await self.tracker.error(
                    name="fetch.load.failed",
                    request_id=ctx.request_id,
                    step="fetch.load",
                    error_code="fetch_cache_miss",
                    error_message="cache miss in crawl_mode=never",
                    data={
                        "error_tag": ctx.error.tag,
                        "url": ctx.url,
                        "url_index": ctx.url_index,
                        "crawl_mode": ctx.page.crawl_mode,
                        "source": "cache",
                        "fatal": True,
                    },
                )
                return ctx
            await self.tracker.info(
                name="fetch.load.cache_hit",
                request_id=ctx.request_id,
                step="fetch.load",
                data={
                    "mode": mode,
                    "cache_key": cache_key,
                    "latency_ms": cache_fetch_ms,
                },
            )
        elif mode == "always":
            try:
                fetched = await crawl_once()
            except Exception as exc:  # noqa: BLE001
                crawl_exc = exc if isinstance(exc, Exception) else Exception(str(exc))
                await self.tracker.error(
                    name="fetch.load.crawl_failed",
                    request_id=ctx.request_id,
                    step="fetch.load",
                    error_type=type(crawl_exc).__name__,
                    data={
                        "mode": mode,
                        "cache_key": cache_key,
                    },
                )
            if fetched is None:
                if pre_fetched_content:
                    fetched = CrawlResult(
                        url=url,
                        status_code=200,
                        content_type="text/html",
                        content=pre_fetched_content.encode("utf-8"),
                        crawl_backend="pre_fetched",
                        rendered=False,
                        content_kind="html",
                        headers={},
                        attempt_chain=["pre_fetched"],
                    )
                    await self.tracker.info(
                        name="fetch.load.pre_fetched_fallback",
                        request_id=ctx.request_id,
                        step="fetch.load",
                        data={
                            "mode": mode,
                            "url": url,
                        },
                    )
                else:
                    message = str(crawl_exc or "crawl failed")
                    ctx.error.failed = True
                    ctx.error.tag = _resolve_error_tag(
                        code="fetch_crawl_failed",
                        message=message,
                    )
                    ctx.error.detail = message
                    await self.tracker.error(
                        name="fetch.load.failed",
                        request_id=ctx.request_id,
                        step="fetch.load",
                        error_code="fetch_crawl_failed",
                        error_message=message,
                        data={
                            "error_tag": ctx.error.tag,
                            "url": ctx.url,
                            "url_index": ctx.url_index,
                            "crawl_mode": ctx.page.crawl_mode,
                            "source": "crawl",
                            "fatal": True,
                        },
                    )
                    return ctx
            await write_cache(fetched)
            await self.tracker.info(
                name="fetch.load.crawl_succeeded",
                request_id=ctx.request_id,
                step="fetch.load",
                data={
                    "mode": mode,
                    "cache_key": cache_key,
                    "latency_ms": crawl_fetch_ms,
                },
            )
        elif mode == "preferred":
            try:
                fetched = await crawl_once()
            except Exception as exc:  # noqa: BLE001
                crawl_exc = exc if isinstance(exc, Exception) else Exception(str(exc))
                await self.tracker.warning(
                    name="fetch.load.crawl_failed",
                    request_id=ctx.request_id,
                    step="fetch.load",
                    warning_code="fetch_crawl_failed",
                    warning_message=str(crawl_exc),
                    data={
                        "mode": mode,
                        "cache_key": cache_key,
                    },
                )
            if fetched is None:
                cached = await get_cached()
                if cached is None:
                    if pre_fetched_content:
                        fetched = CrawlResult(
                            url=url,
                            status_code=200,
                            content_type="text/html",
                            content=pre_fetched_content.encode("utf-8"),
                            crawl_backend="pre_fetched",
                            rendered=False,
                            content_kind="html",
                            headers={},
                            attempt_chain=["pre_fetched"],
                        )
                        await self.tracker.info(
                            name="fetch.load.pre_fetched_fallback",
                            request_id=ctx.request_id,
                            step="fetch.load",
                            data={
                                "mode": mode,
                                "url": url,
                            },
                        )
                    else:
                        await self.tracker.error(
                            name="fetch.load.cache_miss",
                            request_id=ctx.request_id,
                            step="fetch.load",
                            data={"mode": mode, "cache_key": cache_key},
                        )
                        message = str(crawl_exc or "crawl failed and cache miss")
                        ctx.error.failed = True
                        ctx.error.tag = _resolve_error_tag(
                            code="fetch_crawl_failed",
                            message=message,
                        )
                        ctx.error.detail = message
                        await self.tracker.error(
                            name="fetch.load.failed",
                            request_id=ctx.request_id,
                            step="fetch.load",
                            error_code="fetch_crawl_failed",
                            error_message=message,
                            data={
                                "error_tag": ctx.error.tag,
                                "url": ctx.url,
                                "url_index": ctx.url_index,
                                "crawl_mode": ctx.page.crawl_mode,
                                "source": "crawl",
                                "fatal": True,
                            },
                        )
                        return ctx
                else:
                    fetched = cached
                await self.tracker.info(
                    name="fetch.load.cache_hit",
                    request_id=ctx.request_id,
                    step="fetch.load",
                    data={
                        "mode": mode,
                        "cache_key": cache_key,
                        "latency_ms": cache_fetch_ms,
                    },
                )
            else:
                await write_cache(fetched)
                await self.tracker.info(
                    name="fetch.load.crawl_succeeded",
                    request_id=ctx.request_id,
                    step="fetch.load",
                    data={
                        "mode": mode,
                        "cache_key": cache_key,
                        "latency_ms": crawl_fetch_ms,
                    },
                )
        else:
            cached = await get_cached()
            if cached is not None:
                fetched = cached
                await self.tracker.info(
                    name="fetch.load.cache_hit",
                    request_id=ctx.request_id,
                    step="fetch.load",
                    data={
                        "mode": mode,
                        "cache_key": cache_key,
                        "latency_ms": cache_fetch_ms,
                    },
                )
            else:
                await self.tracker.warning(
                    name="fetch.load.cache_miss",
                    request_id=ctx.request_id,
                    step="fetch.load",
                    data={
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
                    await self.tracker.error(
                        name="fetch.load.crawl_failed",
                        request_id=ctx.request_id,
                        step="fetch.load",
                        error_type=type(crawl_exc).__name__,
                        data={
                            "mode": mode,
                            "cache_key": cache_key,
                        },
                    )
                if fetched is None:
                    if pre_fetched_content:
                        fetched = CrawlResult(
                            url=url,
                            status_code=200,
                            content_type="text/html",
                            content=pre_fetched_content.encode("utf-8"),
                            crawl_backend="pre_fetched",
                            rendered=False,
                            content_kind="html",
                            headers={},
                            attempt_chain=["pre_fetched"],
                        )
                        await self.tracker.info(
                            name="fetch.load.pre_fetched_fallback",
                            request_id=ctx.request_id,
                            step="fetch.load",
                            data={
                                "mode": mode,
                                "url": url,
                            },
                        )
                    else:
                        message = str(crawl_exc or "crawl failed")
                        ctx.error.failed = True
                        ctx.error.tag = _resolve_error_tag(
                            code="fetch_crawl_failed",
                            message=message,
                        )
                        ctx.error.detail = message
                        await self.tracker.error(
                            name="fetch.load.failed",
                            request_id=ctx.request_id,
                            step="fetch.load",
                            error_code="fetch_crawl_failed",
                            error_message=message,
                            data={
                                "error_tag": ctx.error.tag,
                                "url": ctx.url,
                                "url_index": ctx.url_index,
                                "crawl_mode": ctx.page.crawl_mode,
                                "source": "crawl",
                                "fatal": True,
                            },
                        )
                        return ctx
                await write_cache(fetched)
                await self.tracker.info(
                    name="fetch.load.crawl_succeeded",
                    request_id=ctx.request_id,
                    step="fetch.load",
                    data={
                        "mode": mode,
                        "cache_key": cache_key,
                        "latency_ms": crawl_fetch_ms,
                    },
                )
        assert fetched is not None
        await self.meter.record(
            name="fetch.page",
            request_id=ctx.request_id,
            key=f"{ctx.request_id}:fetch.page:{ctx.url_index}",
            unit="page",
        )
        ctx.page.raw = fetched
        return ctx


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
        crawl_backend=str(obj.get("crawl_backend") or "curl_cffi"),
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
