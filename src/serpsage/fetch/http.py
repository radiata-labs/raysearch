from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlparse

import httpx

from serpsage.contracts.base import Component
from serpsage.contracts.protocols import Cache, Clock, FetchResult, Fetcher, Telemetry, stable_json
from serpsage.fetch.rate_limit import RateLimiter
from serpsage.settings.models import AppSettings


@dataclass(frozen=True)
class SimpleFetchResult:
    url: str
    status_code: int
    content_type: str | None
    content: bytes


class HttpFetcher(Component[None], Fetcher):
    def __init__(
        self,
        *,
        settings: AppSettings,
        telemetry: Telemetry,
        clock: Clock,
        http: httpx.AsyncClient,
        cache: Cache,
        rate_limiter: RateLimiter,
    ) -> None:
        super().__init__(settings=settings, telemetry=telemetry, clock=clock)
        self._http = http
        self._cache = cache
        self._rl = rate_limiter

    async def afetch(self, *, url: str) -> FetchResult:
        fetch_cfg = self.settings.enrich.fetch
        span = self.telemetry.start_span("fetch.http", url=url)
        host = urlparse(url).netloc.lower()
        try:
            cache_key = _hash_key({"url": url, "kind": "http"})
            cached = await self._cache.aget(namespace="fetch", key=cache_key)
            if cached:
                import json

                payload = json.loads(cached.decode("utf-8"))
                span.set_attr("cache_hit", True)
                return SimpleFetchResult(
                    url=url,
                    status_code=int(payload["status_code"]),
                    content_type=payload.get("content_type"),
                    content=bytes.fromhex(payload["content_hex"]),
                )

            await self._rl.acquire(host=host)
            try:
                data = await _fetch_with_retry(
                    http=self._http,
                    url=url,
                    settings=self.settings,
                )
            finally:
                await self._rl.release(host=host)

            await self._cache.aset(
                namespace="fetch",
                key=cache_key,
                value=data["cache_bytes"],
                ttl_s=int(self.settings.cache.fetch_ttl_s),
            )
            span.set_attr("cache_hit", False)
            return SimpleFetchResult(
                url=data["url"],
                status_code=int(data["status_code"]),
                content_type=data.get("content_type"),
                content=data["content"],
            )
        finally:
            span.end()


async def _fetch_with_retry(*, http: httpx.AsyncClient, url: str, settings: AppSettings) -> dict[str, Any]:
    fetch_cfg = settings.enrich.fetch
    retry = fetch_cfg.retry
    headers = {"User-Agent": fetch_cfg.user_agent}
    timeout = httpx.Timeout(fetch_cfg.timeout_s)

    last_exc: Exception | None = None
    for attempt in range(1, int(retry.max_attempts) + 1):
        try:
            async with http.stream(
                "GET",
                url,
                headers=headers,
                timeout=timeout,
                follow_redirects=bool(fetch_cfg.follow_redirects),
            ) as resp:
                resp.raise_for_status()
                content_type = resp.headers.get("content-type")
                data = await _read_with_limit(resp.aiter_bytes(), max_bytes=int(fetch_cfg.max_bytes))
                cache_bytes = _encode_fetch_cache(
                    status_code=int(resp.status_code),
                    content_type=content_type,
                    content=data,
                )
                return {
                    "url": str(resp.url),
                    "status_code": int(resp.status_code),
                    "content_type": content_type,
                    "content": data,
                    "cache_bytes": cache_bytes,
                }
        except (httpx.TimeoutException, httpx.NetworkError, httpx.HTTPStatusError) as exc:
            last_exc = exc
            # Retry on 429/5xx or transient network issues.
            status = getattr(getattr(exc, "response", None), "status_code", None)
            retryable = status is None or int(status) == 429 or (500 <= int(status) < 600)
            if not retryable or attempt >= int(retry.max_attempts):
                raise
            delay = _backoff_ms(attempt, retry.base_delay_ms, retry.max_delay_ms)
            await anyio_sleep_ms(delay)
            continue
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            raise
    assert last_exc is not None
    raise last_exc


async def _read_with_limit(aiter, *, max_bytes: int) -> bytes:
    chunks: list[bytes] = []
    total = 0
    async for part in aiter:
        if not part:
            continue
        total += len(part)
        if total > max_bytes:
            raise ValueError(f"exceeded max_bytes={max_bytes}")
        chunks.append(part)
    return b"".join(chunks)


def _encode_fetch_cache(*, status_code: int, content_type: str | None, content: bytes) -> bytes:
    import json

    payload = {
        "status_code": int(status_code),
        "content_type": content_type,
        "content_hex": content.hex(),
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")


def _hash_key(obj: Any) -> str:
    import hashlib

    return hashlib.sha256(stable_json(obj).encode("utf-8")).hexdigest()


def _backoff_ms(attempt: int, base_ms: int, max_ms: int) -> int:
    base = max(1, int(base_ms))
    cap = max(base, int(max_ms))
    # Decorrelated jitter.
    exp = min(cap, int(base * (2 ** max(0, attempt - 1))))
    return int(min(cap, random.randint(base, exp)))


async def anyio_sleep_ms(ms: int) -> None:
    import anyio

    await anyio.sleep(max(0.0, float(ms) / 1000.0))


__all__ = ["HttpFetcher", "SimpleFetchResult", "_read_with_limit"]
