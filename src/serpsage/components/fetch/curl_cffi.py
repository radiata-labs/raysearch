from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

import anyio

from serpsage.components.fetch.utils import (
    browser_headers,
    get_delay_s,
    parse_retry_after_s,
)
from serpsage.contracts.services import FetcherBase
from serpsage.models.fetch import FetchAttempt, FetchResult

CurlSessionFactory: type[AsyncSession] | None = None
try:
    from curl_cffi.requests import AsyncSession as _CurlAsyncSession

    CurlSessionFactory = _CurlAsyncSession
    CURL_CFFI_AVAILABLE = True
except Exception:  # noqa: BLE001
    CURL_CFFI_AVAILABLE = False

if TYPE_CHECKING:
    from curl_cffi.requests import AsyncSession

    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import RetrySettings


class CurlCffiFetcher(FetcherBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)
        if not CURL_CFFI_AVAILABLE:
            raise RuntimeError("curl_cffi is not available; install curl_cffi")
        self._session: AsyncSession | None = None

    @override
    async def on_init(self) -> None:
        if self._session is None and CurlSessionFactory is not None:
            self._session = CurlSessionFactory()

    @override
    async def afetch(self, *, url: str) -> FetchResult:
        with self.span("fetch.curl", url=url) as sp:
            res = await self.fetch_attempt(url=url, span=sp)
            if int(res.status_code) <= 0:
                raise RuntimeError("curl_cffi fetch failed")
            return FetchResult(
                url=str(res.url or url),
                status_code=int(res.status_code),
                content_type=res.content_type,
                content=bytes(res.content or b""),
            )

    @override
    async def on_close(self) -> None:
        if self._session is None:
            return
        try:
            await self._session.close()
        except Exception:
            return
        finally:
            self._session = None

    async def fetch_attempt(
        self, *, url: str, span: SpanBase, retry: RetrySettings | None = None
    ) -> FetchAttempt:
        if self._session is None:
            await self.ainit()
        if self._session is None:
            raise RuntimeError("curl_cffi session is not initialized")

        fetch_cfg = self.settings.enrich.fetch
        curl_cfg = fetch_cfg.curl_cffi
        started = time.time()

        headers = browser_headers(fetch_cfg)
        proxy = self.settings.http.proxy

        timeout_s = max(0.5, fetch_cfg.timeout_s)

        retry = retry or fetch_cfg.curl_cffi.retry
        max_attempts = max(1, int(getattr(retry, "max_attempts", 3)))

        last_status: int | None = None
        last_url = url
        last_ct: str | None = None
        last_body: bytes = b""
        last_enc: str | None = None
        last_len_hdr: str | None = None

        for attempt in range(1, max_attempts + 1):
            span.set_attr("curl_attempt", int(attempt))
            try:
                resp = await self._session.get(
                    url,
                    headers=headers | fetch_cfg.extra_headers,
                    cookies=fetch_cfg.cookies or None,
                    timeout=timeout_s,
                    allow_redirects=bool(fetch_cfg.follow_redirects),
                    proxy=proxy,
                    impersonate=cast("Any", str(curl_cfg.impersonate)),
                    http_version="v2" if bool(curl_cfg.http2) else "v1",
                    verify=bool(curl_cfg.verify_ssl),
                )
                last_status = int(getattr(resp, "status_code", 0) or 0)
                last_url = str(getattr(resp, "url", url))
                hdrs = cast("dict[str, str]", getattr(resp, "headers", None))

                def _hget(hdrs: dict[str, str] | None, name: str) -> str | None:
                    if hdrs is None:
                        return None
                    try:
                        return hdrs.get(name)  # type: ignore[no-any-return]
                    except Exception:
                        return None

                last_ct = _hget(hdrs, "content-type")
                last_enc = _hget(hdrs, "content-encoding")
                last_len_hdr = _hget(hdrs, "content-length")
                last_body = bytes(getattr(resp, "content", b"") or b"")

                if last_status == 429 or (500 <= last_status < 600):
                    if attempt >= max_attempts:
                        break
                    ra = parse_retry_after_s(_hget(hdrs, "retry-after"))
                    delay = ra if ra is not None else get_delay_s(retry.delay_ms)
                    span.set_attr("curl_retry_reason", "status")
                    span.set_attr("curl_retry_delay_s", float(delay))
                    await anyio.sleep(delay)
                    continue

                break
            except Exception as exc:  # noqa: BLE001
                span.set_attr("curl_error_type", type(exc).__name__)
                if attempt >= max_attempts:
                    break
                delay = get_delay_s(retry.delay_ms)
                span.set_attr("curl_retry_reason", "network")
                span.set_attr("curl_retry_delay_s", float(delay))
                await anyio.sleep(delay)
                continue

        elapsed_ms = int((time.time() - started) * 1000)
        span.set_attr("curl_status", int(last_status or 0))
        span.set_attr("curl_elapsed_ms", int(elapsed_ms))
        span.set_attr("curl_bytes", int(len(last_body)))

        return FetchAttempt(
            url=last_url,
            status_code=int(last_status or 0),
            content_type=last_ct,
            content=last_body,
            strategy_used="curl_cffi",
            content_encoding=last_enc,
            content_length_header=last_len_hdr,
        )


__all__ = ["CURL_CFFI_AVAILABLE", "CurlCffiFetcher"]
