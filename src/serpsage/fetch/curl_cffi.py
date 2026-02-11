from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

from serpsage.contracts.services import FetcherBase
from serpsage.fetch.common import browser_headers, truncate_bytes
from serpsage.models.fetch import FetchAttempt, FetchResult

CurlSessionFactory: Any = None
try:
    from curl_cffi.requests import AsyncSession as _CurlAsyncSession

    CurlSessionFactory = _CurlAsyncSession
    CURL_CFFI_AVAILABLE = True
except Exception:  # noqa: BLE001
    CURL_CFFI_AVAILABLE = False

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


class CurlCffiFetcher(FetcherBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)
        if not CURL_CFFI_AVAILABLE:
            raise RuntimeError("curl_cffi is not available; install curl_cffi")
        self._session: Any | None = None

    @override
    async def on_init(self) -> None:
        if self._session is None:
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
            fn = getattr(self._session, "aclose", None)
            if fn is not None:
                await fn()
            else:
                fn = getattr(self._session, "close", None)
                if fn is not None:
                    out = fn()
                    if hasattr(out, "__await__"):
                        await out
        except Exception:
            return
        finally:
            self._session = None

    async def fetch_attempt(self, *, url: str, span: Any) -> FetchAttempt:
        if self._session is None:
            await self.ainit()
        session = self._session
        if session is None:
            raise RuntimeError("curl_cffi session is not initialized")

        fetch_cfg = self.settings.enrich.fetch
        started = time.time()

        headers = browser_headers(fetch_cfg)
        cookies = dict(getattr(fetch_cfg, "cookies", None) or {})
        proxy = getattr(fetch_cfg, "proxy", None)

        timeout_s = float(getattr(fetch_cfg, "timeout_s", 10.0))
        budget = float(getattr(fetch_cfg, "total_budget_s", 3.0))
        timeout_s = max(0.5, min(timeout_s, max(0.5, budget * 0.6)))

        try:
            resp = await session.get(
                url,
                headers=headers,
                cookies=cookies or None,
                timeout=timeout_s,
                allow_redirects=bool(fetch_cfg.follow_redirects),
                proxy=proxy,
                impersonate=cast(
                    "Any", str(getattr(fetch_cfg, "curl_impersonate", "chrome120"))
                ),
                http_version="v2"
                if bool(getattr(fetch_cfg, "curl_http2", True))
                else "v1",
                verify=bool(getattr(fetch_cfg, "curl_verify_ssl", True)),
            )
            status_code = int(getattr(resp, "status_code", 0) or 0)
            final_url = str(getattr(resp, "url", url))
            hdrs = getattr(resp, "headers", None)

            def _hget(name: str) -> str | None:
                if hdrs is None:
                    return None
                try:
                    return hdrs.get(name)  # type: ignore[no-any-return]
                except Exception:
                    return None

            content_type = _hget("content-type")
            content_encoding = _hget("content-encoding")
            content_length_header = _hget("content-length")

            data = bytes(getattr(resp, "content", b"") or b"")
            max_bytes = int(fetch_cfg.max_bytes)
            behavior = str(
                getattr(fetch_cfg, "max_bytes_behavior", "truncate") or "truncate"
            )
            if behavior == "error" and max_bytes > 0 and len(data) > max_bytes:
                raise ValueError(f"exceeded max_bytes={max_bytes}")
            data, truncated = truncate_bytes(data, max_bytes=max_bytes)

            elapsed_ms = int((time.time() - started) * 1000)
            span.set_attr("curl_status", int(status_code))
            span.set_attr("curl_elapsed_ms", int(elapsed_ms))
            span.set_attr("curl_bytes", int(len(data)))
            span.set_attr("curl_truncated", bool(truncated))

            return FetchAttempt(
                url=final_url,
                status_code=status_code,
                content_type=content_type,
                content=data,
                truncated=bool(truncated),
                strategy_used="curl_cffi",
                content_encoding=content_encoding,
                content_length_header=content_length_header,
            )
        except Exception as exc:  # noqa: BLE001
            span.set_attr("curl_error_type", type(exc).__name__)
            return FetchAttempt(
                url=url,
                status_code=0,
                content_type=None,
                content=b"",
                truncated=False,
                strategy_used="curl_cffi",
            )


__all__ = ["CURL_CFFI_AVAILABLE", "CurlCffiFetcher"]
