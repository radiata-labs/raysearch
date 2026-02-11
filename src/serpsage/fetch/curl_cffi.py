from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any
from typing_extensions import override

try:
    from curl_cffi.requests import AsyncSession

    CURL_CFFI_AVAILABLE = True
except Exception:  # noqa: BLE001
    AsyncSession = Any  # type: ignore[assignment]
    CURL_CFFI_AVAILABLE = False


from serpsage.contracts.base import WorkUnit
from serpsage.contracts.protocols import Fetcher, FetchResult

if TYPE_CHECKING:
    from serpsage.app.runtime import CoreRuntime
    from serpsage.fetch.auto import AttemptResult


@dataclass
class CurlFetchOutcome:
    final_url: str
    status_code: int
    content_type: str | None
    content: bytes
    truncated: bool
    content_encoding: str | None
    content_length_header: str | None


@dataclass(frozen=True)
class SimpleFetchResult:
    url: str
    status_code: int
    content_type: str | None
    content: bytes


def _browser_headers(fetch_cfg: Any) -> dict[str, str]:
    ua = str(fetch_cfg.user_agent)
    lang = str(getattr(fetch_cfg, "accept_language", "") or "en")
    enc = (
        "gzip, deflate"
        if bool(getattr(fetch_cfg, "disable_br", True))
        else "gzip, deflate, br"
    )

    headers = {
        "User-Agent": ua,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": lang,
        "Accept-Encoding": enc,
        "Upgrade-Insecure-Requests": "1",
        "DNT": "1",
    }
    extra = getattr(fetch_cfg, "extra_headers", None) or {}
    for k, v in extra.items():
        if not k:
            continue
        headers[str(k)] = str(v)
    return headers


def _truncate_bytes(data: bytes, *, max_bytes: int) -> tuple[bytes, bool]:
    if max_bytes <= 0:
        return data, False
    if len(data) <= max_bytes:
        return data, False
    return data[:max_bytes], True


class CurlCffiFetcher(WorkUnit, Fetcher):
    """curl_cffi-based fetcher for stronger TLS/browser impersonation."""

    def __init__(self, *, rt: CoreRuntime) -> None:
        super().__init__(rt=rt)
        if not CURL_CFFI_AVAILABLE:
            raise RuntimeError("curl_cffi is not available; install curl_cffi")
        self._session: AsyncSession = AsyncSession()  # type: ignore

    @override
    async def afetch(self, *, url: str) -> FetchResult:
        with self.span("fetch.curl", url=url) as sp:
            res = await self.fetch_attempt(url=url, span=sp)
            if int(res.status_code) <= 0:
                raise RuntimeError("curl_cffi fetch failed")
            return SimpleFetchResult(
                url=str(res.final_url or url),
                status_code=int(res.status_code),
                content_type=res.content_type,
                content=bytes(res.content or b""),
            )

    @override
    async def aclose(self) -> None:
        # curl_cffi session close API differs across versions.
        try:
            fn = getattr(self._session, "aclose", None)
            if fn is not None:
                await fn()
                return
            fn = getattr(self._session, "close", None)
            if fn is not None:
                out = fn()
                if hasattr(out, "__await__"):
                    await out
        except Exception:
            return

    async def fetch_attempt(self, *, url: str, span: Any) -> AttemptResult:  # noqa: ANN401
        fetch_cfg = self.settings.enrich.fetch
        started = time.time()

        headers = _browser_headers(fetch_cfg)
        cookies = dict(getattr(fetch_cfg, "cookies", None) or {})
        proxy = getattr(fetch_cfg, "proxy", None)

        timeout_s = float(getattr(fetch_cfg, "timeout_s", 10.0))
        budget = float(getattr(fetch_cfg, "total_budget_s", 3.0))
        # Keep curl attempt short under auto strategy.
        timeout_s = max(0.5, min(timeout_s, max(0.5, budget * 0.6)))

        try:
            resp = await self._session.get(
                url,
                headers=headers,
                cookies=cookies or None,
                timeout=timeout_s,
                allow_redirects=bool(fetch_cfg.follow_redirects),
                proxy=proxy,
                impersonate=str(getattr(fetch_cfg, "curl_impersonate", "chrome120")),
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
            data, truncated = _truncate_bytes(data, max_bytes=max_bytes)

            elapsed_ms = int((time.time() - started) * 1000)
            span.set_attr("curl_status", int(status_code))
            span.set_attr("curl_elapsed_ms", int(elapsed_ms))
            span.set_attr("curl_bytes", int(len(data)))
            span.set_attr("curl_truncated", bool(truncated))

            from serpsage.fetch.auto import AttemptResult  # noqa: PLC0415

            return AttemptResult(
                final_url=final_url,
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
            # Return an "empty" attempt result; auto strategy will treat it as unusable.
            from serpsage.fetch.auto import AttemptResult  # noqa: PLC0415

            return AttemptResult(
                final_url=url,
                status_code=0,
                content_type=None,
                content=b"",
                truncated=False,
                strategy_used="curl_cffi",
            )


__all__ = ["CURL_CFFI_AVAILABLE", "CurlCffiFetcher", "SimpleFetchResult"]
