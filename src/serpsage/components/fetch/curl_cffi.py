from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

import anyio

from serpsage.components.fetch.utils import (
    blocked_marker_hit,
    browser_headers,
    classify_content_kind,
    estimate_text_quality,
    get_delay_s,
    parse_retry_after_s,
)
from serpsage.contracts.services import FetcherBase
from serpsage.core.tuning import (
    DEFAULT_FOLLOW_REDIRECTS,
    fetch_profile_for_depth,
)
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
    async def afetch(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
        allow_render: bool = True,
        depth: str | None = None,
        rank_index: int = 0,
    ) -> FetchResult:
        _ = allow_render, rank_index
        with self.span("fetch.curl", url=url) as sp:
            res = await self.fetch_attempt(
                url=url,
                span=sp,
                timeout_s=timeout_s,
                depth=depth,
            )
            if int(res.status_code) <= 0:
                raise RuntimeError("curl_cffi fetch failed")
            return FetchResult(
                url=str(res.url or url),
                status_code=int(res.status_code),
                content_type=res.content_type,
                content=bytes(res.content or b""),
                fetch_mode="curl_cffi",
                rendered=False,
                content_kind=res.content_kind,
                headers=dict(res.headers or {}),
                attempt_chain=list(res.attempt_chain or []),
                quality_score=float(res.quality_score or res.content_score),
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
        self,
        *,
        url: str,
        span: SpanBase,
        retry: RetrySettings | None = None,
        timeout_s: float | None = None,
        depth: str | None = None,
    ) -> FetchAttempt:
        if self._session is None:
            await self.ainit()
        if self._session is None:
            raise RuntimeError("curl_cffi session is not initialized")

        tune = fetch_profile_for_depth(depth)
        started = time.time()
        proxy = self.settings.http.proxy
        req_timeout_s = (
            max(0.35, float(timeout_s))
            if timeout_s is not None
            else max(0.35, tune.http_timeout_s)
        )
        max_attempts = max(
            1,
            int(
                getattr(retry, "max_attempts", 0) or int(tune.retry_attempts),
            ),
        )
        delay_ms = int(getattr(retry, "delay_ms", 0) or int(tune.retry_delay_ms))

        last_status: int | None = None
        last_url = url
        last_ct: str | None = None
        last_body: bytes = b""
        last_enc: str | None = None
        last_len_hdr: str | None = None
        last_headers: dict[str, str] = {}
        last_truncated = False

        for attempt in range(1, max_attempts + 1):
            span.set_attr("curl_attempt", int(attempt))
            try:
                resp = await self._session.get(
                    url,
                    headers=browser_headers(profile="browser"),
                    timeout=req_timeout_s,
                    allow_redirects=bool(DEFAULT_FOLLOW_REDIRECTS),
                    proxy=proxy,
                    impersonate=cast("Any", "chrome124"),
                    http_version="v2",
                    verify=True,
                )
                last_status = int(getattr(resp, "status_code", 0) or 0)
                last_url = str(getattr(resp, "url", url))
                hdrs = cast("dict[str, str] | None", getattr(resp, "headers", None))
                body = bytes(getattr(resp, "content", b"") or b"")
                if not isinstance(hdrs, dict):
                    hdrs = {}
                last_headers = {str(k): str(v) for k, v in hdrs.items()}
                last_ct = last_headers.get("content-type")
                last_enc = last_headers.get("content-encoding")
                last_len_hdr = last_headers.get("content-length")
                last_body, last_truncated = self._truncate_by_kind(
                    body=body,
                    content_type=last_ct,
                    url=last_url,
                    depth=depth,
                )

                if last_status == 429 or (500 <= last_status < 600):
                    if attempt >= max_attempts:
                        break
                    ra = parse_retry_after_s(last_headers.get("retry-after"))
                    delay = ra if ra is not None else get_delay_s(delay_ms)
                    span.set_attr("curl_retry_reason", "status")
                    span.set_attr("curl_retry_delay_s", float(delay))
                    await anyio.sleep(delay)
                    continue
                break
            except Exception as exc:  # noqa: BLE001
                span.set_attr("curl_error_type", type(exc).__name__)
                if attempt >= max_attempts:
                    break
                delay = get_delay_s(delay_ms)
                span.set_attr("curl_retry_reason", "network")
                span.set_attr("curl_retry_delay_s", float(delay))
                await anyio.sleep(delay)
                continue

        elapsed_ms = int((time.time() - started) * 1000)
        span.set_attr("curl_status", int(last_status or 0))
        span.set_attr("curl_elapsed_ms", int(elapsed_ms))
        span.set_attr("curl_bytes", int(len(last_body)))
        span.set_attr("curl_truncated", bool(last_truncated))
        content_kind = classify_content_kind(
            content_type=last_ct,
            url=last_url,
            content=last_body,
        )
        text_chars, content_score, _ = estimate_text_quality(
            last_body,
            content_kind=content_kind,
        )
        blocked = bool(blocked_marker_hit(last_body))
        quality_score = float(content_score - (0.3 if blocked else 0.0))
        span.set_attr("content_kind", content_kind)
        span.set_attr("text_chars", int(text_chars))
        span.set_attr("content_score", float(content_score))

        return FetchAttempt(
            url=last_url,
            status_code=int(last_status or 0),
            content_type=last_ct,
            content=last_body,
            strategy_used="curl_cffi",
            fetch_mode="curl_cffi",
            rendered=False,
            content_kind=content_kind,
            headers=last_headers,
            content_encoding=last_enc,
            content_length_header=last_len_hdr,
            content_score=float(content_score),
            text_chars=int(text_chars),
            blocked=blocked,
            attempt_chain=["curl_cffi"],
            quality_score=float(quality_score),
        )

    def _truncate_by_kind(
        self,
        *,
        body: bytes,
        content_type: str | None,
        url: str,
        depth: str | None,
    ) -> tuple[bytes, bool]:
        tune = fetch_profile_for_depth(depth)
        kind = classify_content_kind(
            content_type=content_type, url=url, content=body[:128]
        )
        if kind == "pdf":
            budget = int(tune.pdf_byte_budget)
        elif kind == "text":
            budget = int(tune.text_byte_budget)
        elif kind in {"unknown", "binary"}:
            budget = int(tune.binary_byte_budget)
        else:
            budget = int(tune.html_byte_budget)
        if len(body) <= budget:
            return body, False
        return body[:budget], True


__all__ = ["CURL_CFFI_AVAILABLE", "CurlCffiFetcher"]
