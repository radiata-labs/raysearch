from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing_extensions import override

import anyio
import httpx

from serpsage.components.fetch.utils import (
    browser_headers,
    classify_content_kind,
    estimate_text_quality,
    get_delay_s,
    parse_retry_after_s,
)
from serpsage.contracts.services import FetcherBase
from serpsage.models.fetch import FetchAttempt, FetchResult

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.domain.http import HttpClient
    from serpsage.settings.models import RetrySettings


class HttpxFetcher(FetcherBase):
    def __init__(self, *, rt, http: HttpClient) -> None:  # noqa: ANN001
        super().__init__(rt=rt)
        self.bind_deps(http)
        self._http = http.client

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
        _ = allow_render, depth, rank_index
        with self.span("fetch.httpx", url=url) as sp:
            res = await self.fetch_attempt(
                url=url, profile="browser", span=sp, timeout_s=timeout_s
            )
            if int(res.status_code) <= 0:
                raise RuntimeError("httpx fetch failed")
            return FetchResult(
                url=str(res.url or url),
                status_code=int(res.status_code),
                content_type=res.content_type,
                content=bytes(res.content or b""),
                fetch_mode="httpx",
                rendered=False,
                content_kind=res.content_kind,
                headers=dict(res.headers or {}),
            )

    async def fetch_attempt(
        self,
        *,
        url: str,
        profile: str,
        span: SpanBase,
        retry: RetrySettings | None = None,
        timeout_s: float | None = None,
    ) -> FetchAttempt:
        fetch_cfg = self.settings.enrich.fetch
        retry = retry or fetch_cfg.httpx.retry

        started = time.time()
        max_attempts = max(1, int(getattr(retry, "max_attempts", 3)))

        req_timeout_s = (
            max(0.5, float(timeout_s))
            if timeout_s is not None
            else max(0.5, fetch_cfg.timeout_s)
        )
        timeout = httpx.Timeout(req_timeout_s)

        last_status: int | None = None
        last_url = url
        last_ct: str | None = None
        last_body: bytes = b""
        last_truncated = False
        last_enc: str | None = None
        last_len_hdr: str | None = None
        last_headers: dict[str, str] = {}

        for attempt in range(1, max_attempts + 1):
            span.set_attr("httpx_profile", profile)
            span.set_attr("httpx_attempt", int(attempt))
            try:
                async with self._http.stream(
                    "GET",
                    url,
                    headers=browser_headers(fetch_cfg, profile=profile),
                    cookies=fetch_cfg.cookies or None,
                    timeout=timeout,
                    follow_redirects=bool(fetch_cfg.follow_redirects),
                ) as resp:
                    last_status = int(resp.status_code)
                    last_url = str(resp.url)
                    last_ct = resp.headers.get("content-type")
                    last_enc = resp.headers.get("content-encoding")
                    last_len_hdr = resp.headers.get("content-length")
                    last_headers = {str(k): str(v) for k, v in resp.headers.items()}

                    last_body = b"".join(
                        [part async for part in resp.aiter_bytes() if part]
                    )

                    if last_status == 429 or (500 <= last_status < 600):
                        if attempt >= max_attempts:
                            break
                        ra = parse_retry_after_s(resp.headers.get("retry-after"))
                        delay = ra if ra is not None else get_delay_s(retry.delay_ms)
                        span.set_attr("httpx_retry_reason", "status")
                        span.set_attr("httpx_retry_delay_s", float(delay))
                        await anyio.sleep(delay)
                        continue

                    break
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                span.set_attr("httpx_error_type", type(exc).__name__)
                if attempt >= max_attempts:
                    break
                delay = get_delay_s(retry.delay_ms)
                span.set_attr("httpx_retry_reason", "network")
                span.set_attr("httpx_retry_delay_s", float(delay))
                await anyio.sleep(delay)
                continue
            except Exception as exc:  # noqa: BLE001
                span.set_attr("httpx_error_type", type(exc).__name__)
                break

        elapsed_ms = int((time.time() - started) * 1000)
        span.set_attr("httpx_elapsed_ms", int(elapsed_ms))
        span.set_attr("httpx_status", int(last_status or 0))
        span.set_attr("httpx_bytes", int(len(last_body)))
        span.set_attr("httpx_truncated", bool(last_truncated))

        content_kind = classify_content_kind(
            content_type=last_ct, url=last_url, content=last_body
        )
        text_chars, content_score, _ = estimate_text_quality(
            last_body, content_kind=content_kind
        )
        span.set_attr("content_kind", content_kind)
        span.set_attr("text_chars", int(text_chars))
        span.set_attr("content_score", float(content_score))

        return FetchAttempt(
            url=last_url,
            status_code=int(last_status or 0),
            content_type=last_ct,
            content=last_body,
            strategy_used="httpx",
            fetch_mode="httpx",
            rendered=False,
            content_kind=content_kind,
            headers=last_headers,
            content_encoding=last_enc,
            content_length_header=last_len_hdr,
            content_score=float(content_score),
            text_chars=int(text_chars),
            blocked=False,
        )


__all__ = ["HttpxFetcher"]
