from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing_extensions import override

import anyio
import httpx

from serpsage.components.fetch.utils import (
    blocked_marker_hit,
    browser_headers,
    classify_content_kind,
    estimate_text_quality,
    get_delay_s,
    parse_retry_after_s,
)
from serpsage.contracts.services import FetcherBase
from serpsage.core.tuning import DEFAULT_FOLLOW_REDIRECTS, fetch_profile_for_depth
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
        _ = allow_render, rank_index
        with self.span("fetch.httpx", url=url) as sp:
            res = await self.fetch_attempt(
                url=url,
                profile="browser",
                span=sp,
                timeout_s=timeout_s,
                depth=depth,
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
                attempt_chain=list(res.attempt_chain or []),
                quality_score=float(res.quality_score or res.content_score),
            )

    async def fetch_attempt(
        self,
        *,
        url: str,
        profile: str,
        span: SpanBase,
        retry: RetrySettings | None = None,
        timeout_s: float | None = None,
        depth: str | None = None,
    ) -> FetchAttempt:
        tune = fetch_profile_for_depth(depth)
        started = time.time()

        req_timeout_s = (
            max(0.35, float(timeout_s))
            if timeout_s is not None
            else max(0.35, float(tune.http_timeout_s))
        )
        timeout = httpx.Timeout(req_timeout_s)
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
                    headers=browser_headers(profile=profile),
                    timeout=timeout,
                    follow_redirects=bool(DEFAULT_FOLLOW_REDIRECTS),
                ) as resp:
                    last_status = int(resp.status_code)
                    last_url = str(resp.url)
                    last_ct = resp.headers.get("content-type")
                    last_enc = resp.headers.get("content-encoding")
                    last_len_hdr = resp.headers.get("content-length")
                    last_headers = {str(k): str(v) for k, v in resp.headers.items()}
                    kind_hint = classify_content_kind(
                        content_type=last_ct,
                        url=last_url,
                        content=b"",
                    )
                    last_body, last_truncated = await self._read_stream_body(
                        resp=resp,
                        kind_hint=kind_hint,
                        depth=depth,
                    )

                    if last_status == 429 or (500 <= last_status < 600):
                        if attempt >= max_attempts:
                            break
                        ra = parse_retry_after_s(resp.headers.get("retry-after"))
                        delay = ra if ra is not None else get_delay_s(delay_ms)
                        span.set_attr("httpx_retry_reason", "status")
                        span.set_attr("httpx_retry_delay_s", float(delay))
                        await anyio.sleep(delay)
                        continue
                    break
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                span.set_attr("httpx_error_type", type(exc).__name__)
                if attempt >= max_attempts:
                    break
                delay = get_delay_s(delay_ms)
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
            strategy_used="httpx",
            fetch_mode="httpx",
            rendered=False,
            content_kind=content_kind,
            headers=last_headers,
            content_encoding=last_enc,
            content_length_header=last_len_hdr,
            content_score=float(content_score),
            text_chars=int(text_chars),
            blocked=blocked,
            attempt_chain=[f"httpx:{profile}"],
            quality_score=float(quality_score),
        )

    async def _read_stream_body(
        self,
        *,
        resp: httpx.Response,
        kind_hint: str,
        depth: str | None,
    ) -> tuple[bytes, bool]:
        tune = fetch_profile_for_depth(depth)
        if kind_hint == "pdf":
            budget = int(tune.pdf_byte_budget)
        elif kind_hint == "text":
            budget = int(tune.text_byte_budget)
        elif kind_hint in {"binary", "unknown"}:
            budget = int(tune.binary_byte_budget)
        else:
            budget = int(tune.html_byte_budget)

        total = 0
        truncated = False
        parts: list[bytes] = []
        check_every = 3
        chunk_count = 0
        async for part in resp.aiter_bytes():
            if not part:
                continue
            parts.append(part)
            total += len(part)
            chunk_count += 1
            if total >= budget:
                truncated = True
                break
            if (
                kind_hint == "html"
                and total >= 260_000
                and chunk_count % check_every == 0
            ):
                body = b"".join(parts)
                text_chars, content_score, _ = estimate_text_quality(
                    body,
                    content_kind="html",
                )
                if text_chars >= int(
                    tune.html_early_accept_chars
                ) and content_score >= float(tune.html_early_accept_score):
                    break
        return b"".join(parts), truncated


__all__ = ["HttpxFetcher"]
