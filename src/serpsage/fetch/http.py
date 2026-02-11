from __future__ import annotations

import random
import time
from typing import Any
from typing_extensions import override

import anyio
import httpx

from serpsage.contracts.services import FetcherBase
from serpsage.fetch.common import browser_headers, looks_like_html, parse_content_type
from serpsage.models.fetch import FetchAttempt, FetchResult


def _backoff_s(attempt: int, base_ms: int, max_ms: int) -> float:
    base = max(1, int(base_ms))
    cap = max(base, int(max_ms))
    exp = min(cap, int(base * (2 ** max(0, attempt - 1))))
    return float(min(cap, random.randint(base, exp))) / 1000.0


def _parse_retry_after_s(v: str | None) -> float | None:
    if not v:
        return None
    v = v.strip()
    if not v:
        return None
    if v.isdigit():
        return float(int(v))
    return None


async def _read_with_limit(
    aiter, *, max_bytes: int, behavior: str
) -> tuple[bytes, bool]:
    if max_bytes <= 0:
        chunks: list[bytes] = [part async for part in aiter if part]
        return b"".join(chunks), False

    buf = bytearray()
    truncated = False
    async for part in aiter:
        if not part:
            continue
        remain = max_bytes - len(buf)
        if remain <= 0:
            truncated = True
            break
        if len(part) <= remain:
            buf.extend(part)
        else:
            if behavior == "error":
                raise ValueError(f"exceeded max_bytes={max_bytes}")
            buf.extend(part[:remain])
            truncated = True
            break
    return bytes(buf), bool(truncated)


class HttpxFetcher(FetcherBase):
    def __init__(self, *, rt, http: httpx.AsyncClient) -> None:  # noqa: ANN001
        super().__init__(rt=rt)
        self._http = http

    @override
    async def afetch(self, *, url: str) -> FetchResult:
        with self.span("fetch.httpx", url=url) as sp:
            res = await self.fetch_attempt(url=url, profile="browser", span=sp)
            if int(res.status_code) <= 0:
                raise RuntimeError("httpx fetch failed")
            return FetchResult(
                url=str(res.url or url),
                status_code=int(res.status_code),
                content_type=res.content_type,
                content=bytes(res.content or b""),
            )

    async def fetch_attempt(
        self, *, url: str, profile: str, span: Any
    ) -> FetchAttempt:
        fetch_cfg = self.settings.enrich.fetch
        retry = fetch_cfg.retry

        started = time.time()
        max_attempts = min(
            int(getattr(fetch_cfg, "max_attempts_per_strategy", 3)),
            int(getattr(retry, "max_attempts", 3)),
        )
        max_attempts = max(1, max_attempts)

        budget = float(getattr(fetch_cfg, "total_budget_s", 3.0))
        timeout_s = float(getattr(fetch_cfg, "timeout_s", 10.0))
        timeout_s = max(0.5, min(timeout_s, max(0.5, budget * 0.7)))
        timeout = httpx.Timeout(timeout_s)

        headers = browser_headers(fetch_cfg, profile=profile)
        cookies = dict(getattr(fetch_cfg, "cookies", None) or {})

        allow = {
            str(x).strip().lower()
            for x in (fetch_cfg.allow_content_types or [])
            if str(x).strip()
        }
        sniff_n = max(1024, int(getattr(fetch_cfg, "sniff_html_bytes", 16384)))
        behavior = str(
            getattr(fetch_cfg, "max_bytes_behavior", "truncate") or "truncate"
        )
        max_bytes = int(fetch_cfg.max_bytes)

        last_status: int | None = None
        last_url = url
        last_ct: str | None = None
        last_body: bytes = b""
        last_truncated = False
        last_enc: str | None = None
        last_len_hdr: str | None = None

        for attempt in range(1, max_attempts + 1):
            span.set_attr("httpx_profile", profile)
            span.set_attr("httpx_attempt", int(attempt))
            try:
                async with self._http.stream(
                    "GET",
                    url,
                    headers=headers,
                    cookies=cookies or None,
                    timeout=timeout,
                    follow_redirects=bool(fetch_cfg.follow_redirects),
                ) as resp:
                    last_status = int(resp.status_code)
                    last_url = str(resp.url)
                    last_ct = resp.headers.get("content-type")
                    last_enc = resp.headers.get("content-encoding")
                    last_len_hdr = resp.headers.get("content-length")

                    ct_main = parse_content_type(last_ct)
                    sniff_needed = not (ct_main and (not allow or ct_main in allow))

                    body, truncated = await _read_with_limit(
                        resp.aiter_bytes(),
                        max_bytes=max_bytes,
                        behavior=behavior,
                    )
                    last_body = body
                    last_truncated = bool(truncated)

                    if last_status == 429 or (500 <= last_status < 600):
                        if attempt >= max_attempts:
                            break
                        ra = _parse_retry_after_s(resp.headers.get("retry-after"))
                        delay = (
                            ra
                            if ra is not None
                            else _backoff_s(
                                attempt, retry.base_delay_ms, retry.max_delay_ms
                            )
                        )
                        delay = min(delay, 1.5)
                        span.set_attr("httpx_retry_reason", "status")
                        span.set_attr("httpx_retry_delay_s", float(delay))
                        await anyio.sleep(delay)
                        continue

                    if allow and ct_main and ct_main not in allow or sniff_needed and allow and ct_main != "text/plain":
                        if not looks_like_html(body[:sniff_n]):
                            break

                    break
            except (httpx.TimeoutException, httpx.NetworkError) as exc:
                span.set_attr("httpx_error_type", type(exc).__name__)
                if attempt >= max_attempts:
                    break
                delay = _backoff_s(attempt, retry.base_delay_ms, retry.max_delay_ms)
                delay = min(delay, 1.5)
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

        return FetchAttempt(
            url=last_url,
            status_code=int(last_status or 0),
            content_type=last_ct,
            content=last_body,
            truncated=bool(last_truncated),
            strategy_used="httpx",
            content_encoding=last_enc,
            content_length_header=last_len_hdr,
        )


__all__ = ["HttpxFetcher", "_read_with_limit"]
