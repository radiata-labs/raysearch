from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

import anyio

from serpsage.components.fetch.base import FetcherBase
from serpsage.components.fetch.utils import (
    blocked_marker_hit,
    browser_headers,
    classify_content_kind,
    estimate_text_quality,
    get_delay_s,
    parse_retry_after_s,
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
    async def _afetch_inner(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
    ) -> FetchResult:
        res = await self.fetch_attempt(
            url=url,
            timeout_s=timeout_s,
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
        retry: RetrySettings | None = None,
        timeout_s: float | None = None,
    ) -> FetchAttempt:
        if self._session is None:
            raise RuntimeError("curl_cffi session is not initialized")
        fetch_cfg = self.settings.fetch
        proxy = self.settings.http.proxy
        req_timeout_s = timeout_s or fetch_cfg.timeout_s
        max_attempts = max(
            1,
            int(
                getattr(retry, "max_attempts", 0) or 2,
            ),
        )
        delay_ms = int(getattr(retry, "delay_ms", 0) or 90)

        last_status: int | None = None
        last_url = url
        last_ct: str | None = None
        last_body: bytes = b""
        last_enc: str | None = None
        last_len_hdr: str | None = None
        last_headers: dict[str, str] = {}
        last_truncated = False

        for _attempt in range(1, max_attempts + 1):
            try:
                resp = await self._session.get(
                    url,
                    headers=browser_headers(
                        profile="browser",
                        user_agent=str(fetch_cfg.user_agent),
                        randomize=True,  # Use random UA for anti-fingerprinting
                    ),
                    timeout=req_timeout_s,
                    allow_redirects=bool(fetch_cfg.follow_redirects),
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
                )

                if last_status == 429 or (500 <= last_status < 600):
                    if _attempt >= max_attempts:
                        break
                    ra = parse_retry_after_s(last_headers.get("retry-after"))
                    delay = ra if ra is not None else get_delay_s(delay_ms)
                    await anyio.sleep(delay)
                    continue
                break
            except Exception:  # noqa: BLE001
                if _attempt >= max_attempts:
                    break
                delay = get_delay_s(delay_ms)
                await anyio.sleep(delay)
                continue

        content_kind = classify_content_kind(
            content_type=last_ct,
            url=last_url,
            content=last_body,
        )
        text_chars, content_score, script_ratio = estimate_text_quality(
            last_body,
            content_kind=content_kind,
        )
        blocked = bool(
            blocked_marker_hit(
                last_body,
                markers=tuple(self.settings.fetch.quality.blocked_markers),
            )
        )

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
            script_ratio=float(script_ratio),
            blocked=blocked,
            attempt_chain=["curl_cffi"],
        )

    def _truncate_by_kind(
        self,
        *,
        body: bytes,
        content_type: str | None,
        url: str,
    ) -> tuple[bytes, bool]:
        kind = classify_content_kind(
            content_type=content_type, url=url, content=body[:128]
        )
        if kind == "pdf":
            budget = 16_000_000
        elif kind == "text":
            budget = 900_000
        elif kind in {"unknown", "binary"}:
            budget = 3_000_000
        else:
            budget = 1_800_000
        if len(body) <= budget:
            return body, False
        return body[:budget], True


__all__ = ["CURL_CFFI_AVAILABLE", "CurlCffiFetcher"]
