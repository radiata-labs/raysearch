from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, cast
from typing_extensions import override

import anyio

from serpsage.components.fetch.base import FetcherBase
from serpsage.components.fetch.utils import (
    analyze_content,
    browser_headers,
    classify_content_kind,
    get_delay_s,
    parse_retry_after_s,
)
from serpsage.models.fetch import FetchAttempt, FetchResult

CurlSessionFactory: type[Any] | None = None
try:
    from curl_cffi.requests import AsyncSession as _CurlAsyncSession

    CurlSessionFactory = _CurlAsyncSession
    CURL_CFFI_AVAILABLE = True
except Exception:  # noqa: BLE001
    CURL_CFFI_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Callable

    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import RetrySettings


@dataclass(slots=True)
class CurlProgressiveResult:
    attempt: FetchAttempt
    finished: bool
    bytes_read: int


class CurlCffiFetcher(FetcherBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)
        if not CURL_CFFI_AVAILABLE:
            raise RuntimeError("curl_cffi is not available; install curl_cffi")
        self._session: Any | None = None

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
        progressive = await self.fetch_progressive_attempt(
            url=url,
            retry=retry,
            timeout_s=timeout_s,
            scout_bytes=None,
            continue_predicate=None,
        )
        return progressive.attempt

    async def fetch_progressive_attempt(
        self,
        *,
        url: str,
        retry: RetrySettings | None = None,
        timeout_s: float | None = None,
        scout_bytes: int | None,
        continue_predicate: Callable[[FetchAttempt], bool] | None,
    ) -> CurlProgressiveResult:
        if self._session is None:
            raise RuntimeError("curl_cffi session is not initialized")
        fetch_cfg = self.settings.fetch
        proxy = self.settings.http.proxy
        req_timeout_s = timeout_s or fetch_cfg.timeout_s
        max_attempts = max(1, int(getattr(retry, "max_attempts", 0) or 2))
        delay_ms = int(getattr(retry, "delay_ms", 0) or 90)
        last_result: CurlProgressiveResult | None = None
        for attempt_index in range(1, max_attempts + 1):
            try:
                result = await self._stream_attempt_once(
                    url=url,
                    timeout_s=req_timeout_s,
                    proxy=proxy,
                    scout_bytes=scout_bytes,
                    continue_predicate=continue_predicate,
                )
                last_result = result
                status_code = int(result.attempt.status_code or 0)
                if status_code == 429 or (500 <= status_code < 600):
                    if attempt_index >= max_attempts:
                        break
                    retry_after = parse_retry_after_s(
                        result.attempt.headers.get("retry-after")
                    )
                    delay = (
                        retry_after
                        if retry_after is not None
                        else get_delay_s(delay_ms)
                    )
                    await anyio.sleep(delay)
                    continue
                return result
            except Exception:
                if attempt_index >= max_attempts:
                    break
                await anyio.sleep(get_delay_s(delay_ms))
        if last_result is not None:
            return last_result
        return CurlProgressiveResult(
            attempt=self._build_attempt(
                url=url,
                status_code=0,
                content_type=None,
                content=b"",
                headers={},
                finished=False,
            ),
            finished=False,
            bytes_read=0,
        )

    async def _stream_attempt_once(
        self,
        *,
        url: str,
        timeout_s: float,
        proxy: str | None,
        scout_bytes: int | None,
        continue_predicate: Callable[[FetchAttempt], bool] | None,
    ) -> CurlProgressiveResult:
        fetch_cfg = self.settings.fetch
        headers = browser_headers(
            profile="browser",
            user_agent=str(fetch_cfg.user_agent),
            randomize=True,
        )
        session = cast("Any", self._session)
        async with session.stream(
            "GET",
            url,
            headers=headers,
            timeout=timeout_s,
            allow_redirects=bool(fetch_cfg.follow_redirects),
            proxy=proxy,
            impersonate=cast("Any", "chrome124"),
            http_version="v2",
            verify=True,
        ) as resp:
            status_code = int(getattr(resp, "status_code", 0) or 0)
            final_url = str(getattr(resp, "url", url) or url)
            response_headers = cast(
                "dict[str, str] | None",
                getattr(resp, "headers", None),
            )
            if not isinstance(response_headers, dict):
                response_headers = {}
            normalized_headers = {
                str(key): str(value) for key, value in response_headers.items()
            }
            content_type = normalized_headers.get("content-type")
            read_limit = self._read_limit_for_content(
                content_type=content_type,
                url=final_url,
            )
            target_scout_bytes = max(0, int(scout_bytes or 0))
            chunks: list[bytes] = []
            bytes_read = 0
            finished = True
            async for chunk in resp.aiter_content():
                if not chunk:
                    continue
                remaining = read_limit - bytes_read
                if remaining <= 0:
                    finished = False
                    break
                if len(chunk) > remaining:
                    chunk = chunk[:remaining]
                    finished = False
                chunks.append(chunk)
                bytes_read += len(chunk)
                if target_scout_bytes > 0 and bytes_read >= target_scout_bytes:
                    partial_attempt = self._build_attempt(
                        url=final_url,
                        status_code=status_code,
                        content_type=content_type,
                        content=b"".join(chunks),
                        headers=normalized_headers,
                        finished=False,
                    )
                    should_continue = True
                    if continue_predicate is not None:
                        should_continue = bool(continue_predicate(partial_attempt))
                    if not should_continue:
                        return CurlProgressiveResult(
                            attempt=partial_attempt,
                            finished=False,
                            bytes_read=bytes_read,
                        )
                    target_scout_bytes = 0
            body = b"".join(chunks)
            attempt = self._build_attempt(
                url=final_url,
                status_code=status_code,
                content_type=content_type,
                content=body,
                headers=normalized_headers,
                finished=finished,
            )
            return CurlProgressiveResult(
                attempt=attempt,
                finished=finished,
                bytes_read=bytes_read,
            )

    def _build_attempt(
        self,
        *,
        url: str,
        status_code: int,
        content_type: str | None,
        content: bytes,
        headers: dict[str, str],
        finished: bool,
    ) -> FetchAttempt:
        analysis = analyze_content(
            content=content,
            content_type=content_type,
            url=url,
            markers=tuple(self.settings.fetch.quality.blocked_markers),
        )
        attempt_chain = ["curl_cffi"]
        if not finished:
            attempt_chain.append("curl_cffi:scout")
        return FetchAttempt(
            url=url,
            status_code=int(status_code or 0),
            content_type=content_type,
            content=content,
            strategy_used="curl_cffi",
            fetch_mode="curl_cffi",
            rendered=False,
            content_kind=analysis.content_kind,
            headers=headers,
            content_encoding=headers.get("content-encoding"),
            content_length_header=headers.get("content-length"),
            content_score=float(analysis.content_score),
            text_chars=int(analysis.text_chars),
            script_ratio=float(analysis.script_ratio),
            blocked=bool(analysis.blocked),
            attempt_chain=attempt_chain,
        )

    def _read_limit_for_content(self, *, content_type: str | None, url: str) -> int:
        kind = classify_content_kind(
            content_type=content_type,
            url=url,
            content=b"",
        )
        if kind == "pdf":
            return 16_000_000
        if kind == "text":
            return 900_000
        if kind in {"unknown", "binary"}:
            return 3_000_000
        return 1_800_000


__all__ = ["CURL_CFFI_AVAILABLE", "CurlCffiFetcher", "CurlProgressiveResult"]
