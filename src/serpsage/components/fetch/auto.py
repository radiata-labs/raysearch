from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING
from typing_extensions import override
from urllib.parse import urlparse

import httpx

from serpsage.components.fetch.base import FetcherBase
from serpsage.components.fetch.utils import (
    blocked_marker_hit,
    browser_headers,
    classify_content_kind,
    estimate_text_quality,
    has_nextjs_signals,
    has_spa_signals,
)
from serpsage.models.fetch import FetchAttempt, FetchResult

if TYPE_CHECKING:
    from serpsage.components.fetch.curl_cffi import CurlCffiFetcher
    from serpsage.components.fetch.playwright import PlaywrightFetcher
    from serpsage.components.http import HttpClientBase
    from serpsage.components.rate_limit import RateLimiterBase
    from serpsage.core.runtime import Runtime
_MIN_BYTES = 32
_PROBE_MAX_BYTES = 50_000
_BLOCK_STATUSES = {401, 403, 429}


@dataclass(slots=True)
class _ProbeSnapshot:
    status_code: int = 0
    final_url: str = ""
    content_type: str | None = None
    content_kind: str = "unknown"
    text_chars: int = 0
    script_ratio: float = 0.0
    blocked_marker: bool = False
    anti_bot: bool = False
    nextjs: bool = False
    spa: bool = False
    low_text: bool = False
    error_type: str | None = None
    bytes_read: int = 0


class AutoFetcher(FetcherBase):
    def __init__(
        self,
        *,
        rt: Runtime,
        rate_limiter: RateLimiterBase,
        http: HttpClientBase,
        curl_fetcher: CurlCffiFetcher,
        playwright_fetcher: PlaywrightFetcher,
    ) -> None:
        super().__init__(rt=rt)
        self._rl = rate_limiter
        self._http = http.client
        self._curl = curl_fetcher
        self._playwright = playwright_fetcher
        self.bind_deps(
            rate_limiter,
            http,
            curl_fetcher,
            playwright_fetcher,
        )

    @override
    async def _afetch_inner(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
    ) -> FetchResult:
        return await self._afetch(url=url, timeout_s=timeout_s)

    async def _afetch(
        self,
        *,
        url: str,
        timeout_s: float | None,
    ) -> FetchResult:
        host = urlparse(url).netloc.lower()
        await self._rl.acquire(host=host)
        try:
            attempt = await self._fetch_useful(
                url=url,
                strategy=str(self.settings.fetch.backend or "auto").lower(),
                timeout_s=timeout_s,
            )
        finally:
            await self._rl.release(host=host)
        return FetchResult(
            url=attempt.url,
            status_code=int(attempt.status_code),
            content_type=attempt.content_type,
            content=attempt.content,
            fetch_mode=attempt.fetch_mode,
            rendered=bool(attempt.rendered),
            content_kind=attempt.content_kind,
            headers=dict(attempt.headers or {}),
            attempt_chain=list(attempt.attempt_chain or []),
        )

    async def _fetch_useful(
        self,
        *,
        url: str,
        strategy: str,
        timeout_s: float | None,
    ) -> FetchAttempt:
        deadline_ts = time.monotonic() + self._resolve_timeout_s(timeout_s)
        quality_cfg = self.settings.fetch.quality
        if strategy == "curl_cffi":
            attempt = await self._run_curl(
                url=url,
                deadline_ts=deadline_ts,
            )
            score, _ = self._content_quality_score(attempt)
            if score >= float(quality_cfg.quality_score_threshold):
                return attempt
            raise RuntimeError(f"fetch_unusable:curl_cffi:{int(attempt.status_code)}")
        if strategy == "playwright":
            attempt = await self._run_playwright(
                url=url,
                deadline_ts=deadline_ts,
                render_reason="backend_playwright",
            )
            score, _ = self._content_quality_score(attempt)
            if score >= float(quality_cfg.quality_score_threshold):
                return attempt
            raise RuntimeError(f"fetch_unusable:playwright:{int(attempt.status_code)}")
        if strategy != "auto":
            raise ValueError(
                "unsupported fetch backend "
                f"`{strategy}`; expected curl_cffi|playwright|auto"
            )
        return await self._fetch_auto(
            url=url,
            deadline_ts=deadline_ts,
        )

    async def _fetch_auto(
        self,
        *,
        url: str,
        deadline_ts: float,
    ) -> FetchAttempt:
        probe = await self._probe_http(url=url, deadline_ts=deadline_ts)
        selected_backend, selection_reason = self._choose_backend(probe)
        probe_chain = self._probe_chain_item(probe)
        decision_chain = f"decision:{selected_backend}:{selection_reason}"
        if selected_backend == "playwright":
            try:
                attempt = await self._run_playwright(
                    url=url,
                    deadline_ts=deadline_ts,
                    render_reason=f"auto_probe:{selection_reason}",
                )
            except Exception as exc:  # noqa: BLE001
                raise RuntimeError(
                    f"fetch_unusable:auto:playwright_error:{type(exc).__name__}"
                ) from exc
            if self._is_useful(attempt):
                return self._attach_attempt_chain(
                    winner=attempt,
                    chain_prefix=[probe_chain, decision_chain],
                )
            raise RuntimeError(
                f"fetch_unusable:auto:playwright:{int(attempt.status_code)}"
            )
        curl_attempt: FetchAttempt | None = None
        curl_error_type: str | None = None
        try:
            curl_attempt = await self._run_curl(
                url=url,
                deadline_ts=deadline_ts,
            )
        except Exception as exc:  # noqa: BLE001
            curl_error_type = type(exc).__name__
        force_playwright_for_nextjs = bool(
            curl_attempt is not None
            and curl_attempt.content_kind == "html"
            and has_nextjs_signals(bytes(curl_attempt.content or b""))
        )
        if (
            curl_attempt is not None
            and self._is_useful(curl_attempt)
            and not force_playwright_for_nextjs
        ):
            return self._attach_attempt_chain(
                winner=curl_attempt,
                chain_prefix=[probe_chain, decision_chain],
            )
        try:
            playwright_attempt = await self._run_playwright(
                url=url,
                deadline_ts=deadline_ts,
                render_reason="curl_nonperfect_fallback",
            )
        except Exception as exc:  # noqa: BLE001
            raise RuntimeError(
                f"fetch_unusable:auto:playwright_fallback_error:{type(exc).__name__}"
            ) from exc
        if self._is_useful(playwright_attempt):
            prefix = [probe_chain, decision_chain, "fallback:playwright"]
            if curl_error_type is not None:
                prefix.append(f"curl_error:{curl_error_type}")
            elif curl_attempt is not None:
                prefix.extend(list(curl_attempt.attempt_chain or ["curl_cffi"]))
            return self._attach_attempt_chain(
                winner=playwright_attempt,
                chain_prefix=prefix,
            )
        raise RuntimeError(
            f"fetch_unusable:auto:playwright:{int(playwright_attempt.status_code)}"
        )

    async def _probe_http(
        self,
        *,
        url: str,
        deadline_ts: float,
    ) -> _ProbeSnapshot:
        snap = _ProbeSnapshot(final_url=url)
        try:
            timeout_s = self._remaining_timeout_s(deadline_ts)
            timeout = httpx.Timeout(timeout_s)
            fetch_cfg = self.settings.fetch
            probe_headers = browser_headers(
                profile="browser",
                user_agent=str(fetch_cfg.user_agent),
                randomize=True,
            )
            # Prefer gzip/deflate so body decoding remains analyzable for probe logic.
            probe_headers["Accept-Encoding"] = "gzip, deflate"
            async with self._http.stream(
                "GET",
                url,
                headers=probe_headers,
                timeout=timeout,
                follow_redirects=bool(fetch_cfg.follow_redirects),
            ) as resp:
                snap.status_code = int(resp.status_code)
                snap.final_url = str(resp.url)
                snap.content_type = resp.headers.get("content-type")
                body = await self._read_probe_body(resp=resp)
            quality = self.settings.fetch.quality
            content_kind = classify_content_kind(
                content_type=snap.content_type,
                url=snap.final_url,
                content=body,
            )
            text_chars, _, script_ratio = estimate_text_quality(
                body,
                content_kind=content_kind,
            )
            marker_hit = bool(
                blocked_marker_hit(
                    body,
                    markers=tuple(quality.blocked_markers or []),
                )
            )
            anti_bot = int(snap.status_code) in _BLOCK_STATUSES or marker_hit
            nextjs = bool(content_kind == "html" and has_nextjs_signals(body))
            spa = bool(
                content_kind == "html"
                and script_ratio >= float(quality.script_ratio_threshold)
                and has_spa_signals(body)
            )
            low_text = bool(
                content_kind == "html" and int(text_chars) < int(quality.min_text_chars)
            )
            snap.content_kind = content_kind
            snap.text_chars = int(text_chars)
            snap.script_ratio = float(script_ratio)
            snap.blocked_marker = marker_hit
            snap.anti_bot = anti_bot
            snap.nextjs = nextjs
            snap.spa = spa
            snap.low_text = low_text
            snap.bytes_read = int(len(body))
        except Exception as exc:  # noqa: BLE001
            snap.error_type = type(exc).__name__
        return snap

    async def _read_probe_body(self, *, resp: httpx.Response) -> bytes:
        total = 0
        parts: list[bytes] = []
        async for chunk in resp.aiter_bytes():
            if not chunk:
                continue
            remain = _PROBE_MAX_BYTES - total
            if remain <= 0:
                break
            if len(chunk) > remain:
                chunk = chunk[:remain]
            parts.append(chunk)
            total += len(chunk)
            if total >= _PROBE_MAX_BYTES:
                break
        return b"".join(parts)

    def _choose_backend(self, probe: _ProbeSnapshot) -> tuple[str, str]:
        if probe.error_type is not None:
            return "curl_cffi", "probe_error"
        if probe.anti_bot:
            return "playwright", "anti_bot"
        if probe.nextjs:
            return "playwright", "nextjs"
        if probe.spa:
            return "playwright", "spa"
        if probe.low_text:
            return "playwright", "low_text"
        return "curl_cffi", "default"

    async def _run_curl(
        self,
        *,
        url: str,
        deadline_ts: float,
    ) -> FetchAttempt:
        timeout_s = self._remaining_timeout_s(deadline_ts)
        return await self._curl.fetch_attempt(
            url=url,
            timeout_s=timeout_s,
        )

    async def _run_playwright(
        self,
        *,
        url: str,
        deadline_ts: float,
        render_reason: str,
    ) -> FetchAttempt:
        timeout_s = self._remaining_timeout_s(deadline_ts)
        return await self._playwright.fetch_attempt(
            url=url,
            timeout_s=timeout_s,
            render_reason=render_reason,
        )

    def _resolve_timeout_s(self, timeout_s: float | None) -> float:
        resolved = float(timeout_s or 0.0)
        if resolved <= 0.0:
            resolved = float(self.settings.fetch.timeout_s)
        return max(0.05, resolved)

    def _remaining_timeout_s(self, deadline_ts: float) -> float:
        remaining = float(deadline_ts - time.monotonic())
        if remaining <= 0.0:
            raise TimeoutError("fetch timeout reached before backend request")
        return remaining

    def _is_blocked(self, res: FetchAttempt) -> bool:
        status = int(res.status_code or 0)
        if status in _BLOCK_STATUSES:
            return True
        if blocked_marker_hit(
            res.content,
            markers=tuple(self.settings.fetch.quality.blocked_markers or []),
        ):
            return True
        return bool(res.blocked)

    def _is_useful(self, res: FetchAttempt) -> bool:
        """Legacy method for backward compatibility."""
        score, _ = self._content_quality_score(res)
        return score >= 0.15

    def _content_quality_score(self, res: FetchAttempt) -> tuple[float, list[str]]:
        """Calculate a quality score (0.0-1.0) for fetched content."""
        quality = self.settings.fetch.quality
        reasons: list[str] = []
        status = int(res.status_code or 0)
        content_len = len(res.content or b"")
        text_chars = int(res.text_chars or 0)
        content_score = float(res.content_score or 0.0)
        score = 0.0
        if status < 200 or status >= 400:
            reasons.append(f"bad_status:{status}")
            return 0.0, reasons
        if self._is_blocked(res):
            reasons.append("blocked")
            return 0.0, reasons
        len_ratio = min(1.0, max(0, content_len - _MIN_BYTES) / (1000 - _MIN_BYTES))
        score += len_ratio * 0.25
        if len_ratio < 0.3:
            reasons.append(f"short_content:{content_len}")
        if res.content_kind == "html":
            text_ratio = min(
                1.0,
                max(0, text_chars - int(quality.min_text_chars))
                / (1000 - int(quality.min_text_chars)),
            )
            score += text_ratio * 0.45
            if text_ratio < 0.3:
                reasons.append(f"low_text:{text_chars}")
        else:
            score += content_score * 0.45
        if res.content_kind not in {"binary", "unknown"}:
            score += 0.15
        else:
            reasons.append(f"unknown_content_kind:{res.content_kind}")
        script_ratio = float(res.script_ratio or 0.0)
        script_score = max(0, 1.0 - script_ratio * 2)
        score += script_score * 0.15
        return score, reasons

    def _attach_attempt_chain(
        self,
        *,
        winner: FetchAttempt,
        chain_prefix: list[str],
    ) -> FetchAttempt:
        out: list[str] = []
        for item in chain_prefix + list(winner.attempt_chain or [winner.fetch_mode]):
            token = str(item or "").strip()
            if not token or token in out:
                continue
            out.append(token)
        if not out:
            out = [winner.fetch_mode]
        return winner.model_copy(update={"attempt_chain": out})

    def _probe_chain_item(self, probe: _ProbeSnapshot) -> str:
        if probe.error_type is not None:
            return f"probe:error:{probe.error_type}"
        return f"probe:http:get:{int(probe.status_code or 0)}"


__all__ = ["AutoFetcher"]
