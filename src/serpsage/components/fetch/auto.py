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
    has_spa_signals,
)
from serpsage.models.fetch import FetchAttempt, FetchResult

if TYPE_CHECKING:
    from serpsage.components.fetch.curl_cffi import CurlCffiFetcher
    from serpsage.components.fetch.playwright import PlaywrightFetcher
    from serpsage.components.http import HttpClientBase
    from serpsage.components.rate_limit import RateLimiterBase
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase

_MIN_BYTES = 32
_PROBE_MAX_BYTES = 220_000
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
        backend = str(self.settings.fetch.backend or "auto").lower()
        with self.span("fetch.auto", url=url, strategy=backend) as sp:
            return await self._afetch(
                url=url,
                timeout_s=timeout_s,
                span=sp,
            )

    async def _afetch(
        self,
        *,
        url: str,
        timeout_s: float | None,
        span: SpanBase,
    ) -> FetchResult:
        host = urlparse(url).netloc.lower()
        await self._rl.acquire(host=host)
        try:
            attempt = await self._fetch_useful(
                url=url,
                strategy=str(self.settings.fetch.backend or "auto").lower(),
                span=span,
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
        span: SpanBase,
        timeout_s: float | None,
    ) -> FetchAttempt:
        deadline_ts = time.monotonic() + self._resolve_timeout_s(timeout_s)
        quality_cfg = self.settings.fetch.quality
        if strategy == "curl_cffi":
            attempt = await self._run_curl(
                url=url,
                span=span,
                deadline_ts=deadline_ts,
            )
            # Use quality score instead of binary check for more lenient acceptance
            score, _ = self._content_quality_score(attempt)
            if score >= float(quality_cfg.quality_score_threshold):
                return attempt
            raise RuntimeError(f"fetch_unusable:curl_cffi:{int(attempt.status_code)}")

        if strategy == "playwright":
            attempt = await self._run_playwright(
                url=url,
                span=span,
                deadline_ts=deadline_ts,
                render_reason="backend_playwright",
            )
            # Use quality score instead of binary check for more lenient acceptance
            score, _ = self._content_quality_score(attempt)
            if score >= float(quality_cfg.quality_score_threshold):
                return attempt
            raise RuntimeError(f"fetch_unusable:playwright:{int(attempt.status_code)}")

        if strategy != "auto":
            raise ValueError(
                f"unsupported fetch backend `{strategy}`; expected curl_cffi|playwright|auto"
            )

        return await self._fetch_auto(
            url=url,
            span=span,
            deadline_ts=deadline_ts,
        )

    async def _fetch_auto(
        self,
        *,
        url: str,
        span: SpanBase,
        deadline_ts: float,
    ) -> FetchAttempt:
        probe = await self._probe_http(url=url, deadline_ts=deadline_ts, span=span)
        self._set_probe_attrs(span=span, probe=probe)
        selected_backend, selection_reason = self._choose_backend(probe)
        span.set_attr("selected_backend", selected_backend)
        span.set_attr("selection_reason", selection_reason)
        span.set_attr("fallback_triggered", False)

        probe_chain = self._probe_chain_item(probe)
        decision_chain = f"decision:{selected_backend}:{selection_reason}"

        if selected_backend == "playwright":
            try:
                attempt = await self._run_playwright(
                    url=url,
                    span=span,
                    deadline_ts=deadline_ts,
                    render_reason=f"auto_probe:{selection_reason}",
                )
            except Exception as exc:  # noqa: BLE001
                span.set_attr("playwright_error_type", type(exc).__name__)
                raise RuntimeError(
                    f"fetch_unusable:auto:playwright_error:{type(exc).__name__}"
                ) from exc
            if self._is_useful(attempt):
                winner = self._attach_attempt_chain(
                    winner=attempt,
                    chain_prefix=[probe_chain, decision_chain],
                )
                self._set_winner_attrs(span=span, winner=winner)
                return winner
            raise RuntimeError(
                f"fetch_unusable:auto:playwright:{int(attempt.status_code)}"
            )

        curl_attempt: FetchAttempt | None = None
        curl_error_type: str | None = None
        try:
            curl_attempt = await self._run_curl(
                url=url,
                span=span,
                deadline_ts=deadline_ts,
            )
        except Exception as exc:  # noqa: BLE001
            curl_error_type = type(exc).__name__
            span.set_attr("curl_error_type", curl_error_type)

        if curl_attempt is not None and self._is_useful(curl_attempt):
            winner = self._attach_attempt_chain(
                winner=curl_attempt,
                chain_prefix=[probe_chain, decision_chain],
            )
            self._set_winner_attrs(span=span, winner=winner)
            return winner

        span.set_attr("fallback_triggered", True)
        fallback_reason = "curl_nonperfect"
        if curl_error_type is not None:
            fallback_reason = f"curl_error:{curl_error_type}"
        span.set_attr("fallback_reason", fallback_reason)

        try:
            playwright_attempt = await self._run_playwright(
                url=url,
                span=span,
                deadline_ts=deadline_ts,
                render_reason="curl_nonperfect_fallback",
            )
        except Exception as exc:  # noqa: BLE001
            span.set_attr("playwright_error_type", type(exc).__name__)
            raise RuntimeError(
                f"fetch_unusable:auto:playwright_fallback_error:{type(exc).__name__}"
            ) from exc

        if self._is_useful(playwright_attempt):
            prefix = [probe_chain, decision_chain, "fallback:playwright"]
            if curl_error_type is not None:
                prefix.append(f"curl_error:{curl_error_type}")
            elif curl_attempt is not None:
                prefix.extend(list(curl_attempt.attempt_chain or ["curl_cffi"]))
            winner = self._attach_attempt_chain(
                winner=playwright_attempt,
                chain_prefix=prefix,
            )
            self._set_winner_attrs(span=span, winner=winner)
            return winner
        raise RuntimeError(
            f"fetch_unusable:auto:playwright:{int(playwright_attempt.status_code)}"
        )

    async def _probe_http(
        self,
        *,
        url: str,
        deadline_ts: float,
        span: SpanBase,
    ) -> _ProbeSnapshot:
        with self.span("fetch.auto.probe", url=url) as probe_span:
            snap = _ProbeSnapshot(final_url=url)
            try:
                timeout_s = self._remaining_timeout_s(deadline_ts)
                timeout = httpx.Timeout(timeout_s)
                fetch_cfg = self.settings.fetch
                async with self._http.stream(
                    "GET",
                    url,
                    headers=browser_headers(
                        profile="browser",
                        user_agent=str(fetch_cfg.user_agent),
                        randomize=True,  # Use random real browser UA for probing
                    ),
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
                spa = bool(
                    content_kind == "html"
                    and script_ratio >= float(quality.script_ratio_threshold)
                    and has_spa_signals(body)
                )
                low_text = bool(
                    content_kind == "html"
                    and int(text_chars) < int(quality.min_text_chars)
                )
                snap.content_kind = content_kind
                snap.text_chars = int(text_chars)
                snap.script_ratio = float(script_ratio)
                snap.blocked_marker = marker_hit
                snap.anti_bot = anti_bot
                snap.spa = spa
                snap.low_text = low_text
                snap.bytes_read = int(len(body))
            except Exception as exc:  # noqa: BLE001
                snap.error_type = type(exc).__name__
                probe_span.set_attr("probe_error_type", snap.error_type)
                span.set_attr("probe_error_type", snap.error_type)

            probe_span.set_attr("status_code", int(snap.status_code))
            probe_span.set_attr("content_kind", str(snap.content_kind))
            probe_span.set_attr("text_chars", int(snap.text_chars))
            probe_span.set_attr("script_ratio", float(snap.script_ratio))
            probe_span.set_attr("anti_bot", bool(snap.anti_bot))
            probe_span.set_attr("spa", bool(snap.spa))
            probe_span.set_attr("low_text", bool(snap.low_text))
            probe_span.set_attr("bytes_read", int(snap.bytes_read))
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
        if probe.spa:
            return "playwright", "spa"
        if probe.low_text:
            return "playwright", "low_text"
        return "curl_cffi", "default"

    async def _run_curl(
        self,
        *,
        url: str,
        span: SpanBase,
        deadline_ts: float,
    ) -> FetchAttempt:
        with self.span("fetch.auto.backend", url=url, backend="curl_cffi") as csp:
            timeout_s = self._remaining_timeout_s(deadline_ts)
            csp.set_attr("timeout_s", float(timeout_s))
            attempt = await self._curl.fetch_attempt(
                url=url,
                span=csp,
                timeout_s=timeout_s,
            )
            useful = self._is_useful(attempt)
            csp.set_attr("useful", bool(useful))
            span.set_attr("curl_status", int(attempt.status_code or 0))
            span.set_attr("curl_useful", bool(useful))
            return attempt

    async def _run_playwright(
        self,
        *,
        url: str,
        span: SpanBase,
        deadline_ts: float,
        render_reason: str,
    ) -> FetchAttempt:
        with self.span("fetch.auto.backend", url=url, backend="playwright") as psp:
            timeout_s = self._remaining_timeout_s(deadline_ts)
            psp.set_attr("timeout_s", float(timeout_s))
            attempt = await self._playwright.fetch_attempt(
                url=url,
                span=psp,
                timeout_s=timeout_s,
                render_reason=render_reason,
            )
            useful = self._is_useful(attempt)
            psp.set_attr("useful", bool(useful))
            span.set_attr("playwright_status", int(attempt.status_code or 0))
            span.set_attr("playwright_useful", bool(useful))

            # Record detailed failure reasons for debugging
            if not useful:
                quality = self.settings.fetch.quality
                status = int(attempt.status_code or 0)
                content_len = len(attempt.content or b"")
                text_chars = int(attempt.text_chars or 0)

                failure_reasons: list[str] = []
                if status < 200 or status >= 400:
                    failure_reasons.append(f"bad_status:{status}")
                if self._is_blocked(attempt):
                    failure_reasons.append("blocked")
                if content_len < _MIN_BYTES:
                    failure_reasons.append(f"content_too_short:{content_len}")
                if attempt.content_kind in {"binary", "unknown"}:
                    failure_reasons.append(f"bad_content_kind:{attempt.content_kind}")
                if attempt.content_kind == "html" and text_chars < int(
                    quality.min_text_chars
                ):
                    failure_reasons.append(f"low_text_chars:{text_chars}")

                if failure_reasons:
                    span.set_attr(
                        "playwright_failure_reasons", ";".join(failure_reasons)
                    )
                    psp.set_attr("failure_reasons", ";".join(failure_reasons))

            return attempt

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
        """Legacy method for backward compatibility.

        Delegates to _content_quality_score with threshold of 0.15.
        """
        score, _ = self._content_quality_score(res)
        return score >= 0.15

    def _content_quality_score(self, res: FetchAttempt) -> tuple[float, list[str]]:
        """Calculate a quality score (0.0-1.0) for fetched content.

        Returns:
            Tuple of (score, failure_reasons) where:
            - score: 0.0 (unusable) to 1.0 (excellent)
            - failure_reasons: List of reasons for score reduction

        Scoring breakdown:
        - HTTP status valid (200-399): Required, else 0.0
        - Not blocked: Required, else 0.0
        - Content length: Up to 0.25 points
        - Text content: Up to 0.45 points
        - Content kind recognized: Up to 0.15 points
        - Good script ratio: Up to 0.15 points
        """
        quality = self.settings.fetch.quality
        reasons: list[str] = []

        # Base score components
        status = int(res.status_code or 0)
        content_len = len(res.content or b"")
        text_chars = int(res.text_chars or 0)
        content_score = float(res.content_score or 0.0)

        score = 0.0

        # Check 1: HTTP status must be 2xx or 3xx
        if status < 200 or status >= 400:
            reasons.append(f"bad_status:{status}")
            return 0.0, reasons

        # Check 2: Must not be blocked
        if self._is_blocked(res):
            reasons.append("blocked")
            return 0.0, reasons

        # Score component 1: Content length (0.25 points max)
        # 32 bytes = 0, 1000+ bytes = full points
        len_ratio = min(1.0, max(0, content_len - _MIN_BYTES) / (1000 - _MIN_BYTES))
        score += len_ratio * 0.25
        if len_ratio < 0.3:
            reasons.append(f"short_content:{content_len}")

        # Score component 2: Text characters for HTML (0.45 points max)
        # 100 chars = 0, 1000+ chars = full points
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
            # Non-HTML gets partial points based on content_score
            score += content_score * 0.45

        # Score component 3: Content kind recognized (0.15 points)
        if res.content_kind not in {"binary", "unknown"}:
            score += 0.15
        else:
            reasons.append(f"unknown_content_kind:{res.content_kind}")

        # Score component 4: Script ratio (0.15 points max)
        # Lower script ratio = higher score
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

    def _set_probe_attrs(self, *, span: SpanBase, probe: _ProbeSnapshot) -> None:
        span.set_attr("probe_status", int(probe.status_code))
        span.set_attr("probe_content_kind", str(probe.content_kind))
        span.set_attr("probe_text_chars", int(probe.text_chars))
        span.set_attr("probe_script_ratio", float(probe.script_ratio))
        span.set_attr("probe_blocked_marker", bool(probe.blocked_marker))
        span.set_attr("probe_anti_bot", bool(probe.anti_bot))
        span.set_attr("probe_spa", bool(probe.spa))
        span.set_attr("probe_low_text", bool(probe.low_text))
        span.set_attr("probe_bytes", int(probe.bytes_read))
        if probe.error_type is not None:
            span.set_attr("probe_error_type", probe.error_type)

    def _set_winner_attrs(self, *, span: SpanBase, winner: FetchAttempt) -> None:
        span.set_attr("winner_mode", str(winner.fetch_mode))
        span.set_attr("winner_status", int(winner.status_code or 0))
        span.set_attr("winner_content_kind", str(winner.content_kind))
        span.set_attr("winner_text_chars", int(winner.text_chars or 0))
        span.set_attr("winner_content_score", float(winner.content_score or 0.0))


__all__ = ["AutoFetcher"]
