from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing_extensions import override
from urllib.parse import urlparse

from serpsage.components.fetch.base import FetcherBase
from serpsage.components.fetch.utils import (
    blocked_marker_hit,
    has_nextjs_signals,
    has_spa_signals,
    normalize_route_key,
)
from serpsage.models.components.fetch import FetchAttempt, FetchResult

if TYPE_CHECKING:
    from serpsage.components.fetch.curl_cffi import CurlCffiFetcher
    from serpsage.components.fetch.playwright import PlaywrightFetcher
    from serpsage.components.http import HttpClientBase
    from serpsage.components.rate_limit import RateLimiterBase
    from serpsage.core.runtime import Runtime

_MIN_BYTES = 32
_BLOCK_STATUSES = {401, 403, 429}
_MIN_SUCCESS_EPS = 0.15


@dataclass(slots=True)
class _BackendMemory:
    samples: int = 0
    success_ema: float = 0.60
    useful_ema: float = 0.60
    latency_ema_ms: float = 700.0

    def score(self) -> float:
        probability = max(
            _MIN_SUCCESS_EPS,
            self.success_ema * 0.45 + self.useful_ema * 0.55,
        )
        return float(self.latency_ema_ms) / probability


@dataclass(slots=True)
class _RouteMemory:
    curl: _BackendMemory = field(default_factory=_BackendMemory)
    playwright: _BackendMemory = field(
        default_factory=lambda: _BackendMemory(
            success_ema=0.72,
            useful_ema=0.72,
            latency_ema_ms=1700.0,
        )
    )
    last_used_ts: float = 0.0


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
        self._curl = curl_fetcher
        self._playwright = playwright_fetcher
        self._route_memory: OrderedDict[str, _RouteMemory] = OrderedDict()
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
        route_key = normalize_route_key(url)
        route_memory = self._get_route_memory(route_key)
        direct_backend = self._choose_direct_backend(route_memory)
        if direct_backend == "playwright":
            try:
                direct_attempt = await self._run_playwright_with_learning(
                    url=url,
                    deadline_ts=deadline_ts,
                    route_key=route_key,
                    render_reason="route_memory",
                    chain_prefix=["decision:playwright:route_memory"],
                )
            except RuntimeError:
                direct_attempt = None
            if direct_attempt is not None:
                return direct_attempt
        scout_bytes = int(self.settings.fetch.auto.scout_bytes)
        curl_started = time.monotonic()
        progressive = await self._curl.fetch_progressive_attempt(
            url=url,
            timeout_s=self._remaining_timeout_s(deadline_ts),
            scout_bytes=scout_bytes,
            continue_predicate=lambda attempt: self._should_continue_curl(
                attempt=attempt,
                route_memory=route_memory,
            ),
        )
        curl_attempt = progressive.attempt
        curl_latency_ms = int((time.monotonic() - curl_started) * 1000)
        self._record_route_result(
            route_key=route_key,
            backend="curl_cffi",
            attempt=curl_attempt,
            latency_ms=curl_latency_ms,
        )
        if progressive.finished and self._is_useful(curl_attempt):
            return self._attach_attempt_chain(
                winner=curl_attempt,
                chain_prefix=["decision:curl_cffi:scout"],
            )
        if not progressive.finished and self._should_accept_truncated_curl(
            curl_attempt
        ):
            return self._attach_attempt_chain(
                winner=curl_attempt,
                chain_prefix=["decision:curl_cffi:truncated"],
            )
        playwright_reason = self._playwright_reason(curl_attempt)
        playwright_attempt = await self._run_playwright_with_learning(
            url=url,
            deadline_ts=deadline_ts,
            route_key=route_key,
            render_reason=playwright_reason,
            chain_prefix=[
                "decision:curl_cffi:scout",
                f"fallback:playwright:{playwright_reason}",
            ],
        )
        if playwright_attempt is not None:
            return playwright_attempt
        if direct_backend == "playwright" and self._is_useful(curl_attempt):
            return self._attach_attempt_chain(
                winner=curl_attempt,
                chain_prefix=["decision:playwright:route_memory", "fallback:curl_cffi"],
            )
        raise RuntimeError(
            f"fetch_unusable:auto:playwright:{int(curl_attempt.status_code)}"
        )

    def _choose_direct_backend(self, route_memory: _RouteMemory) -> str | None:
        auto_cfg = self.settings.fetch.auto
        if route_memory.playwright.samples < int(auto_cfg.direct_route_min_samples):
            return None
        if route_memory.playwright.useful_ema < float(
            auto_cfg.direct_playwright_min_useful
        ):
            return None
        if route_memory.playwright.score() > (
            route_memory.curl.score() * float(auto_cfg.direct_playwright_cost_ratio)
        ):
            return None
        return "playwright"

    def _should_continue_curl(
        self,
        *,
        attempt: FetchAttempt,
        route_memory: _RouteMemory,
    ) -> bool:
        if self._is_blocked(attempt):
            return False
        if attempt.content_kind != "html":
            return True
        quality = self.settings.fetch.quality
        low_text = int(attempt.text_chars or 0) < int(quality.min_text_chars)
        js_heavy = float(attempt.script_ratio or 0.0) >= float(
            quality.script_ratio_threshold
        )
        nextjs = has_nextjs_signals(bytes(attempt.content or b""))
        spa = has_spa_signals(bytes(attempt.content or b""))
        curl_score, _ = self._content_quality_score(attempt)
        route_prefers_playwright = bool(
            route_memory.playwright.samples
            >= int(self.settings.fetch.auto.direct_route_min_samples)
            and route_memory.playwright.score() < route_memory.curl.score()
        )
        if spa and (low_text or js_heavy):
            return False
        if nextjs and low_text:
            return False
        if js_heavy and curl_score < float(quality.quality_score_threshold):
            return False
        return not (route_prefers_playwright and curl_score < 0.55)

    def _should_accept_truncated_curl(self, attempt: FetchAttempt) -> bool:
        if attempt.content_kind in {"pdf", "text"} and self._is_useful(attempt):
            return True
        if attempt.content_kind != "html":
            return False
        return int(attempt.text_chars or 0) >= (
            int(self.settings.fetch.quality.min_text_chars) * 6
        )

    def _playwright_reason(self, attempt: FetchAttempt) -> str:
        if self._is_blocked(attempt):
            return "blocked"
        if attempt.content_kind == "html":
            if has_spa_signals(bytes(attempt.content or b"")):
                return "spa"
            if has_nextjs_signals(bytes(attempt.content or b"")):
                return "nextjs_low_text"
            if int(attempt.text_chars or 0) < int(
                self.settings.fetch.quality.min_text_chars
            ):
                return "low_text"
        return "curl_nonperfect_fallback"

    async def _run_playwright_with_learning(
        self,
        *,
        url: str,
        deadline_ts: float,
        route_key: str,
        render_reason: str,
        chain_prefix: list[str],
    ) -> FetchAttempt | None:
        started = time.monotonic()
        try:
            attempt = await self._run_playwright(
                url=url,
                deadline_ts=deadline_ts,
                render_reason=render_reason,
            )
        except Exception as exc:
            self._record_route_result(
                route_key=route_key,
                backend="playwright",
                attempt=None,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
            raise RuntimeError(
                f"fetch_unusable:auto:playwright_error:{type(exc).__name__}"
            ) from exc
        latency_ms = int((time.monotonic() - started) * 1000)
        self._record_route_result(
            route_key=route_key,
            backend="playwright",
            attempt=attempt,
            latency_ms=latency_ms,
        )
        if self._is_useful(attempt):
            return self._attach_attempt_chain(
                winner=attempt,
                chain_prefix=chain_prefix,
            )
        return None

    def _get_route_memory(self, route_key: str) -> _RouteMemory:
        memory = self._route_memory.pop(route_key, None)
        if memory is None:
            memory = _RouteMemory()
        memory.last_used_ts = time.monotonic()
        self._route_memory[route_key] = memory
        max_items = max(8, int(self.settings.fetch.auto.route_memory_size))
        while len(self._route_memory) > max_items:
            self._route_memory.popitem(last=False)
        return memory

    def _record_route_result(
        self,
        *,
        route_key: str,
        backend: str,
        attempt: FetchAttempt | None,
        latency_ms: int,
    ) -> None:
        memory = self._get_route_memory(route_key)
        stats = memory.playwright if backend == "playwright" else memory.curl
        alpha = float(self.settings.fetch.auto.learning_rate)
        success = (
            1.0 if attempt is not None and int(attempt.status_code or 0) >= 200 else 0.0
        )
        useful = 1.0 if attempt is not None and self._is_useful(attempt) else 0.0
        stats.samples += 1
        stats.success_ema = ((1.0 - alpha) * stats.success_ema) + (alpha * success)
        stats.useful_ema = ((1.0 - alpha) * stats.useful_ema) + (alpha * useful)
        stats.latency_ema_ms = ((1.0 - alpha) * stats.latency_ema_ms) + (
            alpha * float(max(1, latency_ms))
        )

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
        score, _ = self._content_quality_score(res)
        return score >= float(self.settings.fetch.quality.quality_score_threshold)

    def _content_quality_score(self, res: FetchAttempt) -> tuple[float, list[str]]:
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
        len_ratio = min(1.0, max(0, content_len - _MIN_BYTES) / (1200 - _MIN_BYTES))
        score += len_ratio * 0.22
        if len_ratio < 0.25:
            reasons.append(f"short_content:{content_len}")
        if res.content_kind == "html":
            text_ratio = min(
                1.0,
                max(0, text_chars - int(quality.min_text_chars))
                / (1800 - int(quality.min_text_chars)),
            )
            score += text_ratio * 0.48
            if text_ratio < 0.3:
                reasons.append(f"low_text:{text_chars}")
        else:
            score += content_score * 0.48
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


__all__ = ["AutoFetcher"]
