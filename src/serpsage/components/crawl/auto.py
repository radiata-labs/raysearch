from __future__ import annotations

import time
from collections import OrderedDict
from dataclasses import dataclass, field
from typing import Any, cast
from typing_extensions import override
from urllib.parse import urlparse

from serpsage.components.base import ComponentMeta
from serpsage.components.crawl.base import CrawlConfigBase, CrawlerBase
from serpsage.components.crawl.utils import (
    blocked_marker_hit,
    has_nextjs_signals,
    has_spa_signals,
    normalize_route_key,
)
from serpsage.components.rate_limit.base import RateLimiterBase
from serpsage.models.components.crawl import CrawlAttempt, CrawlResult

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


class AutoCrawlerConfig(CrawlConfigBase):
    __setting_family__ = "crawl"
    __setting_name__ = "auto"


_AUTO_CRAWLER_META = ComponentMeta(
    version="1.0.0",
    summary="Adaptive crawler choosing curl_cffi or Playwright.",
)


class AutoCrawler(CrawlerBase[AutoCrawlerConfig]):
    meta = _AUTO_CRAWLER_META

    route_memory: OrderedDict[str, _RouteMemory]

    def __init__(self) -> None:
        super().__init__()
        self._rate_limiter = _coerce_rate_limiter(
            self.components.require_default_component("rate_limit")
        )
        self._curl = _coerce_crawler(
            "curl_cffi",
            self.components.require_component_optional("crawl", "curl_cffi"),
        )
        self._playwright = _coerce_crawler(
            "playwright",
            self.components.require_component_optional("crawl", "playwright"),
        )
        self.route_memory: OrderedDict[str, _RouteMemory] = OrderedDict()
        self.bind_deps(self._rate_limiter, self._curl, self._playwright)

    @override
    async def _acrawl_inner(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
    ) -> CrawlResult:
        return await self._acrawl(url=url, timeout_s=timeout_s)

    async def _acrawl(
        self,
        *,
        url: str,
        timeout_s: float | None,
    ) -> CrawlResult:
        host = urlparse(url).netloc.lower()
        await self._rate_limiter.acquire(host=host)
        try:
            attempt = await self._crawl_useful(url=url, timeout_s=timeout_s)
        finally:
            await self._rate_limiter.release(host=host)
        return CrawlResult(
            url=attempt.url,
            status_code=int(attempt.status_code),
            content_type=attempt.content_type,
            content=attempt.content,
            crawl_backend=attempt.crawl_backend,
            rendered=bool(attempt.rendered),
            content_kind=attempt.content_kind,
            headers=dict(attempt.headers or {}),
            attempt_chain=list(attempt.attempt_chain or []),
        )

    async def _crawl_useful(
        self,
        *,
        url: str,
        timeout_s: float | None,
    ) -> CrawlAttempt:
        deadline_ts = time.monotonic() + self._resolve_timeout_s(timeout_s)
        return await self._crawl_auto(url=url, deadline_ts=deadline_ts)

    async def _crawl_auto(
        self,
        *,
        url: str,
        deadline_ts: float,
    ) -> CrawlAttempt:
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
        scout_bytes = int(self.config.auto.scout_bytes)
        curl_started = time.monotonic()
        curl = _require_crawler("curl_cffi", self._curl)
        progressive = await cast("Any", curl).crawl_progressive_attempt(
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
            f"crawl_unusable:auto:playwright:{int(curl_attempt.status_code)}"
        )

    def _choose_direct_backend(self, route_memory: _RouteMemory) -> str | None:
        auto_cfg = self.config.auto
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
        attempt: CrawlAttempt,
        route_memory: _RouteMemory,
    ) -> bool:
        if self._is_blocked(attempt):
            return False
        if attempt.content_kind != "html":
            return True
        quality = self.config.quality
        low_text = int(attempt.text_chars or 0) < int(quality.min_text_chars)
        js_heavy = float(attempt.script_ratio or 0.0) >= float(
            quality.script_ratio_threshold
        )
        nextjs = has_nextjs_signals(bytes(attempt.content or b""))
        spa = has_spa_signals(bytes(attempt.content or b""))
        curl_score, _ = self._content_quality_score(attempt)
        route_prefers_playwright = bool(
            route_memory.playwright.samples
            >= int(self.config.auto.direct_route_min_samples)
            and route_memory.playwright.score() < route_memory.curl.score()
        )
        if spa and (low_text or js_heavy):
            return False
        if nextjs and low_text:
            return False
        if js_heavy and curl_score < float(quality.quality_score_threshold):
            return False
        return not (route_prefers_playwright and curl_score < 0.55)

    def _should_accept_truncated_curl(self, attempt: CrawlAttempt) -> bool:
        if attempt.content_kind in {"pdf", "text"} and self._is_useful(attempt):
            return True
        if attempt.content_kind != "html":
            return False
        return int(attempt.text_chars or 0) >= (
            int(self.config.quality.min_text_chars) * 6
        )

    def _playwright_reason(self, attempt: CrawlAttempt) -> str:
        if self._is_blocked(attempt):
            return "blocked"
        if attempt.content_kind == "html":
            if has_spa_signals(bytes(attempt.content or b"")):
                return "spa"
            if has_nextjs_signals(bytes(attempt.content or b"")):
                return "nextjs_low_text"
            if int(attempt.text_chars or 0) < int(self.config.quality.min_text_chars):
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
    ) -> CrawlAttempt | None:
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
                f"crawl_unusable:auto:playwright_error:{type(exc).__name__}"
            ) from exc
        latency_ms = int((time.monotonic() - started) * 1000)
        self._record_route_result(
            route_key=route_key,
            backend="playwright",
            attempt=attempt,
            latency_ms=latency_ms,
        )
        if self._is_useful(attempt):
            return self._attach_attempt_chain(winner=attempt, chain_prefix=chain_prefix)
        return None

    def _get_route_memory(self, route_key: str) -> _RouteMemory:
        memory = self.route_memory.pop(route_key, None)
        if memory is None:
            memory = _RouteMemory()
        memory.last_used_ts = time.monotonic()
        self.route_memory[route_key] = memory
        max_items = max(8, int(self.config.auto.route_memory_size))
        while len(self.route_memory) > max_items:
            self.route_memory.popitem(last=False)
        return memory

    def _record_route_result(
        self,
        *,
        route_key: str,
        backend: str,
        attempt: CrawlAttempt | None,
        latency_ms: int,
    ) -> None:
        memory = self._get_route_memory(route_key)
        stats = memory.playwright if backend == "playwright" else memory.curl
        alpha = float(self.config.auto.learning_rate)
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

    async def _run_curl(self, *, url: str, deadline_ts: float) -> CrawlAttempt:
        timeout_s = self._remaining_timeout_s(deadline_ts)
        curl = _require_crawler("curl_cffi", self._curl)
        return cast(
            "CrawlAttempt",
            await cast("Any", curl).crawl_attempt(url=url, timeout_s=timeout_s),
        )

    async def _run_playwright(
        self,
        *,
        url: str,
        deadline_ts: float,
        render_reason: str,
    ) -> CrawlAttempt:
        timeout_s = self._remaining_timeout_s(deadline_ts)
        playwright = _require_crawler("playwright", self._playwright)
        return cast(
            "CrawlAttempt",
            await cast("Any", playwright).crawl_attempt(
                url=url,
                timeout_s=timeout_s,
                render_reason=render_reason,
            ),
        )

    def _resolve_timeout_s(self, timeout_s: float | None) -> float:
        resolved = float(timeout_s or 0.0)
        if resolved <= 0.0:
            resolved = float(self.config.timeout_s)
        return max(0.05, resolved)

    def _remaining_timeout_s(self, deadline_ts: float) -> float:
        remaining = float(deadline_ts - time.monotonic())
        if remaining <= 0.0:
            raise TimeoutError("crawl timeout reached before backend request")
        return remaining

    def _is_blocked(self, res: CrawlAttempt) -> bool:
        status = int(res.status_code or 0)
        if status in _BLOCK_STATUSES:
            return True
        if blocked_marker_hit(
            res.content,
            markers=tuple(self.config.quality.blocked_markers or []),
        ):
            return True
        return bool(res.blocked)

    def _is_useful(self, res: CrawlAttempt) -> bool:
        score, _ = self._content_quality_score(res)
        return score >= float(self.config.quality.quality_score_threshold)

    def _content_quality_score(self, res: CrawlAttempt) -> tuple[float, list[str]]:
        quality = self.config.quality
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
        winner: CrawlAttempt,
        chain_prefix: list[str],
    ) -> CrawlAttempt:
        out: list[str] = []
        for item in chain_prefix + list(winner.attempt_chain or [winner.crawl_backend]):
            token = str(item or "").strip()
            if not token or token in out:
                continue
            out.append(token)
        if not out:
            out = [winner.crawl_backend]
        return winner.model_copy(update={"attempt_chain": out})


__all__ = ["AutoCrawler", "AutoCrawlerConfig"]


def _coerce_rate_limiter(value: object) -> RateLimiterBase[Any]:
    if not isinstance(value, RateLimiterBase):
        raise TypeError("default rate_limit component must implement RateLimiterBase")
    return value


def _coerce_crawler(
    name: str,
    value: object | None,
) -> CrawlerBase[Any] | None:
    if value is None:
        return None
    if not isinstance(value, CrawlerBase):
        raise TypeError(f"crawl component `{name}` must implement CrawlerBase")
    return value


def _require_crawler(
    name: str,
    value: CrawlerBase[Any] | None,
) -> CrawlerBase[Any]:
    if value is None:
        raise RuntimeError(f"crawl component `{name}` is not enabled")
    return value
