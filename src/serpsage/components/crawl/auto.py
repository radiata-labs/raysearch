"""Auto-selecting crawler that dispatches to specialized crawlers.

Resolution order:
1. Try each SpecializedCrawlerBase's can_handle()
2. Use the first one that returns True
3. Fall back to curl_cffi/playwright for normal web crawling

Configuration
=============

Specialized crawlers are discovered automatically from the component registry.
To add a new specialized crawler:

1. Create a class inheriting from SpecializedCrawlerBase
2. Implement can_handle() classmethod
3. Register it in the component configuration

Example configuration in this project:

.. code:: yaml

   crawl:
     auto:
       enabled: true
     doi:
       enabled: true
     reddit:
       enabled: true
"""

from __future__ import annotations

import re
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any
from typing_extensions import override
from urllib.parse import urlparse

from pydantic import Field, field_validator, model_validator

from serpsage.components.crawl.base import (
    CrawlerBase,
    CrawlerConfigBase,
    SpecializedCrawlerBase,
)
from serpsage.components.crawl.curl_cffi import CurlCffiCrawler
from serpsage.components.crawl.playwright import PlaywrightCrawler
from serpsage.components.crawl.utils import (
    has_nextjs_signals,
    has_spa_signals,
    normalize_route_key,
)
from serpsage.components.loads import ComponentRegistry
from serpsage.components.rate_limit.base import RateLimiterBase
from serpsage.dependencies import CACHE_TOKEN, Depends, solve_dependencies
from serpsage.models.components.crawl import CrawlAttempt, CrawlResult

BLOCK_STATUSES = {401, 403, 429}
MIN_CONTENT_BYTES = 32
MIN_SUCCESS_PROB = 0.15
LEARNING_RATE = 0.22
ROUTE_CACHE_SIZE = 4096
ROUTE_PLAYWRIGHT_MIN_SAMPLES = 3
ROUTE_PLAYWRIGHT_SCORE_RATIO = 0.78
ROUTE_PLAYWRIGHT_MIN_USEFUL = 0.72

DirectPlaywrightRule = str | Callable[[str], bool]
DIRECT_PLAYWRIGHT_RULES: frozenset[DirectPlaywrightRule] = frozenset({"github.com"})
RULE_REASON_RE = re.compile(r"[^a-z0-9]+")


@dataclass(slots=True)
class RouteStats:
    curl_samples: int = 0
    curl_success_ema: float = 0.60
    curl_useful_ema: float = 0.60
    curl_latency_ms: float = 700.0
    playwright_samples: int = 0
    playwright_success_ema: float = 0.72
    playwright_useful_ema: float = 0.72
    playwright_latency_ms: float = 1700.0
    last_used_ts: float = 0.0

    def record(
        self, *, backend: str, success: bool, useful: bool, latency_ms: int
    ) -> None:
        if backend == "playwright":
            self.playwright_samples += 1
            self.playwright_success_ema = self._ema(
                current=self.playwright_success_ema,
                observed=1.0 if success else 0.0,
            )
            self.playwright_useful_ema = self._ema(
                current=self.playwright_useful_ema,
                observed=1.0 if useful else 0.0,
            )
            self.playwright_latency_ms = self._ema(
                current=self.playwright_latency_ms,
                observed=max(1, latency_ms),
            )
            return
        self.curl_samples += 1
        self.curl_success_ema = self._ema(
            current=self.curl_success_ema,
            observed=1.0 if success else 0.0,
        )
        self.curl_useful_ema = self._ema(
            current=self.curl_useful_ema,
            observed=1.0 if useful else 0.0,
        )
        self.curl_latency_ms = self._ema(
            current=self.curl_latency_ms,
            observed=max(1, latency_ms),
        )

    def score(self, backend: str) -> float:
        if backend == "playwright":
            success = self.playwright_success_ema
            useful = self.playwright_useful_ema
            latency_ms = self.playwright_latency_ms
        else:
            success = self.curl_success_ema
            useful = self.curl_useful_ema
            latency_ms = self.curl_latency_ms
        probability = max(MIN_SUCCESS_PROB, success * 0.45 + useful * 0.55)
        return latency_ms / probability

    @staticmethod
    def _ema(*, current: float, observed: float) -> float:
        return ((1.0 - LEARNING_RATE) * current) + (LEARNING_RATE * observed)


class AutoCrawlerConfig(CrawlerConfigBase):
    __setting_family__ = "crawl"
    __setting_name__ = "auto"

    scout_bytes: int = 48_000
    min_text_chars: int = 100
    max_script_ratio: float = 0.35
    min_useful_score: float = 0.15
    playwright_rules: set[str] = Field(default_factory=set)

    @field_validator("scout_bytes", "min_text_chars")
    @classmethod
    def _validate_positive_int(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("auto crawler integer settings must be > 0")
        return value

    @field_validator("playwright_rules", mode="before")
    @classmethod
    def _normalize_playwright_rules(cls, value: object) -> set[str]:
        if value is None:
            return set()
        if isinstance(value, str):
            return {value.strip().lower()} if value.strip() else set()
        if isinstance(value, (list, set, tuple)):
            return {str(item).strip().lower() for item in value if str(item).strip()}
        text = str(value).strip().lower()
        return {text} if text else set()

    @model_validator(mode="after")
    def _validate_thresholds(self) -> AutoCrawlerConfig:
        if not 0.0 <= self.max_script_ratio <= 1.0:
            raise ValueError("max_script_ratio must be between 0 and 1")
        if not 0.0 <= self.min_useful_score <= 1.0:
            raise ValueError("min_useful_score must be between 0 and 1")
        return self


async def specialized_crawlers_factory(
    cache: dict[Any, Any] = Depends(CACHE_TOKEN),
    registry: ComponentRegistry = Depends(),
) -> tuple[SpecializedCrawlerBase, ...]:
    """Factory function: collect all enabled specialized crawlers."""
    crawlers: list[SpecializedCrawlerBase] = []
    for spec in registry.enabled_specs("crawl"):
        # Skip auto, curl_cffi, playwright (they are not specialized)
        if spec.name in ("auto", "curl_cffi", "playwright"):
            continue
        if not issubclass(spec.cls, SpecializedCrawlerBase):
            continue
        instance = await solve_dependencies(spec.cls, dependency_cache=cache)
        if isinstance(instance, SpecializedCrawlerBase):
            crawlers.append(instance)
    return tuple(crawlers)


class AutoCrawler(CrawlerBase[AutoCrawlerConfig]):
    rate_limiter: RateLimiterBase[Any] = Depends()
    curl: CurlCffiCrawler = Depends()
    playwright: PlaywrightCrawler = Depends()

    route_stats: OrderedDict[str, RouteStats]

    def __init__(
        self,
        *,
        specialized: tuple[SpecializedCrawlerBase, ...] = Depends(
            specialized_crawlers_factory
        ),
    ) -> None:
        super().__init__()
        self.route_stats = OrderedDict()
        self.specialized = specialized

    @override
    async def _acrawl(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
    ) -> CrawlResult:
        host = urlparse(url).netloc.lower()

        # Try specialized crawlers first (they handle their own rate limiting)
        for crawler in self.specialized:
            if type(crawler).can_handle(url=url):
                try:
                    result = await crawler._acrawl(url=url, timeout_s=timeout_s)
                    # Only accept successful results
                    if result.status_code == 200 and result.content:
                        return result
                except Exception:  # noqa: BLE001
                    pass  # Fall through to normal flow

        await self.rate_limiter.acquire(host=host)
        try:
            attempt = await self._crawl_attempt(url=url, timeout_s=timeout_s)
        finally:
            await self.rate_limiter.release(host=host)
        return CrawlResult(
            url=attempt.url,
            status_code=attempt.status_code,
            content_type=attempt.content_type,
            content=attempt.content,
            crawl_backend=attempt.crawl_backend,
            rendered=attempt.rendered,
            content_kind=attempt.content_kind,
            headers=dict(attempt.headers or {}),
            attempt_chain=list(attempt.attempt_chain or []),
        )

    # Main flow

    async def _crawl_attempt(
        self,
        *,
        url: str,
        timeout_s: float | None,
    ) -> CrawlAttempt:
        deadline_ts = time.monotonic() + self._resolve_timeout_s(timeout_s)
        route_key = normalize_route_key(url)
        route = self._get_route_stats(route_key)

        direct_reason = self._direct_playwright_reason(url)
        if direct_reason:
            attempt = await self._try_playwright(
                url=url,
                deadline_ts=deadline_ts,
                route_key=route_key,
                chain_prefix=[f"decision:playwright:{direct_reason}"],
            )
            if attempt is not None:
                return attempt

        if self._route_prefers_playwright(route):
            attempt = await self._try_playwright(
                url=url,
                deadline_ts=deadline_ts,
                route_key=route_key,
                chain_prefix=["decision:playwright:route_memory"],
            )
            if attempt is not None:
                return attempt

        curl_attempt, curl_finished = await self._run_curl_scout(
            url=url,
            deadline_ts=deadline_ts,
            route_key=route_key,
            route=route,
        )
        if self._should_accept_curl(curl_attempt, finished=curl_finished):
            label = "scout" if curl_finished else "truncated"
            return self._with_attempt_chain(
                curl_attempt,
                [f"decision:curl_cffi:{label}"],
            )

        fallback_reason = self._fallback_playwright_reason(curl_attempt)
        recovered = await self._try_playwright(
            url=url,
            deadline_ts=deadline_ts,
            route_key=route_key,
            chain_prefix=[
                "decision:curl_cffi:scout",
                f"fallback:playwright:{fallback_reason}",
            ],
        )
        if recovered is not None:
            return recovered

        if self._route_prefers_playwright(route) and self._is_useful(curl_attempt):
            return self._with_attempt_chain(
                curl_attempt,
                ["decision:playwright:route_memory", "fallback:curl_cffi"],
            )

        raise RuntimeError(f"crawl_unusable:auto:playwright:{curl_attempt.status_code}")

    # Direct URL rules

    @classmethod
    def _reason_for_rule(cls, rule: DirectPlaywrightRule) -> str:
        if callable(rule):
            name = rule.__name__.removeprefix("_is_").removesuffix("_url")
        else:
            name = rule
        normalized = RULE_REASON_RE.sub("_", name.lower()).strip("_")
        return normalized or "custom"

    def _direct_playwright_reason(self, url: str) -> str:
        url_lower = url.lower()
        for rule in DIRECT_PLAYWRIGHT_RULES:
            if isinstance(rule, str) and rule in url_lower:
                return self._reason_for_rule(rule)
        for rule in DIRECT_PLAYWRIGHT_RULES:
            if callable(rule) and rule(url_lower):
                return self._reason_for_rule(rule)
        for rule in self.config.playwright_rules:
            if rule in url_lower:
                return self._reason_for_rule(rule)
        return ""

    # Backend runners

    async def _try_playwright(
        self,
        *,
        url: str,
        deadline_ts: float,
        route_key: str,
        chain_prefix: list[str],
    ) -> CrawlAttempt | None:
        started = time.monotonic()
        attempt: CrawlAttempt | None = None
        try:
            attempt = await self.playwright.crawl_attempt(
                url=url,
                timeout_s=self._remaining_timeout_s(deadline_ts),
            )
        except Exception:
            pass
        finally:
            self._record_route_attempt(
                route_key=route_key,
                backend="playwright",
                attempt=attempt,
                latency_ms=int((time.monotonic() - started) * 1000),
            )
        if attempt is None or not self._is_useful(attempt):
            return None
        return self._with_attempt_chain(attempt, chain_prefix)

    async def _run_curl_scout(
        self,
        *,
        url: str,
        deadline_ts: float,
        route_key: str,
        route: RouteStats,
    ) -> tuple[CrawlAttempt, bool]:
        started = time.monotonic()
        progressive = await self.curl.crawl_progressive_attempt(
            url=url,
            timeout_s=self._remaining_timeout_s(deadline_ts),
            scout_bytes=self.config.scout_bytes,
            continue_predicate=lambda attempt: self._should_continue_curl(
                attempt=attempt,
                route=route,
            ),
        )
        self._record_route_attempt(
            route_key=route_key,
            backend="curl_cffi",
            attempt=progressive.attempt,
            latency_ms=int((time.monotonic() - started) * 1000),
        )
        return progressive.attempt, progressive.finished

    # Route learning

    def _get_route_stats(self, route_key: str) -> RouteStats:
        route = self.route_stats.pop(route_key, None)
        if route is None:
            route = RouteStats()
        route.last_used_ts = time.monotonic()
        self.route_stats[route_key] = route
        while len(self.route_stats) > max(8, ROUTE_CACHE_SIZE):
            self.route_stats.popitem(last=False)
        return route

    def _record_route_attempt(
        self,
        *,
        route_key: str,
        backend: str,
        attempt: CrawlAttempt | None,
        latency_ms: int,
    ) -> None:
        route = self._get_route_stats(route_key)
        route.record(
            backend=backend,
            success=attempt is not None and attempt.status_code >= 200,
            useful=attempt is not None and self._is_useful(attempt),
            latency_ms=latency_ms,
        )

    def _route_prefers_playwright(self, route: RouteStats) -> bool:
        if route.playwright_samples < ROUTE_PLAYWRIGHT_MIN_SAMPLES:
            return False
        if route.playwright_useful_ema < ROUTE_PLAYWRIGHT_MIN_USEFUL:
            return False
        return route.score("playwright") <= (
            route.score("curl_cffi") * ROUTE_PLAYWRIGHT_SCORE_RATIO
        )

    # Curl heuristics

    def _should_continue_curl(
        self,
        *,
        attempt: CrawlAttempt,
        route: RouteStats,
    ) -> bool:
        if self._is_blocked(attempt):
            return False
        if attempt.content_kind != "html":
            return True

        content = bytes(attempt.content or b"")
        low_text = attempt.text_chars < self.config.min_text_chars
        js_heavy = attempt.script_ratio >= self.config.max_script_ratio
        looks_like_spa = has_spa_signals(content)
        looks_like_nextjs = has_nextjs_signals(content)
        quality_score = self._quality_score(attempt)

        if looks_like_spa and (low_text or js_heavy):
            return False
        if looks_like_nextjs and low_text:
            return False
        if js_heavy and quality_score < self.config.min_useful_score:
            return False
        return not (self._route_prefers_playwright(route) and quality_score < 0.55)

    def _should_accept_curl(self, attempt: CrawlAttempt, *, finished: bool) -> bool:
        if finished:
            return self._is_useful(attempt)
        if attempt.content_kind in {"pdf", "text", "markdown"}:
            return self._is_useful(attempt)
        if attempt.content_kind != "html":
            return False
        return attempt.text_chars >= (self.config.min_text_chars * 6)

    def _fallback_playwright_reason(self, attempt: CrawlAttempt) -> str:
        if self._is_blocked(attempt):
            return "blocked"
        if attempt.content_kind == "html":
            content = bytes(attempt.content or b"")
            if has_spa_signals(content):
                return "spa"
            if has_nextjs_signals(content):
                return "nextjs"
            if attempt.text_chars < self.config.min_text_chars:
                return "low_text"
        return "curl_unusable"

    # Quality checks

    def _is_blocked(self, attempt: CrawlAttempt) -> bool:
        return attempt.status_code in BLOCK_STATUSES or attempt.blocked

    def _is_useful(self, attempt: CrawlAttempt) -> bool:
        return self._quality_score(attempt) >= self.config.min_useful_score

    def _quality_score(self, attempt: CrawlAttempt) -> float:
        status = attempt.status_code
        if status < 200 or status >= 400 or self._is_blocked(attempt):
            return 0.0

        content_len = len(attempt.content or b"")
        score = (
            min(
                1.0,
                max(0, content_len - MIN_CONTENT_BYTES) / (1200 - MIN_CONTENT_BYTES),
            )
            * 0.22
        )

        if attempt.content_kind == "html":
            text_score = min(
                1.0,
                max(0, attempt.text_chars - self.config.min_text_chars)
                / (1800 - self.config.min_text_chars),
            )
            score += text_score * 0.48
        else:
            score += attempt.content_score * 0.48

        if attempt.content_kind not in {"binary", "unknown"}:
            score += 0.15

        score += max(0.0, 1.0 - attempt.script_ratio * 2.0) * 0.15
        return score

    # Shared helpers

    def _with_attempt_chain(
        self,
        attempt: CrawlAttempt,
        chain_prefix: list[str],
    ) -> CrawlAttempt:
        chain: list[str] = []
        for item in chain_prefix + list(
            attempt.attempt_chain or [attempt.crawl_backend]
        ):
            token = str(item).strip()
            if token and token not in chain:
                chain.append(token)
        if not chain:
            chain = [attempt.crawl_backend]
        return attempt.model_copy(update={"attempt_chain": chain})

    def _resolve_timeout_s(self, timeout_s: float | None) -> float:
        return max(
            0.05, timeout_s if timeout_s and timeout_s > 0 else self.config.timeout_s
        )

    def _remaining_timeout_s(self, deadline_ts: float) -> float:
        remaining = deadline_ts - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("crawl timeout reached before backend request")
        return remaining


__all__ = [
    "AutoCrawler",
    "AutoCrawlerConfig",
    "specialized_crawlers_factory",
]
