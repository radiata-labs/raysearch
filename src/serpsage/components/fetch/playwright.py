from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING
from typing_extensions import override

import anyio

from serpsage.components.fetch.utils import (
    blocked_marker_hit,
    classify_content_kind,
    estimate_text_quality,
)
from serpsage.contracts.services import FetcherBase
from serpsage.core.tuning import (
    DEFAULT_FETCH_USER_AGENT,
    PLAYWRIGHT_BLOCK_RESOURCES,
    PLAYWRIGHT_HEADLESS,
    PLAYWRIGHT_JS_CONCURRENCY,
    PLAYWRIGHT_NAV_TIMEOUT_MS,
    PLAYWRIGHT_WAIT_NETWORK_IDLE_MS,
)
from serpsage.models.fetch import FetchAttempt, FetchResult

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page, Playwright

    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime

PLAYWRIGHT_AVAILABLE = False
_pw_factory = None
try:
    from playwright.async_api import async_playwright

    PLAYWRIGHT_AVAILABLE = True
    _pw_factory = async_playwright
except Exception:  # noqa: BLE001
    PLAYWRIGHT_AVAILABLE = False

_BLOCK_RESOURCE_TYPES = {"image", "media", "font"}
_BLOCK_TRACKING_RE = (
    "googletagmanager",
    "google-analytics",
    "doubleclick",
    "hotjar",
    "mixpanel",
    "segment",
    "clarity.ms",
)


class PlaywrightFetcher(FetcherBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)
        if not PLAYWRIGHT_AVAILABLE or _pw_factory is None:
            raise RuntimeError("playwright is not available; install playwright")
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._sem = anyio.Semaphore(max(1, int(PLAYWRIGHT_JS_CONCURRENCY)))

    @override
    async def on_init(self) -> None:
        if self._browser is not None:
            return
        if _pw_factory is None:
            raise RuntimeError("playwright is not available")
        self._pw = await _pw_factory().start()
        self._browser = await self._pw.chromium.launch(
            headless=bool(PLAYWRIGHT_HEADLESS)
        )
        self._context = await self._browser.new_context(
            user_agent=str(DEFAULT_FETCH_USER_AGENT),
            ignore_https_errors=True,
            locale="en-US",
        )

    @override
    async def on_close(self) -> None:
        if self._context is not None:
            with contextlib.suppress(Exception):
                await self._context.close()
            self._context = None
        if self._browser is not None:
            with contextlib.suppress(Exception):
                await self._browser.close()
            self._browser = None
        if self._pw is not None:
            with contextlib.suppress(Exception):
                await self._pw.stop()
            self._pw = None

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
        with self.span("fetch.playwright", url=url) as sp:
            attempt = await self.fetch_attempt(
                url=url,
                span=sp,
                timeout_s=timeout_s,
                depth=depth,
            )
            if int(attempt.status_code or 0) <= 0:
                raise RuntimeError("playwright fetch failed")
            return FetchResult(
                url=attempt.url,
                status_code=int(attempt.status_code),
                content_type=attempt.content_type,
                content=attempt.content,
                fetch_mode="playwright",
                rendered=True,
                content_kind=attempt.content_kind,
                headers=dict(attempt.headers or {}),
                attempt_chain=list(attempt.attempt_chain or []),
                quality_score=float(attempt.quality_score or attempt.content_score),
            )

    async def fetch_attempt(
        self,
        *,
        url: str,
        span: SpanBase,
        timeout_s: float | None = None,
        render_reason: str | None = None,
        depth: str | None = None,
    ) -> FetchAttempt:
        if self._browser is None or self._context is None:
            await self.ainit()
        if self._browser is None or self._context is None:
            raise RuntimeError("playwright browser is not initialized")

        timeout_ms = int(PLAYWRIGHT_NAV_TIMEOUT_MS)
        if timeout_s is not None:
            timeout_ms = max(350, min(timeout_ms, int(timeout_s * 1000)))
        wait_idle_ms = max(0, int(PLAYWRIGHT_WAIT_NETWORK_IDLE_MS))

        started = time.time()
        async with self._sem:
            page = await self._context.new_page()
            try:
                await self._prepare_page(page)
                status, final_url, headers = await self._navigate(
                    page=page,
                    url=url,
                    timeout_ms=timeout_ms,
                    wait_idle_ms=wait_idle_ms,
                )
                html = await page.content()
            finally:
                await page.close()

        body = (html or "").encode("utf-8", errors="ignore")
        elapsed_ms = int((time.time() - started) * 1000)
        content_type = headers.get("content-type")
        content_kind = classify_content_kind(
            content_type=content_type,
            url=final_url,
            content=body,
        )
        text_chars, content_score, _ = estimate_text_quality(
            body, content_kind=content_kind
        )
        blocked = bool(blocked_marker_hit(body))
        quality_score = float(content_score - (0.3 if blocked else 0.0))

        span.set_attr("playwright_status", int(status))
        span.set_attr("playwright_elapsed_ms", int(elapsed_ms))
        span.set_attr("content_kind", content_kind)
        span.set_attr("content_score", float(content_score))
        span.set_attr("text_chars", int(text_chars))
        if render_reason:
            span.set_attr("render_reason", str(render_reason))

        return FetchAttempt(
            url=final_url,
            status_code=int(status),
            content_type=content_type,
            content=body,
            strategy_used="playwright",
            fetch_mode="playwright",
            rendered=True,
            content_kind=content_kind,
            headers=headers,
            content_encoding=headers.get("content-encoding"),
            content_length_header=headers.get("content-length"),
            content_score=float(content_score),
            text_chars=int(text_chars),
            blocked=blocked,
            render_reason=render_reason,
            attempt_chain=["playwright"],
            quality_score=float(quality_score),
        )

    async def _prepare_page(self, page: Page) -> None:
        if not bool(PLAYWRIGHT_BLOCK_RESOURCES):
            return

        async def route_handler(route) -> None:  # noqa: ANN001
            req = route.request
            resource_type = str(req.resource_type or "").lower()
            req_url = str(req.url or "").lower()
            if resource_type in _BLOCK_RESOURCE_TYPES:
                await route.abort()
                return
            if any(key in req_url for key in _BLOCK_TRACKING_RE):
                await route.abort()
                return
            await route.continue_()

        await page.route("**/*", route_handler)

    async def _navigate(
        self,
        *,
        page: Page,
        url: str,
        timeout_ms: int,
        wait_idle_ms: int,
    ) -> tuple[int, str, dict[str, str]]:
        status = 0
        final_url = url
        headers: dict[str, str] = {}
        resp = await page.goto(
            url,
            timeout=timeout_ms,
            wait_until="domcontentloaded",
        )
        if wait_idle_ms > 0:
            with contextlib.suppress(Exception):
                await page.wait_for_load_state("networkidle", timeout=wait_idle_ms)
        final_url = str(page.url or url)
        if resp is None:
            status = 200
            return status, final_url, headers
        status = int(resp.status or 0)
        headers = {str(k): str(v) for k, v in resp.headers.items()}
        return status, final_url, headers


__all__ = ["PLAYWRIGHT_AVAILABLE", "PlaywrightFetcher"]
