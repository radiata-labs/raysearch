from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING
from typing_extensions import override

import anyio

from serpsage.components.fetch.base import FetcherBase
from serpsage.components.fetch.utils import (
    blocked_marker_hit,
    classify_content_kind,
    estimate_text_quality,
)
from serpsage.models.fetch import FetchAttempt, FetchResult

if TYPE_CHECKING:
    from playwright.async_api import (
        Browser,
        BrowserContext,
        Page,
        Playwright,
        Route,
    )

    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase

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
        self._sem = anyio.Semaphore(
            max(1, int(self.settings.fetch.render.js_concurrency))
        )

    @override
    async def on_init(self) -> None:
        if self._browser is not None:
            return
        if _pw_factory is None:
            raise RuntimeError("playwright is not available")
        self._pw = await _pw_factory().start()
        # NOTE: Security features are disabled for web scraping compatibility:
        # - disable-blink-features=AutomationControlled: Hide automation detection
        # - disable-web-security: Allow cross-origin requests for iframe content
        # - disable-features=IsolateOrigins,site-per-process: Enable full page access
        # These are safe in headless crawler context but should not be used for browsing
        self._browser = await self._pw.chromium.launch(
            headless=True,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
                "--disable-setuid-sandbox",
                "--disable-web-security",
                "--disable-features=IsolateOrigins,site-per-process",
            ],
        )
        self._context = await self._browser.new_context(
            user_agent=str(self.settings.fetch.user_agent),
            ignore_https_errors=True,
            locale="en-US",
            timezone_id="America/New_York",
            viewport={"width": 1920, "height": 1080},
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
    async def _afetch_inner(
        self,
        *,
        url: str,
        timeout_s: float | None = None,
    ) -> FetchResult:
        with self.span("fetch.playwright", url=url) as sp:
            attempt = await self.fetch_attempt(
                url=url,
                span=sp,
                timeout_s=timeout_s,
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
            )

    async def fetch_attempt(
        self,
        *,
        url: str,
        span: SpanBase,
        timeout_s: float | None = None,
        render_reason: str | None = None,
    ) -> FetchAttempt:
        if self._browser is None or self._context is None:
            raise RuntimeError("playwright browser is not initialized")

        timeout_ms = int(self.settings.fetch.render.nav_timeout_ms)
        if timeout_s is not None:
            timeout_ms = min(timeout_ms, int(timeout_s * 1000))
        wait_idle_ms = max(0, int(self.settings.fetch.render.wait_network_idle_ms))

        started = time.time()
        page = None
        async with self._sem:
            try:
                page = await self._context.new_page()
                await self._prepare_page(page)
                status, final_url, headers = await self._navigate(
                    page=page,
                    url=url,
                    timeout_ms=timeout_ms,
                    wait_idle_ms=wait_idle_ms,
                    span=span,
                )
                html = await page.content()
            finally:
                if page is not None:
                    await page.close()

        body = (html or "").encode("utf-8", errors="ignore")
        elapsed_ms = int((time.time() - started) * 1000)
        content_type = headers.get("content-type")
        content_kind = classify_content_kind(
            content_type=content_type,
            url=final_url,
            content=body,
        )
        text_chars, content_score, script_ratio = estimate_text_quality(
            body, content_kind=content_kind
        )
        blocked = bool(
            blocked_marker_hit(
                body,
                markers=tuple(self.settings.fetch.quality.blocked_markers),
            )
        )
        span.set_attr("playwright_status", int(status))
        span.set_attr("playwright_elapsed_ms", int(elapsed_ms))
        span.set_attr("content_kind", content_kind)
        span.set_attr("content_score", float(content_score))
        span.set_attr("text_chars", int(text_chars))
        span.set_attr("script_ratio", float(script_ratio))
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
            script_ratio=float(script_ratio),
            blocked=blocked,
            render_reason=render_reason,
            attempt_chain=["playwright"],
        )

    async def _prepare_page(self, page: Page) -> None:
        # Inject script to hide automation fingerprints
        await page.add_init_script(
            """
            Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
            Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
            Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
            Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});
            Object.defineProperty(navigator, 'deviceMemory', {get: () => 8});
            """
        )

        if not bool(self.settings.fetch.render.block_resources):
            return

        async def route_handler(route: Route) -> None:
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
        span: SpanBase,
    ) -> tuple[int, str, dict[str, str]]:
        status = 0
        final_url = url
        headers: dict[str, str] = {}
        resp = None
        navigation_error: str | None = None

        try:
            resp = await page.goto(
                url,
                timeout=timeout_ms,
                wait_until="domcontentloaded",
            )
        except TimeoutError:
            # Page navigation timed out, but some content may have loaded
            # Continue to extract whatever content is available
            navigation_error = "timeout"
            span.set_attr("navigation_timeout", True)
        except Exception as exc:
            # Other navigation errors (network failure, DNS error, etc.)
            # Log the error type and continue with empty response
            navigation_error = type(exc).__name__
            span.set_attr("playwright_error_type", navigation_error)

        # Wait for network idle with extended timeout for complex pages
        if wait_idle_ms > 0:
            with contextlib.suppress(Exception):
                await page.wait_for_load_state("networkidle", timeout=wait_idle_ms)

        # If page didn't load at all, try waiting for body element
        try:
            with contextlib.suppress(TimeoutError):
                await page.wait_for_selector("body", timeout=min(2000, timeout_ms))
        except Exception:
            pass

        final_url = str(page.url or url)

        if resp is None:
            # No response object - either timeout or error
            # Return 200 if we have any content, otherwise 0
            html = await page.content()
            if len(html) > 100:
                status = 200
            else:
                status = 0
                if navigation_error:
                    span.set_attr("navigation_error", navigation_error)
            return status, final_url, headers

        status = int(resp.status or 0)
        headers = {str(k): str(v) for k, v in resp.headers.items()}
        return status, final_url, headers


__all__ = ["PLAYWRIGHT_AVAILABLE", "PlaywrightFetcher"]
