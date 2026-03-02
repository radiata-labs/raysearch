from __future__ import annotations

import asyncio
import contextlib
from collections.abc import Callable
from typing import TYPE_CHECKING, cast
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
# Suppress "Future exception was never retrieved" warnings from Playwright
# when background page loads are interrupted by timeouts or page.close().
_original_call_exception_handler = getattr(
    asyncio.BaseEventLoop, "call_exception_handler", None
)
LoopExceptionHandler = Callable[[asyncio.AbstractEventLoop, dict[str, object]], None]
_original_call_exception_handler = cast(
    "LoopExceptionHandler | None", _original_call_exception_handler
)


def _suppress_future_exception_warning(
    self: asyncio.AbstractEventLoop,
    context: dict[str, object],
) -> None:
    """Custom exception handler to suppress Playwright future warnings."""
    exc = context.get("exception")
    if exc is not None:
        exc_name = type(exc).__name__
        if exc_name in {"TimeoutError", "TargetClosedError", "Error"}:
            return
    if _original_call_exception_handler:
        _original_call_exception_handler(self, context)


asyncio.BaseEventLoop.call_exception_handler = _suppress_future_exception_warning  # type: ignore[method-assign]
PLAYWRIGHT_AVAILABLE = False
_pw_factory = None
try:
    from playwright.async_api import async_playwright

    PLAYWRIGHT_AVAILABLE = True
    _pw_factory = async_playwright
except Exception:  # noqa: BLE001
    PLAYWRIGHT_AVAILABLE = False
_BLOCK_RESOURCE_TYPES = {"image", "media", "font", "video", "websocket"}
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
        attempt = await self.fetch_attempt(
            url=url,
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
        timeout_s: float | None = None,
        render_reason: str | None = None,
    ) -> FetchAttempt:
        if self._browser is None or self._context is None:
            raise RuntimeError("playwright browser is not initialized")
        timeout_ms = int(self.settings.fetch.render.nav_timeout_ms)
        if timeout_s is not None:
            timeout_ms = min(timeout_ms, int(timeout_s * 1000))
        page = None
        async with self._sem:
            try:
                page = await self._context.new_page()
                await self._prepare_page(page)
                status, final_url, headers = await self._navigate(
                    page=page,
                    url=url,
                    timeout_ms=timeout_ms,
                )
                html = await page.content()
                with contextlib.suppress(TimeoutError):
                    await page.wait_for_load_state(
                        "networkidle",
                        timeout=timeout_ms,
                    )
                await page.wait_for_timeout(min(1200, timeout_ms))
                html = await page.content()
            finally:
                if page is not None:
                    with contextlib.suppress(Exception):
                        await page.close()
        body = (html or "").encode("utf-8", errors="ignore")
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
    ) -> tuple[int, str, dict[str, str]]:
        status = 0
        final_url = url
        headers: dict[str, str] = {}
        resp = None
        try:
            resp = await page.goto(
                url,
                timeout=timeout_ms,
                wait_until="domcontentloaded",
            )
        except TimeoutError:
            pass
        except Exception:
            pass
        try:
            with contextlib.suppress(TimeoutError):
                await page.wait_for_selector("body", timeout=min(2000, timeout_ms))
        except Exception:
            pass
        final_url = str(page.url or url)
        if resp is None:
            html = await page.content()
            status = 200 if len(html) > 100 else 0
            return status, final_url, headers
        status = int(resp.status or 0)
        headers = {str(k): str(v) for k, v in resp.headers.items()}
        return status, final_url, headers


__all__ = ["PLAYWRIGHT_AVAILABLE", "PlaywrightFetcher"]
