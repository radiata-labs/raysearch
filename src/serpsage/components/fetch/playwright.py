from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import Callable
from typing import TYPE_CHECKING, cast
from typing_extensions import override

import anyio

from serpsage.components.base import ComponentMeta
from serpsage.components.fetch.base import FetchConfigBase, FetcherBase
from serpsage.components.fetch.utils import analyze_content
from serpsage.load import register_component
from serpsage.models.components.fetch import FetchAttempt, FetchResult

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Page, Playwright, Route

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
_STEALTH_INIT_SCRIPT = """
Object.defineProperty(navigator, 'webdriver', {get: () => undefined});
Object.defineProperty(navigator, 'plugins', {get: () => [1, 2, 3, 4, 5]});
Object.defineProperty(navigator, 'languages', {get: () => ['en-US', 'en']});
Object.defineProperty(navigator, 'hardwareConcurrency', {get: () => 8});
Object.defineProperty(navigator, 'deviceMemory', {get: () => 8});
"""


class PlaywrightFetcherConfig(FetchConfigBase):
    pass


_PLAYWRIGHT_FETCHER_META = ComponentMeta(
    family="fetch",
    name="playwright",
    version="1.0.0",
    summary="Playwright browser fetch backend.",
    provides=("fetch.playwright_engine",),
    config_model=PlaywrightFetcherConfig,
    config_optional=True,
)


@register_component(meta=_PLAYWRIGHT_FETCHER_META)
class PlaywrightFetcher(FetcherBase):
    meta = _PLAYWRIGHT_FETCHER_META

    def __init__(self) -> None:
        if not PLAYWRIGHT_AVAILABLE or _pw_factory is None:
            raise RuntimeError("playwright is not available; install playwright")
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        self._sem = anyio.Semaphore(max(1, int(self.config.render.js_concurrency)))

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
            user_agent=str(self.config.user_agent),
            ignore_https_errors=True,
            locale="en-US",
            timezone_id="America/New_York",
            viewport={"width": 1920, "height": 1080},
        )
        await self._prepare_context(self._context)

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
        attempt = await self.fetch_attempt(url=url, timeout_s=timeout_s)
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
        timeout_ms = int(self.config.render.nav_timeout_ms)
        if timeout_s is not None:
            timeout_ms = min(timeout_ms, int(timeout_s * 1000))
        page = None
        status = 0
        final_url = url
        headers: dict[str, str] = {}
        html = ""
        async with self._sem:
            try:
                page = await self._context.new_page()
                status, final_url, headers = await self._navigate(
                    page=page,
                    url=url,
                    timeout_ms=timeout_ms,
                )
                await self._await_render_ready(page=page, timeout_ms=timeout_ms)
                html = await page.content()
            finally:
                if page is not None:
                    with contextlib.suppress(Exception):
                        await page.close()
        body = (html or "").encode("utf-8", errors="ignore")
        content_type = headers.get("content-type")
        analysis = analyze_content(
            content=body,
            content_type=content_type,
            url=final_url,
            markers=tuple(self.config.quality.blocked_markers),
        )
        return FetchAttempt(
            url=final_url,
            status_code=int(status),
            content_type=content_type,
            content=body,
            strategy_used="playwright",
            fetch_mode="playwright",
            rendered=True,
            content_kind=analysis.content_kind,
            headers=headers,
            content_encoding=headers.get("content-encoding"),
            content_length_header=headers.get("content-length"),
            content_score=float(analysis.content_score),
            text_chars=int(analysis.text_chars),
            script_ratio=float(analysis.script_ratio),
            blocked=bool(analysis.blocked),
            render_reason=render_reason,
            attempt_chain=["playwright"],
        )

    async def _prepare_context(self, context: BrowserContext) -> None:
        await context.add_init_script(_STEALTH_INIT_SCRIPT)
        if not bool(self.config.render.block_resources):
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

        await context.route("**/*", route_handler)

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
                url, timeout=timeout_ms, wait_until="domcontentloaded"
            )
        except TimeoutError:
            pass
        except Exception:
            pass
        with contextlib.suppress(Exception):
            await page.wait_for_selector("body", timeout=min(1800, timeout_ms))
        final_url = str(page.url or url)
        if resp is None:
            html = await page.content()
            status = 200 if len(html) > 100 else 0
            return status, final_url, headers
        status = int(resp.status or 0)
        headers = {str(k): str(v) for k, v in resp.headers.items()}
        return status, final_url, headers

    async def _await_render_ready(self, *, page: Page, timeout_ms: int) -> None:
        cfg = self.config.render
        deadline = time.monotonic() + (float(timeout_ms) / 1000.0)
        stable_rounds_needed = max(1, int(cfg.readiness_stable_rounds))
        poll_s = max(0.05, float(cfg.readiness_poll_ms) / 1000.0)
        min_text_chars = int(self.config.quality.min_text_chars)
        stable_rounds = 0
        last_signature: tuple[int, int, str] | None = None
        while time.monotonic() < deadline:
            snapshot = await self._dom_snapshot(page)
            if snapshot is None:
                break
            signature = (
                self._snapshot_int(snapshot, "text_len"),
                self._snapshot_int(snapshot, "node_count"),
                str(snapshot.get("ready_state", "")),
            )
            if last_signature is not None and self._signatures_close(
                last_signature, signature
            ):
                stable_rounds += 1
            else:
                stable_rounds = 0
            last_signature = signature
            if (
                self._snapshot_int(snapshot, "text_len") >= min_text_chars
                and stable_rounds >= stable_rounds_needed
            ):
                break
            if (
                str(snapshot.get("ready_state", "")) == "complete"
                and stable_rounds >= 1
            ):
                break
            await anyio.sleep(poll_s)
        settle_ms = max(0, int(cfg.post_ready_wait_ms))
        if settle_ms > 0:
            await page.wait_for_timeout(min(settle_ms, timeout_ms))

    async def _dom_snapshot(self, page: Page) -> dict[str, object] | None:
        try:
            return cast(
                "dict[str, object] | None",
                await page.evaluate(
                    """
                    () => {
                      const root =
                        document.querySelector('main, article, [role="main"], #content, #main, .main, .article')
                        || document.body;
                      const text = (root?.innerText || document.body?.innerText || '').trim();
                      const nodeCount = root?.querySelectorAll
                        ? root.querySelectorAll('*').length
                        : 0;
                      return {
                        ready_state: document.readyState || '',
                        text_len: text.length,
                        node_count: nodeCount
                      };
                    }
                    """
                ),
            )
        except Exception:
            return None

    def _signatures_close(
        self,
        previous: tuple[int, int, str],
        current: tuple[int, int, str],
    ) -> bool:
        return (
            abs(previous[0] - current[0]) <= 32
            and abs(previous[1] - current[1]) <= 8
            and previous[2] == current[2]
        )

    def _snapshot_int(self, snapshot: dict[str, object], key: str) -> int:
        value = snapshot.get(key, 0)
        if isinstance(value, bool):
            return int(value)
        if isinstance(value, int):
            return value
        if isinstance(value, float):
            return int(value)
        if isinstance(value, str) and value.isdigit():
            return int(value)
        return 0


__all__ = ["PLAYWRIGHT_AVAILABLE", "PlaywrightFetcher", "PlaywrightFetcherConfig"]
