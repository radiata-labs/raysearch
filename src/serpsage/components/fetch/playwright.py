from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing_extensions import override

import anyio

from serpsage.components.fetch.utils import (
    classify_content_kind,
    estimate_text_quality,
)
from serpsage.contracts.services import FetcherBase
from serpsage.models.fetch import FetchAttempt, FetchResult

if TYPE_CHECKING:
    from playwright.async_api import Browser, BrowserContext, Playwright

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


class PlaywrightFetcher(FetcherBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)
        if not PLAYWRIGHT_AVAILABLE or _pw_factory is None:
            raise RuntimeError("playwright is not available; install playwright")
        self._pw: Playwright | None = None
        self._browser: Browser | None = None
        self._context: BrowserContext | None = None
        js_cfg = self.settings.enrich.fetch.playwright
        self._sem = anyio.Semaphore(max(1, int(js_cfg.js_concurrency)))

    @override
    async def on_init(self) -> None:
        if self._browser is not None:
            return
        if _pw_factory is None:
            raise RuntimeError("playwright is not available")
        self._pw = await _pw_factory().start()
        self._browser = await self._pw.chromium.launch(
            headless=bool(self.settings.enrich.fetch.playwright.headless)
        )
        self._context = await self._browser.new_context(
            user_agent=str(self.settings.enrich.fetch.user_agent),
            ignore_https_errors=True,
        )

    @override
    async def on_close(self) -> None:
        if self._context is not None:
            try:
                await self._context.close()
            except Exception:
                pass
            self._context = None
        if self._browser is not None:
            try:
                await self._browser.close()
            except Exception:
                pass
            self._browser = None
        if self._pw is not None:
            try:
                await self._pw.stop()
            except Exception:
                pass
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
        _ = allow_render, depth, rank_index
        with self.span("fetch.playwright", url=url) as sp:
            attempt = await self.fetch_attempt(url=url, span=sp, timeout_s=timeout_s)
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
            )

    async def fetch_attempt(
        self,
        *,
        url: str,
        span: SpanBase,
        timeout_s: float | None = None,
        render_reason: str | None = None,
    ) -> FetchAttempt:
        if self._browser is None:
            await self.ainit()
        if self._browser is None or self._context is None:
            raise RuntimeError("playwright browser is not initialized")

        cfg = self.settings.enrich.fetch.playwright
        timeout_ms = max(300, int(cfg.nav_timeout_ms))
        if timeout_s is not None:
            timeout_ms = max(300, min(timeout_ms, int(timeout_s * 1000)))
        wait_idle_ms = max(0, int(cfg.wait_for_network_idle_ms))

        started = time.time()
        async with self._sem:
            page = await self._context.new_page()

            if bool(cfg.block_resources):
                blocked_types = {"image", "media", "font"}

                async def route_handler(route) -> None:  # noqa: ANN001
                    req = route.request
                    if req.resource_type in blocked_types:
                        await route.abort()
                        return
                    await route.continue_()

                await page.route("**/*", route_handler)

            status = 0
            final_url = url
            content_type: str | None = None
            body = b""
            headers: dict[str, str] = {}

            try:
                resp = await page.goto(
                    url,
                    timeout=timeout_ms,
                    wait_until="domcontentloaded",
                )
                if wait_idle_ms > 0:
                    try:
                        await page.wait_for_load_state(
                            "networkidle",
                            timeout=wait_idle_ms,
                        )
                    except Exception:
                        pass
                html = await page.content()
                body = (html or "").encode("utf-8", errors="ignore")
                final_url = str(page.url or url)
                if resp is not None:
                    status = int(resp.status or 0)
                    headers = {str(k): str(v) for k, v in resp.headers.items()}
                    content_type = headers.get("content-type")
                else:
                    status = 200 if body else 0
            finally:
                await page.close()

        elapsed_ms = int((time.time() - started) * 1000)
        content_kind = classify_content_kind(
            content_type=content_type,
            url=final_url,
            content=body,
        )
        text_chars, content_score, _ = estimate_text_quality(body, content_kind=content_kind)
        span.set_attr("playwright_status", int(status))
        span.set_attr("playwright_elapsed_ms", int(elapsed_ms))
        span.set_attr("content_kind", content_kind)
        span.set_attr("content_score", float(content_score))
        span.set_attr("text_chars", int(text_chars))

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
            blocked=False,
            render_reason=render_reason,
        )


__all__ = ["PLAYWRIGHT_AVAILABLE", "PlaywrightFetcher"]
