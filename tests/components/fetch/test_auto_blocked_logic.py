from __future__ import annotations

import time

from serpsage.components.fetch.auto import AutoFetcher
from serpsage.contracts.lifecycle import ClockBase
from serpsage.contracts.services import CacheBase, RateLimiterBase
from serpsage.core.runtime import Runtime
from serpsage.core.workunit import WorkUnit
from serpsage.models.fetch import FetchAttempt
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class _Clock(ClockBase):
    def now_ms(self) -> int:
        return int(time.time() * 1000)


class _NoopCache(CacheBase):
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        _ = namespace, key
        return None

    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        _ = namespace, key, value, ttl_s
        return


class _NoopRateLimiter(RateLimiterBase):
    async def acquire(self, *, host: str) -> None:
        _ = host
        return

    async def release(self, *, host: str) -> None:
        _ = host
        return


class _NoopHttpxFetcher(WorkUnit):
    async def fetch_attempt(self, **kwargs: object) -> FetchAttempt:
        _ = kwargs
        raise RuntimeError("not used in this test")


def _build_auto_fetcher(settings: AppSettings) -> AutoFetcher:
    settings.fetch.backend = "auto"
    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_Clock())
    return AutoFetcher(
        rt=rt,
        cache=_NoopCache(rt=rt),
        rate_limiter=_NoopRateLimiter(rt=rt),
        httpx_fetcher=_NoopHttpxFetcher(rt=rt),  # type: ignore[arg-type]
        curl_fetcher=None,
        playwright_fetcher=None,
    )


def _make_attempt(
    *,
    content: bytes,
    text_chars: int,
    content_score: float,
    blocked: bool,
) -> FetchAttempt:
    return FetchAttempt(
        url="https://example.com/page",
        status_code=200,
        content_type="text/html",
        content=content,
        fetch_mode="curl_cffi",
        strategy_used="curl_cffi",
        rendered=False,
        content_kind="html",
        headers={},
        content_encoding=None,
        content_length_header=None,
        content_score=content_score,
        text_chars=text_chars,
        blocked=blocked,
        render_reason=None,
        attempt_chain=["curl_cffi"],
        quality_score=content_score,
    )


def test_is_blocked_does_not_reject_high_quality_html_with_script_marker() -> None:
    settings = AppSettings()
    settings.fetch.quality.blocked_markers = [
        "captcha",
        "verify you are human",
        "access denied",
    ]
    fetcher = _build_auto_fetcher(settings)

    html = """
    <html><head>
      <script>window.config = {"captcha":"hcaptcha"};</script>
    </head>
    <body>
      <article>
        <h1>Large language model</h1>
        <p>This article contains plenty of visible text for quality scoring.</p>
      </article>
    </body></html>
    """.encode("utf-8")
    attempt = _make_attempt(
        content=html,
        text_chars=1400,
        content_score=0.95,
        blocked=True,
    )

    assert fetcher._is_blocked(attempt) is False
    assert fetcher._is_useful(attempt) is True


def test_is_blocked_keeps_low_quality_challenge_page_blocked() -> None:
    settings = AppSettings()
    settings.fetch.quality.blocked_markers = [
        "verify you are human",
        "checking your browser",
    ]
    fetcher = _build_auto_fetcher(settings)

    challenge_html = """
    <html><body>
      <h1>Security Check</h1>
      <p>Checking your browser before accessing this page.</p>
      <p>Please verify you are human to continue.</p>
    </body></html>
    """.encode("utf-8")
    attempt = _make_attempt(
        content=challenge_html,
        text_chars=80,
        content_score=0.12,
        blocked=True,
    )

    assert fetcher._is_blocked(attempt) is True
    assert fetcher._is_useful(attempt) is False
