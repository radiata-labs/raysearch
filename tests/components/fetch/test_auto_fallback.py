from __future__ import annotations

import time

import anyio
import httpx

from serpsage.components.fetch.auto import AutoFetcher
from serpsage.contracts.lifecycle import ClockBase
from serpsage.contracts.services import CacheBase, RateLimiterBase
from serpsage.core.runtime import Runtime
from serpsage.core.workunit import WorkUnit
from serpsage.models.fetch import FetchAttempt, FetchResult
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry, TraceTelemetry

URL = "https://en.wikipedia.org/wiki/Large_language_model"


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


class _StubHttpxFetcher(WorkUnit):
    def __init__(
        self,
        *,
        rt: Runtime,
        browser_outcome: FetchAttempt | Exception,
        compat_outcome: FetchAttempt | Exception,
    ) -> None:
        super().__init__(rt=rt)
        self._browser = browser_outcome
        self._compat = compat_outcome

    async def fetch_attempt(
        self,
        *,
        url: str,
        profile: str,
        **kwargs: object,
    ) -> FetchAttempt:
        _ = url, kwargs
        outcome = self._browser if profile == "browser" else self._compat
        if isinstance(outcome, Exception):
            raise outcome
        return outcome


class _StubCurlFetcher(WorkUnit):
    def __init__(self, *, rt: Runtime, attempt: FetchAttempt) -> None:
        super().__init__(rt=rt)
        self._attempt = attempt

    async def fetch_attempt(self, **kwargs: object) -> FetchAttempt:
        _ = kwargs
        return self._attempt


def _attempt(
    *,
    fetch_mode: str,
    status_code: int,
    content: bytes,
    content_kind: str,
    text_chars: int,
    content_score: float,
    blocked: bool,
) -> FetchAttempt:
    return FetchAttempt(
        url=URL,
        status_code=status_code,
        content_type="text/html",
        content=content,
        fetch_mode=fetch_mode,  # type: ignore[arg-type]
        strategy_used=fetch_mode,  # type: ignore[arg-type]
        rendered=False,
        content_kind=content_kind,  # type: ignore[arg-type]
        headers={},
        content_encoding=None,
        content_length_header=None,
        content_score=content_score,
        text_chars=text_chars,
        blocked=blocked,
        render_reason=None,
        attempt_chain=[fetch_mode],
        quality_score=content_score,
    )


def _build_auto_fetcher(*, telemetry_enabled: bool) -> AutoFetcher:
    settings = AppSettings()
    settings.fetch.backend = "auto"
    settings.fetch.quality.blocked_markers = [
        "captcha",
        "verify you are human",
        "checking your browser",
    ]
    settings.telemetry.enabled = telemetry_enabled
    settings.telemetry.include_events = False
    clock = _Clock()
    telemetry = (
        TraceTelemetry(settings.telemetry, clock=clock)
        if telemetry_enabled
        else NoopTelemetry()
    )
    rt = Runtime(settings=settings, telemetry=telemetry, clock=clock)
    curl_html = """
    <html><head>
      <script>window.cfg = {"captcha":"hcaptcha"}</script>
    </head><body>
      <article><h1>LLM</h1><p>Useful content body with enough visible text.</p></article>
    </body></html>
    """.encode("utf-8")

    return AutoFetcher(
        rt=rt,
        cache=_NoopCache(rt=rt),
        rate_limiter=_NoopRateLimiter(rt=rt),
        httpx_fetcher=_StubHttpxFetcher(
            rt=rt,
            browser_outcome=httpx.ConnectError("dial failed"),
            compat_outcome=httpx.ConnectError("dial failed"),
        ),  # type: ignore[arg-type]
        curl_fetcher=_StubCurlFetcher(
            rt=rt,
            attempt=_attempt(
                fetch_mode="curl_cffi",
                status_code=200,
                content=curl_html,
                content_kind="html",
                text_chars=1300,
                content_score=0.91,
                blocked=True,
            ),
        ),  # type: ignore[arg-type]
        playwright_fetcher=None,
    )


async def _run_auto(telemetry_enabled: bool) -> tuple[FetchResult, dict[str, object]]:
    fetcher = _build_auto_fetcher(telemetry_enabled=telemetry_enabled)
    result = await fetcher.afetch(url=URL, timeout_s=3.0, allow_render=True, rank_index=0)
    telemetry = fetcher.telemetry.summary()
    return result, telemetry


def test_auto_fallback_returns_curl_when_httpx_fails() -> None:
    result, _ = anyio.run(_run_auto, False)
    assert result.fetch_mode == "curl_cffi"
    assert result.status_code == 200
    assert result.content_kind == "html"
    assert len(result.content) > 200


def test_auto_trace_has_winner_summary_and_candidate_spans() -> None:
    _, telemetry = anyio.run(_run_auto, True)
    spans = telemetry.get("spans", [])
    assert isinstance(spans, list)

    auto_spans = [s for s in spans if s.get("name") == "fetch.auto"]
    assert auto_spans
    parent_attrs = auto_spans[-1].get("attrs", {})
    assert parent_attrs.get("winner_mode") == "curl_cffi"
    assert parent_attrs.get("winner_status") == 200
    assert parent_attrs.get("winner_content_kind") == "html"

    candidate_spans = [s for s in spans if s.get("name") == "fetch.auto.candidate"]
    assert candidate_spans
    by_mode = {str(s.get("attrs", {}).get("mode")): s.get("attrs", {}) for s in candidate_spans}
    assert "curl_cffi" in by_mode
    assert "httpx:browser" in by_mode
    assert by_mode["curl_cffi"].get("status") == 200
    assert by_mode["curl_cffi"].get("useful") is True
    assert by_mode["httpx:browser"].get("status") == 0
    assert by_mode["httpx:browser"].get("useful") is False
