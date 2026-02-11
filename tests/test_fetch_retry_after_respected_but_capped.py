from __future__ import annotations

import httpx
import pytest

from serpsage.app.runtime import CoreRuntime
from serpsage.fetch.http import HttpxFetcher
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class FakeClock:
    def now_ms(self) -> int:
        return 0


@pytest.mark.anyio
async def test_retry_after_is_capped(monkeypatch: pytest.MonkeyPatch):
    settings = AppSettings.model_validate(
        {
            "enrich": {
                "enabled": True,
                "fetch": {
                    "timeout_s": 1.0,
                    "max_attempts_per_strategy": 2,
                    "total_budget_s": 3.0,
                    "retry": {"max_attempts": 2, "base_delay_ms": 10, "max_delay_ms": 20},
                },
            },
            "overview": {"enabled": False},
        }
    )
    rt = CoreRuntime(settings=settings, telemetry=NoopTelemetry(), clock=FakeClock())

    calls: list[float] = []

    async def fake_sleep(s: float) -> None:
        calls.append(float(s))

    monkeypatch.setattr("anyio.sleep", fake_sleep)

    n = 0

    def handler(request: httpx.Request) -> httpx.Response:  # noqa: ARG001
        nonlocal n
        n += 1
        if n == 1:
            return httpx.Response(429, headers={"retry-after": "10"}, content=b"")
        return httpx.Response(
            200, headers={"content-type": "text/html"}, content=b"<html><body><p>ok</p></body></html>"
        )

    transport = httpx.MockTransport(handler)
    async with httpx.AsyncClient(transport=transport, follow_redirects=True) as client:
        fx = HttpxFetcher(rt=rt, http=client)
        # span can be a dummy object with set_attr method
        class Span:
            def set_attr(self, name, value):  # noqa: ANN001
                return

        res = await fx.fetch_attempt(url="https://example.com/x", profile="compat", span=Span())

    assert res.status_code == 200
    assert calls, "expected at least one sleep call"
    assert calls[0] <= 1.5

