from __future__ import annotations

import pytest

import serpsage.components.fetch.curl_cffi as mod
from serpsage.components.fetch.curl_cffi import CurlCffiFetcher
from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import Runtime
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class FakeClock(ClockBase):
    def now_ms(self) -> int:
        return 0


class _Resp:
    def __init__(
        self, *, status_code: int, url: str, headers: dict[str, str], content: bytes
    ) -> None:
        self.status_code = status_code
        self.url = url
        self.headers = headers
        self.content = content


class _Session:
    def __init__(self) -> None:
        self.calls = 0

    async def get(self, url: str, **kwargs):  # noqa: ANN003, ANN201, ARG002
        self.calls += 1
        if self.calls == 1:
            return _Resp(
                status_code=429,
                url=url,
                headers={"retry-after": "10", "content-type": "text/html"},
                content=b"",
            )
        return _Resp(
            status_code=200,
            url=url,
            headers={"content-type": "text/html"},
            content=b"<html><body><p>ok</p></body></html>",
        )

    async def close(self) -> None:
        return


@pytest.mark.anyio
async def test_curl_retry_after_is_capped(monkeypatch: pytest.MonkeyPatch) -> None:
    settings = AppSettings.model_validate(
        {
            "enrich": {
                "enabled": True,
                "fetch": {
                    "backend": "curl_cffi",
                    "timeout_s": 1.0,
                    "curl_cffi": {
                        "retry": {
                            "max_attempts": 2,
                            "base_delay_ms": 10,
                            "max_delay_ms": 20,
                        }
                    },
                },
            },
            "overview": {"enabled": False},
        }
    )
    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=FakeClock())

    calls: list[float] = []

    async def fake_sleep(s: float) -> None:
        calls.append(float(s))

    monkeypatch.setattr("anyio.sleep", fake_sleep)
    monkeypatch.setattr(mod, "CURL_CFFI_AVAILABLE", True)
    monkeypatch.setattr(mod, "CurlSessionFactory", _Session)

    fx = CurlCffiFetcher(rt=rt)

    class Span:
        def set_attr(self, name, value):  # noqa: ANN001
            return

    res = await fx.fetch_attempt(url="https://example.com/x", span=Span())
    assert res.status_code == 200
    assert calls, "expected at least one sleep call"
    assert calls[0] <= 1.5

