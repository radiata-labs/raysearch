from __future__ import annotations

from typing import Any

import httpx

from serpsage.app.bootstrap import build_runtime
from serpsage.contracts.services import CacheBase
from serpsage.core.runtime import Overrides
from serpsage.core.workunit import WorkUnit
from serpsage.pipeline.steps.overview import OverviewStep
from serpsage.settings.models import AppSettings


class DummyBuilder(WorkUnit):
    def build_messages(self, *, query: str, results: list[Any]) -> list[dict[str, str]]:
        return [{"role": "user", "content": query}]

    def schema(self) -> dict[str, Any]:
        return {"type": "object"}

    async def build_overview(self, *, query: str, results: list[Any]) -> Any:
        raise RuntimeError("not used in this test")


class DummyCache(CacheBase):
    async def aget(self, *, namespace: str, key: str) -> bytes | None:
        return None

    async def aset(self, *, namespace: str, key: str, value: bytes, ttl_s: int) -> None:
        return


class StatusError(Exception):
    def __init__(self, status: int):
        super().__init__(f"status={status}")
        self.status_code = int(status)


class APIConnectionError(Exception):
    pass


def _make_step() -> OverviewStep:
    settings = AppSettings.model_validate(
        {
            "overview": {
                "enabled": True,
                "use_model": "o1",
                "models": [
                    {
                        "name": "o1",
                        "backend": "openai",
                        "base_url": "https://api.openai.com/v1",
                        "api_key": "sk",
                        "model": "gpt-4.1-mini",
                        "timeout_s": 60.0,
                        "max_retries": 2,
                        "temperature": 0.0,
                        "headers": {},
                        "schema_strict": True,
                    }
                ],
            }
        }
    )
    rt = build_runtime(settings=settings, overrides=Overrides())
    return OverviewStep(rt=rt, builder=DummyBuilder(rt=rt), cache=DummyCache(rt=rt))


def test_cache_key_includes_backend_and_model_name() -> None:
    step = _make_step()
    messages = [{"role": "user", "content": "q"}]
    schema = {"type": "object"}

    key_openai = step._overview_cache_key(
        model="same-model",
        backend="openai",
        model_name="o1",
        base_url="https://api.openai.com/v1",
        messages=messages,
        schema=schema,
        schema_strict=True,
    )
    key_gemini = step._overview_cache_key(
        model="same-model",
        backend="gemini",
        model_name="g1",
        base_url="https://generativelanguage.googleapis.com/v1",
        messages=messages,
        schema=schema,
        schema_strict=True,
    )
    assert key_openai != key_gemini


def test_map_overview_error_provider_agnostic() -> None:
    step = _make_step()
    common = {
        "backend": "gemini",
        "model_name": "g1",
        "model": "gemini-3-flash",
        "base_url": "https://generativelanguage.googleapis.com/v1",
        "attempt": 1,
    }

    cases = [
        (StatusError(400), "overview_bad_request"),
        (StatusError(401), "overview_auth_failed"),
        (StatusError(403), "overview_auth_failed"),
        (StatusError(429), "overview_rate_limited"),
        (StatusError(503), "overview_server_error"),
        (TimeoutError("timed out"), "overview_timeout"),
        (httpx.ReadTimeout("boom"), "overview_timeout"),
        (APIConnectionError("connection error"), "overview_timeout"),
    ]
    for exc, expected in cases:
        code, details = step._map_overview_error(exc, **common)
        assert code == expected
        assert details["backend"] == "gemini"
        assert details["model_name"] == "g1"
        assert details["model"] == "gemini-3-flash"
