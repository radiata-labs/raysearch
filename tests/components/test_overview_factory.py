from __future__ import annotations

import importlib.util
import sys
import types

import pytest

from serpsage.app.bootstrap import build_runtime
from serpsage.components.overview import build_overview_client
from serpsage.core.runtime import Overrides
from serpsage.domain.http import HttpClient
from serpsage.settings.models import AppSettings


def _settings_for_models(*, use_model: str, models: list[dict]) -> AppSettings:
    return AppSettings.model_validate(
        {
            "overview": {
                "enabled": True,
                "use_model": use_model,
                "models": models,
            }
        }
    )


def _openai_model(api_key: str | None = "sk-openai") -> dict:
    return {
        "name": "o1",
        "backend": "openai",
        "base_url": "https://api.openai.com/v1",
        "api_key": api_key,
        "model": "gpt-4.1-mini",
        "timeout_s": 60.0,
        "max_retries": 2,
        "temperature": 0.0,
        "headers": {},
        "schema_strict": True,
    }


def _gemini_model(api_key: str | None = "sk-gemini") -> dict:
    return {
        "name": "g1",
        "backend": "gemini",
        "base_url": "https://generativelanguage.googleapis.com/v1",
        "api_key": api_key,
        "model": "gemini-3-flash",
        "timeout_s": 60.0,
        "max_retries": 2,
        "temperature": 0.0,
        "headers": {},
        "schema_strict": True,
    }


def _build_rt(settings: AppSettings):
    return build_runtime(settings=settings, overrides=Overrides())


def _build_http(rt):
    return HttpClient(rt=rt, ov=Overrides())


def test_factory_routes_openai_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = types.ModuleType("serpsage.components.overview.openai")

    class FakeOpenAIClient:
        def __init__(self, *, rt, http, model_cfg):
            self.rt = rt
            self.http = http
            self.model_cfg = model_cfg

    fake_mod.OpenAIClient = FakeOpenAIClient
    monkeypatch.setitem(sys.modules, "serpsage.components.overview.openai", fake_mod)

    original = importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "openai":
            return object()
        return original(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    settings = _settings_for_models(use_model="o1", models=[_openai_model(), _gemini_model()])
    rt = _build_rt(settings)
    llm = build_overview_client(rt=rt, http=_build_http(rt))
    assert isinstance(llm, FakeOpenAIClient)
    assert llm.model_cfg.name == "o1"


def test_factory_routes_gemini_model(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_mod = types.ModuleType("serpsage.components.overview.gemini")

    class FakeGeminiClient:
        def __init__(self, *, rt, model_cfg):
            self.rt = rt
            self.model_cfg = model_cfg

    fake_mod.GeminiClient = FakeGeminiClient
    monkeypatch.setitem(sys.modules, "serpsage.components.overview.gemini", fake_mod)

    original = importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "google.genai":
            return object()
        return original(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    settings = _settings_for_models(use_model="g1", models=[_openai_model(), _gemini_model()])
    rt = _build_rt(settings)
    llm = build_overview_client(rt=rt, http=_build_http(rt))
    assert isinstance(llm, FakeGeminiClient)
    assert llm.model_cfg.name == "g1"


def test_factory_missing_dependency_error(monkeypatch: pytest.MonkeyPatch) -> None:
    original = importlib.util.find_spec

    def fake_find_spec(name: str):
        if name == "google.genai":
            return None
        return original(name)

    monkeypatch.setattr(importlib.util, "find_spec", fake_find_spec)

    settings = _settings_for_models(use_model="g1", models=[_gemini_model()])
    rt = _build_rt(settings)
    with pytest.raises(RuntimeError, match="google-genai>=1.63.0"):
        build_overview_client(rt=rt, http=_build_http(rt))


def test_factory_missing_api_key_fails_fast() -> None:
    settings = _settings_for_models(use_model="o1", models=[_openai_model(api_key=None)])
    rt = _build_rt(settings)
    with pytest.raises(ValueError, match="overview.models\\[name=o1\\]\\.api_key"):
        build_overview_client(rt=rt, http=_build_http(rt))
