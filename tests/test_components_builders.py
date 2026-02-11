from __future__ import annotations

import httpx
import pytest

import serpsage.components.overview as overview_factory
import serpsage.components.provider as provider_factory
import serpsage.components.rank as rank_factory
from serpsage.components.fetch.http_client_unit import HttpClientUnit
from serpsage.components.overview.null import NullLLMClient
from serpsage.components.overview.openai import OpenAIClient
from serpsage.components.provider.searxng import SearxngProvider
from serpsage.components.rank.blend import BlendRanker
from serpsage.components.rank.heuristic import HeuristicRanker
from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import Runtime
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class _Clock(ClockBase):
    def now_ms(self) -> int:
        return 0


def _rt(settings: AppSettings) -> Runtime:
    return Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_Clock())


@pytest.mark.anyio
async def test_build_provider_selects_searxng() -> None:
    settings = AppSettings.model_validate({"provider": {"backend": "searxng"}})
    rt = _rt(settings)
    async with httpx.AsyncClient() as client:
        unit = HttpClientUnit(rt=rt, client=client, owns_client=False)
        provider = provider_factory.build_provider(rt=rt, http=unit)
    assert isinstance(provider, SearxngProvider)


def test_build_ranker_selects_heuristic() -> None:
    settings = AppSettings.model_validate({"rank": {"backend": "heuristic"}})
    ranker = rank_factory.build_ranker(rt=_rt(settings))
    assert isinstance(ranker, HeuristicRanker)


def test_build_ranker_selects_blend() -> None:
    settings = AppSettings.model_validate({"rank": {"backend": "blend"}})
    ranker = rank_factory.build_ranker(rt=_rt(settings))
    assert isinstance(ranker, BlendRanker)


def test_build_ranker_bm25_fail_fast_when_dependency_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(rank_factory, "BM25_AVAILABLE", False)
    settings = AppSettings.model_validate({"rank": {"backend": "bm25"}})
    with pytest.raises(RuntimeError, match="rank backend `bm25`"):
        rank_factory.build_ranker(rt=_rt(settings))


@pytest.mark.anyio
async def test_build_overview_client_respects_enabled_flag() -> None:
    settings = AppSettings.model_validate({"overview": {"enabled": False}})
    rt = _rt(settings)
    async with httpx.AsyncClient() as client:
        unit = HttpClientUnit(rt=rt, client=client, owns_client=False)
        llm = overview_factory.build_overview_client(rt=rt, http=unit)
    assert isinstance(llm, NullLLMClient)


@pytest.mark.anyio
async def test_build_overview_client_openai_requires_api_key() -> None:
    settings = AppSettings.model_validate(
        {"overview": {"enabled": True, "backend": "openai"}}
    )
    rt = _rt(settings)
    async with httpx.AsyncClient() as client:
        unit = HttpClientUnit(rt=rt, client=client, owns_client=False)
        with pytest.raises(ValueError, match="overview.openai.llm.api_key"):
            overview_factory.build_overview_client(rt=rt, http=unit)


@pytest.mark.anyio
async def test_build_overview_client_openai_selected() -> None:
    settings = AppSettings.model_validate(
        {
            "overview": {
                "enabled": True,
                "backend": "openai",
                "openai": {"llm": {"api_key": "dummy"}},
            }
        }
    )
    rt = _rt(settings)
    async with httpx.AsyncClient() as client:
        unit = HttpClientUnit(rt=rt, client=client, owns_client=False)
        llm = overview_factory.build_overview_client(rt=rt, http=unit)
    assert isinstance(llm, OpenAIClient)
