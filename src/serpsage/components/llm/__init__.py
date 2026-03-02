from __future__ import annotations

import importlib.util
from collections.abc import Callable
from typing import TypeAlias

from serpsage.components.http.base import HttpClientBase
from serpsage.components.llm.base import LLMClientBase
from serpsage.components.llm.router import RoutedLLMClient
from serpsage.core.runtime import Runtime
from serpsage.settings.models import LLMModelSettings

RouteTuple: TypeAlias = tuple[LLMClientBase, str]
RouteBuilder: TypeAlias = Callable[
    [Runtime, HttpClientBase, LLMModelSettings], RouteTuple
]


def _ensure_optional_dep(
    *,
    module_name: str,
    package_name: str,
    backend: str,
    model_name: str,
) -> None:
    if importlib.util.find_spec(module_name) is not None:
        return
    raise RuntimeError(
        "overview model "
        f"`{model_name}` (backend `{backend}`) requires optional dependency "
        f'`{package_name}`. Install with `pip install "serpsage[overview]"`.'
    )


def _require_api_key(model_cfg: LLMModelSettings) -> bool:
    return bool(model_cfg.api_key)


def _build_openai_route(
    rt: Runtime, http: HttpClientBase, model_cfg: LLMModelSettings
) -> RouteTuple:
    _ensure_optional_dep(
        module_name="openai",
        package_name="openai",
        backend="openai",
        model_name=model_cfg.name,
    )
    from serpsage.components.llm.openai import OpenAIClient

    return OpenAIClient(rt=rt, http=http, model_cfg=model_cfg), model_cfg.model


def _build_gemini_route(
    rt: Runtime, http: HttpClientBase, model_cfg: LLMModelSettings
) -> RouteTuple:
    _ = http
    _ensure_optional_dep(
        module_name="google.genai",
        package_name="google-genai>=1.63.0",
        backend="gemini",
        model_name=model_cfg.name,
    )
    from serpsage.components.llm.gemini import GeminiClient

    return GeminiClient(rt=rt, model_cfg=model_cfg), model_cfg.model


def _build_dashscope_route(
    rt: Runtime, http: HttpClientBase, model_cfg: LLMModelSettings
) -> RouteTuple:
    _ = http
    _ensure_optional_dep(
        module_name="dashscope",
        package_name="dashscope",
        backend="dashscope",
        model_name=model_cfg.name,
    )
    from serpsage.components.llm.dashscope import DashScopeClient

    return DashScopeClient(rt=rt, model_cfg=model_cfg), model_cfg.model


def _route_builders() -> dict[str, RouteBuilder]:
    return {
        "openai": _build_openai_route,
        "gemini": _build_gemini_route,
        "dashscope": _build_dashscope_route,
    }


def build_overview_client(*, rt: Runtime, http: HttpClientBase) -> LLMClientBase:
    from serpsage.components.llm.null import NullLLMClient

    builders = _route_builders()
    routes: dict[str, tuple[LLMClientBase, str]] = {}
    for model_cfg in rt.settings.llm.models:
        if not _require_api_key(model_cfg):
            continue
        backend = str(model_cfg.backend or "openai").lower()
        builder = builders.get(backend)
        if builder is None:
            raise ValueError(
                "unsupported overview backend "
                f"`{backend}`; expected openai|gemini|dashscope"
            )
        routes[model_cfg.name] = builder(rt, http, model_cfg)

    if not routes:
        return NullLLMClient(rt=rt)
    return RoutedLLMClient(rt=rt, routes=routes)


__all__ = [
    "LLMClientBase",
    "build_overview_client",
]
