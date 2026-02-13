from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

from serpsage.components.overview.router import RoutedLLMClient

if TYPE_CHECKING:
    from serpsage.contracts.services import HttpClientBase, LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.settings.models import OverviewModelSettings


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


def _require_api_key(model_cfg: OverviewModelSettings) -> bool:
    return bool(model_cfg.api_key)


def build_overview_client(*, rt: Runtime, http: HttpClientBase) -> LLMClientBase:
    from serpsage.components.overview.null import NullLLMClient

    routes: dict[str, tuple[LLMClientBase, str]] = {}
    for model_cfg in rt.settings.llm.models:
        if not _require_api_key(model_cfg):
            continue
        backend = str(model_cfg.backend or "openai").lower()
        if backend == "openai":
            _ensure_optional_dep(
                module_name="openai",
                package_name="openai",
                backend=backend,
                model_name=model_cfg.name,
            )
            from serpsage.components.overview.openai import OpenAIClient

            routes[model_cfg.name] = (
                OpenAIClient(rt=rt, http=http, model_cfg=model_cfg),
                model_cfg.model,
            )
            continue
        if backend == "gemini":
            _ensure_optional_dep(
                module_name="google.genai",
                package_name="google-genai>=1.63.0",
                backend=backend,
                model_name=model_cfg.name,
            )
            from serpsage.components.overview.gemini import GeminiClient

            routes[model_cfg.name] = (
                GeminiClient(rt=rt, model_cfg=model_cfg),
                model_cfg.model,
            )
            continue
        raise ValueError(
            f"unsupported overview backend `{backend}`; expected openai|gemini"
        )

    if not routes:
        return NullLLMClient(rt=rt)
    return RoutedLLMClient(rt=rt, routes=routes)


__all__ = ["build_overview_client"]
