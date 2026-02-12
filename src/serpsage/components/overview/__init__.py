from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serpsage.contracts.services import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.domain.http import HttpClient
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


def _require_api_key(model_cfg: OverviewModelSettings) -> None:
    if model_cfg.api_key:
        return
    raise ValueError(
        "overview enabled requires "
        f"`overview.models[name={model_cfg.name}].api_key` "
        f"for backend `{model_cfg.backend}`"
    )


def build_overview_client(*, rt: Runtime, http: HttpClient) -> LLMClientBase:
    cfg = rt.settings.overview
    if not bool(cfg.enabled):
        from serpsage.components.overview.null import NullLLMClient

        return NullLLMClient(rt=rt)

    model_cfg = cfg.resolve_model()
    _require_api_key(model_cfg)
    backend = str(model_cfg.backend or "openai").lower()
    if backend == "openai":
        _ensure_optional_dep(
            module_name="openai",
            package_name="openai",
            backend=backend,
            model_name=model_cfg.name,
        )
        from serpsage.components.overview.openai import OpenAIClient

        return OpenAIClient(rt=rt, http=http, model_cfg=model_cfg)
    if backend == "gemini":
        _ensure_optional_dep(
            module_name="google.genai",
            package_name="google-genai>=1.63.0",
            backend=backend,
            model_name=model_cfg.name,
        )
        from serpsage.components.overview.gemini import GeminiClient

        return GeminiClient(rt=rt, model_cfg=model_cfg)

    raise ValueError(
        f"unsupported overview backend `{backend}`; expected openai|gemini"
    )


__all__ = [
    "build_overview_client",
]
