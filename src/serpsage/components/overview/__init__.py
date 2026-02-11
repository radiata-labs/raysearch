from __future__ import annotations

from typing import TYPE_CHECKING

from serpsage.components.overview.null import NullLLMClient
from serpsage.components.overview.openai import OpenAIClient
from serpsage.contracts.services import LLMClientBase

if TYPE_CHECKING:
    from serpsage.components.http import HttpClient
    from serpsage.core.runtime import Runtime


def build_overview_client(*, rt: Runtime, http: HttpClient) -> LLMClientBase:
    cfg = rt.settings.overview
    if not bool(cfg.enabled):
        return NullLLMClient(rt=rt)

    backend = str(cfg.backend or "openai").lower()
    if backend == "null":
        return NullLLMClient(rt=rt)
    if backend == "openai":
        if not cfg.openai.llm.api_key:
            raise ValueError(
                "overview backend `openai` requires `overview.openai.llm.api_key` "
                "when `overview.enabled=true`"
            )
        return OpenAIClient(rt=rt, http=http)

    raise ValueError(f"unsupported overview backend `{backend}`; expected openai|null")


__all__ = [
    "NullLLMClient",
    "OpenAIClient",
    "build_overview_client",
]
