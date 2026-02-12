from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serpsage.contracts.services import LLMClientBase
    from serpsage.core.runtime import Runtime
    from serpsage.domain.http import HttpClient


def build_overview_client(*, rt: Runtime, http: HttpClient) -> LLMClientBase:
    cfg = rt.settings.overview
    if not bool(cfg.enabled):
        from serpsage.components.overview.null import NullLLMClient

        return NullLLMClient(rt=rt)

    backend = str(cfg.backend or "openai").lower()
    if backend == "null":
        from serpsage.components.overview.null import NullLLMClient

        return NullLLMClient(rt=rt)
    if backend == "openai":
        if not cfg.openai.llm.api_key:
            raise ValueError(
                "overview backend `openai` requires `overview.openai.llm.api_key` "
                "when `overview.enabled=true`"
            )
        from serpsage.components.overview.openai import OpenAIClient

        return OpenAIClient(rt=rt, http=http)

    raise ValueError(f"unsupported overview backend `{backend}`; expected openai|null")


__all__ = [
    "build_overview_client",
]
