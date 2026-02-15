from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.contracts.services import LLMClientBase

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.models.llm import ChatResult


class NullLLMClient(LLMClientBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def chat(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any] | None = None,
        timeout_s: float | None = None,
    ) -> ChatResult:
        _ = model, messages, schema, timeout_s
        raise RuntimeError("LLM is not configured (missing api_key or disabled).")


__all__ = ["NullLLMClient"]
