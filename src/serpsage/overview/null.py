from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override

from serpsage.contracts.base import WorkUnit
from serpsage.contracts.protocols import LLMClient

if TYPE_CHECKING:
    from serpsage.app.runtime import CoreRuntime
    from serpsage.contracts.llm import ChatJSONResult


class NullLLMClient(WorkUnit, LLMClient):
    def __init__(self, *, rt: CoreRuntime) -> None:
        super().__init__(rt=rt)

    @override
    async def chat_json(
        self,
        *,
        model: str,
        messages: list[dict[str, str]],
        schema: dict[str, Any],
        timeout_s: float | None = None,
    ) -> ChatJSONResult:
        raise RuntimeError("LLM is not configured (missing api_key or disabled).")


__all__ = ["NullLLMClient"]
