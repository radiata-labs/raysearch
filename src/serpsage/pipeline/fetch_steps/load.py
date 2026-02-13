from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import FetchStepContext
from serpsage.pipeline.step import PipelineStep

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.contracts.services import FetcherBase
    from serpsage.core.runtime import Runtime


class FetchLoadStep(PipelineStep[FetchStepContext]):
    span_name = "step.fetch_load"

    def __init__(self, *, rt: Runtime, fetcher: FetcherBase) -> None:
        super().__init__(rt=rt)
        self._fetcher = fetcher
        self.bind_deps(fetcher)

    @override
    async def run_inner(
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        url = (ctx.request.url or "").strip()
        if not url:
            ctx.page.error = "empty url"
            return ctx

        params = dict(ctx.request.params or {})
        timeout_s = _as_float(
            params.get("timeout_s"), float(self.settings.fetch.timeout_s)
        )
        allow_render = _as_bool(
            params.get("allow_render"),
            default=bool(self.settings.fetch.render.enabled),
        )
        depth = str(params.get("depth") or "") or None
        rank_index = _as_int(params.get("rank_index"), 0)

        t0 = time.monotonic()
        fetch = await self._fetcher.afetch(
            url=url,
            timeout_s=timeout_s,
            allow_render=allow_render,
            rank_index=rank_index,
        )
        ctx.page.timing_ms["fetch_ms"] = int((time.monotonic() - t0) * 1000)
        ctx.fetch_result = fetch
        ctx.page.fetch_mode = fetch.fetch_mode
        ctx.page.content_kind = fetch.content_kind
        if fetch.attempt_chain:
            ctx.page.warnings.append(f"attempt_chain:{'->'.join(fetch.attempt_chain)}")
        span.set_attr("fetch_mode", str(fetch.fetch_mode))
        span.set_attr("content_kind", str(fetch.content_kind))
        span.set_attr("status_code", int(fetch.status_code))
        span.set_attr("timeout_s", float(timeout_s))
        span.set_attr("allow_render", bool(allow_render))
        span.set_attr("depth", str(depth or ""))
        span.set_attr("rank_index", int(rank_index))
        return ctx


def _as_float(value: object, default: float) -> float:
    try:
        return float(value)  # pyright: ignore[reportArgumentType]
    except Exception:
        return float(default)


def _as_int(value: object, default: int) -> int:
    try:
        return int(value)  # pyright: ignore[reportArgumentType]
    except Exception:
        return int(default)


def _as_bool(value: object, *, default: bool) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"1", "true", "yes", "on"}:
            return True
        if v in {"0", "false", "no", "off"}:
            return False
    return bool(default)


__all__ = ["FetchLoadStep"]
