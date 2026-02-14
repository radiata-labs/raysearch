from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.pipeline import FetchStepContext
from serpsage.pipeline.step import PipelineStep

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime


class FetchFinalizeStep(PipelineStep[FetchStepContext]):
    span_name = "step.fetch_finalize"

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        if not ctx.return_content:
            ctx.page.markdown = ""
        for idx, ch in enumerate(ctx.page.chunks, 1):
            ch.chunk_id = ch.chunk_id or f"S1:C{idx}"
        if "total_ms" not in ctx.page.timing_ms:
            total = 0
            for key in (
                "fetch_ms",
                "extract_ms",
                "chunk_ms",
                "score_ms",
                "overview_ms",
            ):
                total += int(ctx.page.timing_ms.get(key, 0))
            ctx.page.timing_ms["total_ms"] = total
        span.set_attr("chunks_count", int(len(ctx.page.chunks)))
        span.set_attr("has_overview", bool(ctx.overview is not None))
        span.set_attr("has_content_output", bool(ctx.return_content))
        return ctx


__all__ = ["FetchFinalizeStep"]
