from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing_extensions import override

from anyio import to_thread

from serpsage.models.pipeline import FetchStepContext
from serpsage.pipeline.step import PipelineStep

if TYPE_CHECKING:
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.contracts.services import ExtractorBase
    from serpsage.core.runtime import Runtime
    from serpsage.models.extract import ExtractedDocument


class FetchExtractStep(PipelineStep[FetchStepContext]):
    span_name = "step.fetch_extract"

    def __init__(self, *, rt: Runtime, extractor: ExtractorBase) -> None:
        super().__init__(rt=rt)
        self._extractor = extractor
        self.bind_deps(extractor)

    @override
    async def run_inner(
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        if ctx.fetch_result is None:
            ctx.page.error = ctx.page.error or "missing fetch result"
            return ctx

        include_secondary = (
            bool(ctx.request.include_secondary_content)
            if ctx.request.include_secondary_content is not None
            else bool(self.settings.fetch.extract.include_secondary_content_default)
        )
        collect_links = bool(self.settings.fetch.extract.collect_links_default)
        t0 = time.monotonic()

        def extract() -> ExtractedDocument:
            assert ctx.fetch_result is not None
            return self._extractor.extract(
                url=ctx.fetch_result.url,
                content=ctx.fetch_result.content,
                content_type=ctx.fetch_result.content_type,
                include_secondary_content=include_secondary,
                collect_links=collect_links,
            )

        extracted = await to_thread.run_sync(extract)
        extract_ms = int((time.monotonic() - t0) * 1000)
        ctx.extracted = extracted
        ctx.page.markdown = extracted.markdown
        ctx.page.content_kind = extracted.content_kind
        ctx.page.warnings.extend(extracted.warnings or [])
        ctx.page.timing_ms["extract_ms"] = extract_ms
        span.set_attr("extractor_used", str(extracted.extractor_used))
        span.set_attr("quality_score", float(extracted.quality_score))
        span.set_attr("extract_ms", int(extract_ms))
        span.set_attr("include_secondary_content", bool(include_secondary))
        span.set_attr("collect_links", bool(collect_links))
        span.set_attr("primary_chars", int(extracted.stats.get("primary_chars", 0)))
        span.set_attr(
            "secondary_chars",
            int(extracted.stats.get("secondary_chars", 0)),
        )
        span.set_attr("links_count", int(len(extracted.links or [])))
        span.set_attr("engine_chain", str(extracted.stats.get("engine_chain", "")))
        if not (extracted.plain_text or "").strip():
            ctx.page.error = ctx.page.error or "no content extracted"
        return ctx


__all__ = ["FetchExtractStep"]
