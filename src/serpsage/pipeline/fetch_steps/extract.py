from __future__ import annotations

import time
from typing import TYPE_CHECKING
from typing_extensions import override

from anyio import to_thread

from serpsage.models.errors import AppError
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
        if ctx.fatal:
            return ctx
        if ctx.fetch_result is None:
            ctx.fatal = True
            ctx.errors.append(
                AppError(
                    code="fetch_load_failed",
                    message="missing fetch result",
                    details={
                        "url": ctx.url,
                        "url_index": ctx.url_index,
                        "stage": "extract",
                        "fatal": True,
                        "crawl_mode": ctx.runtime.crawl_mode,
                    },
                )
            )
            return ctx

        collect_links = bool(ctx.runtime.max_links is not None)
        t0 = time.monotonic()

        def extract() -> ExtractedDocument:
            assert ctx.fetch_result is not None
            return self._extractor.extract(
                url=ctx.fetch_result.url,
                content=ctx.fetch_result.content,
                content_type=ctx.fetch_result.content_type,
                content_options=ctx.content_options,
                collect_links=collect_links,
            )

        try:
            extracted = await to_thread.run_sync(extract)
        except Exception as exc:  # noqa: BLE001
            ctx.fatal = True
            ctx.errors.append(
                AppError(
                    code="fetch_extract_failed",
                    message=str(exc),
                    details={
                        "url": ctx.url,
                        "url_index": ctx.url_index,
                        "stage": "extract",
                        "fatal": True,
                        "crawl_mode": ctx.runtime.crawl_mode,
                    },
                )
            )
            return ctx
        extract_ms = int((time.monotonic() - t0) * 1000)

        ctx.extracted = extracted
        ctx.links = _prepare_links(
            extracted=extracted,
            max_links=ctx.runtime.max_links,
        )
        span.set_attr("extractor_used", str(extracted.extractor_used))
        span.set_attr("quality_score", float(extracted.quality_score))
        span.set_attr("extract_ms", int(extract_ms))
        span.set_attr("content_depth", str(ctx.content_options.depth))
        span.set_attr("include_html_tags", bool(ctx.content_options.include_html_tags))
        span.set_attr("collect_links", bool(collect_links))
        span.set_attr("primary_chars", int(extracted.stats.get("primary_chars", 0)))
        span.set_attr(
            "secondary_chars",
            int(extracted.stats.get("secondary_chars", 0)),
        )
        span.set_attr("links_count", int(len(extracted.links or [])))
        span.set_attr("engine_chain", str(extracted.stats.get("engine_chain", "")))
        if not (extracted.plain_text or "").strip():
            ctx.fatal = True
            ctx.errors.append(
                AppError(
                    code="fetch_extract_failed",
                    message="no content extracted",
                    details={
                        "url": ctx.url,
                        "url_index": ctx.url_index,
                        "stage": "extract",
                        "fatal": True,
                        "crawl_mode": ctx.runtime.crawl_mode,
                    },
                )
            )
        return ctx


def _prepare_links(*, extracted: ExtractedDocument, max_links: int | None) -> list[str]:
    if max_links is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for item in list(extracted.links or []):
        url = str(item.url or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= int(max_links):
            break
    return out


__all__ = ["FetchExtractStep"]
