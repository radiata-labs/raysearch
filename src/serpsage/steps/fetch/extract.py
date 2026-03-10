from __future__ import annotations

from typing_extensions import override

from serpsage.components.extract import ExtractorBase
from serpsage.components.extract.utils import markdown_to_text
from serpsage.dependencies import Inject
from serpsage.models.components.extract import ExtractRef
from serpsage.models.steps.fetch import FetchStepContext
from serpsage.steps.base import StepBase


class FetchExtractStep(StepBase[FetchStepContext]):
    extractor: ExtractorBase = Inject()

    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        if ctx.error.failed:
            return ctx
        if ctx.page.raw is None:
            ctx.error.failed = True
            ctx.error.tag = "SOURCE_NOT_AVAILABLE"
            ctx.error.detail = "missing fetch result"
            await self.emit_tracking_event(
                event_name="fetch.extract.error",
                request_id=ctx.request_id,
                stage="extract",
                status="error",
                error_code="fetch_load_failed",
                attrs={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "fatal": True,
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "message": "missing fetch result",
                },
            )
            return ctx
        collect_links = bool(
            ctx.related.enabled
            and (
                ctx.related.link_limit is not None
                or ctx.related.subpages.candidate_limit is not None
            )
        )
        collect_images = bool(
            ctx.related.enabled and ctx.related.image_limit is not None
        )
        try:
            assert ctx.page.raw is not None
            extracted = await self.extractor.extract(
                url=ctx.page.raw.url,
                content=ctx.page.raw.content,
                content_type=ctx.page.raw.content_type,
                content_options=ctx.page.extract,
                collect_links=collect_links,
                collect_images=collect_images,
            )
        except Exception as exc:  # noqa: BLE001
            ctx.error.failed = True
            ctx.error.tag = "CRAWL_UNKNOWN_ERROR"
            ctx.error.detail = str(exc)
            await self.emit_tracking_event(
                event_name="fetch.extract.error",
                request_id=ctx.request_id,
                stage="extract",
                status="error",
                error_code="fetch_extract_failed",
                error_type=type(exc).__name__,
                attrs={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "fatal": True,
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "message": str(exc),
                },
            )
            return ctx
        ctx.page.doc = extracted
        if ctx.related.enabled:
            ctx.related.others.links = _prepare_urls(
                values=[
                    str(item.url or "").strip().split("#")[0]
                    for item in list(extracted.refs.links or [])
                ],
                limit=ctx.related.link_limit,
            )
            ctx.related.subpages.candidates = _prepare_subpage_links(
                values=list(extracted.refs.links or []),
                exclude=[ctx.url],
                limit=ctx.related.subpages.candidate_limit,
            )
            ctx.related.others.image_links = _prepare_urls(
                values=[
                    str(item.url or "").strip()
                    for item in list(extracted.refs.images or [])
                ],
                limit=ctx.related.image_limit,
            )
        else:
            ctx.related.others.links = []
            ctx.related.others.image_links = []
            ctx.related.subpages.candidates = []
        fetch_cfg = ctx.settings.fetch
        markdown = str(extracted.content.markdown or "")
        text_chars = len(markdown_to_text(markdown))
        min_text_chars = int(fetch_cfg.extract.min_text_chars)
        if not markdown.strip():
            ctx.error.failed = True
            ctx.error.tag = "SOURCE_NOT_AVAILABLE"
            ctx.error.detail = "no content extracted"
            await self.emit_tracking_event(
                event_name="fetch.extract.error",
                request_id=ctx.request_id,
                stage="extract",
                status="error",
                error_code="fetch_extract_failed",
                attrs={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "fatal": True,
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "message": "no content extracted",
                },
            )
        elif text_chars < min_text_chars:
            ctx.error.failed = True
            ctx.error.tag = "SOURCE_NOT_AVAILABLE"
            ctx.error.detail = "extracted content below min_text_chars"
            await self.emit_tracking_event(
                event_name="fetch.extract.error",
                request_id=ctx.request_id,
                stage="extract",
                status="error",
                error_code="fetch_extract_failed",
                attrs={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "fatal": True,
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "message": "extracted content below min_text_chars",
                    "text_chars": int(text_chars),
                    "min_text_chars": int(min_text_chars),
                },
            )
        return ctx


def _prepare_urls(*, values: list[str], limit: int | None) -> list[str]:
    if limit is None:
        return []
    out: list[str] = []
    seen: set[str] = set()
    max_items = max(1, int(limit))
    for raw in values:
        url = str(raw or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= max_items:
            break
    return out


def _prepare_subpage_links(
    *, values: list[ExtractRef], exclude: list[str], limit: int | None
) -> list[ExtractRef]:
    if limit is None:
        return []
    out: list[ExtractRef] = []
    seen: set[str] = set()
    max_items = max(1, int(limit))
    for item in values:
        url = str(item.url or "").strip().split("#")[0]
        if not url or url in seen:
            continue
        if url in exclude:
            continue
        seen.add(url)
        out.append(item.model_copy(update={"url": url}))
        if len(out) >= max_items:
            break
    return out


__all__ = ["FetchExtractStep"]
