from __future__ import annotations

from typing_extensions import override

from serpsage.components.extract import ExtractorBase
from serpsage.components.extract.utils import markdown_to_text
from serpsage.dependencies import Depends
from serpsage.models.components.extract import ExtractRef
from serpsage.models.steps.fetch import FetchStepContext
from serpsage.steps.base import StepBase


class FetchExtractStep(StepBase[FetchStepContext]):
    extractor: ExtractorBase = Depends()

    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        if ctx.error.failed:
            return ctx
        if ctx.page.raw is None:
            ctx.error.failed = True
            ctx.error.tag = "SOURCE_NOT_AVAILABLE"
            ctx.error.detail = "missing fetch result"
            await self.tracker.error(
                name="fetch.extract.failed",
                request_id=ctx.request_id,
                step="fetch.extract",
                error_code="fetch_load_failed",
                error_message="missing fetch result",
                data={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "fatal": True,
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
                crawl_backend=ctx.page.raw.crawl_backend,
                content_kind=ctx.page.raw.content_kind,
                content_options=ctx.page.extract,
                collect_links=collect_links,
                collect_images=collect_images,
            )
        except Exception as exc:  # noqa: BLE001
            ctx.error.failed = True
            ctx.error.tag = "CRAWL_UNKNOWN_ERROR"
            ctx.error.detail = str(exc)
            await self.tracker.error(
                name="fetch.extract.failed",
                request_id=ctx.request_id,
                step="fetch.extract",
                error_code="fetch_extract_failed",
                error_type=type(exc).__name__,
                error_message=str(exc),
                data={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "fatal": True,
                },
            )
            return ctx
        ctx.page.doc = extracted
        if ctx.related.enabled:
            # Use the same ordered links list for both others and subpages
            # The ordering is determined by _links (importance score)
            ordered_links = list(extracted.refs.links or [])
            ordered_urls = [
                str(item.url or "").strip().split("#")[0] for item in ordered_links
            ]
            # Collect links and image_links for later filtering in finalize step
            ctx.related.others.links = _collect_urls(values=ordered_urls)
            # Subpages gets the complete ordered list (no truncation here)
            # Filtering and ranking happens in subpages.py
            ctx.related.subpages.candidates = _prepare_subpage_candidates(
                values=ordered_links,
                exclude=[ctx.url],
            )
            ctx.related.others.image_links = _collect_urls(
                values=[
                    str(item.url or "").strip()
                    for item in list(extracted.refs.images or [])
                ],
            )
        else:
            ctx.related.others.links = []
            ctx.related.others.image_links = []
            ctx.related.subpages.candidates = []
        fetch_cfg = self.settings.fetch
        markdown = str(extracted.content.markdown or "")
        text_chars = len(markdown_to_text(markdown))
        min_text_chars = int(fetch_cfg.extract.min_text_chars)
        if not markdown.strip():
            ctx.error.failed = True
            ctx.error.tag = "SOURCE_NOT_AVAILABLE"
            ctx.error.detail = "no content extracted"
            await self.tracker.error(
                name="fetch.extract.failed",
                request_id=ctx.request_id,
                step="fetch.extract",
                error_code="fetch_extract_failed",
                error_message="no content extracted",
                data={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "fatal": True,
                },
            )
        elif text_chars < min_text_chars:
            ctx.error.failed = True
            ctx.error.tag = "SOURCE_NOT_AVAILABLE"
            ctx.error.detail = "extracted content below min_text_chars"
            await self.tracker.error(
                name="fetch.extract.failed",
                request_id=ctx.request_id,
                step="fetch.extract",
                error_code="fetch_extract_failed",
                error_message="extracted content below min_text_chars",
                data={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "fatal": True,
                    "text_chars": int(text_chars),
                    "min_text_chars": int(min_text_chars),
                },
            )
        return ctx


def _collect_urls(*, values: list[str]) -> list[str]:
    """Collect unique URLs, preserving order. Filtering is done in finalize step."""
    out: list[str] = []
    seen: set[str] = set()
    for raw in values:
        url = str(raw or "").strip()
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
    return out


def _prepare_subpage_candidates(
    *, values: list[ExtractRef], exclude: list[str]
) -> list[ExtractRef]:
    """Prepare subpage candidates preserving the order from _links.

    Returns complete list without truncation. Ranking and filtering
    happens in subpages.py based on keywords.
    """
    seen: set[str] = set()
    excluded = {str(item or "").strip().split("#")[0] for item in exclude if item}
    out: list[ExtractRef] = []
    for item in values:
        url = str(item.url or "").strip().split("#")[0]
        if not url or url in seen:
            continue
        if url in excluded:
            continue
        seen.add(url)
        out.append(item.model_copy(update={"url": url}))
    return out


__all__ = ["FetchExtractStep"]
