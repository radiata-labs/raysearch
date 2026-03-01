from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.components.extract.markdown.postprocess import markdown_to_text
from serpsage.models.errors import AppError
from serpsage.models.extract import ExtractedLink
from serpsage.models.pipeline import FetchStepContext
from serpsage.steps.base import StepBase

if TYPE_CHECKING:
    from serpsage.components.extract import ExtractorBase
    from serpsage.core.runtime import Runtime

class FetchExtractStep(StepBase[FetchStepContext]):

    def __init__(self, *, rt: Runtime, extractor: ExtractorBase) -> None:
        super().__init__(rt=rt)
        self._extractor = extractor
        self.bind_deps(extractor)

    @override
    async def run_inner(
        self, ctx: FetchStepContext
    ) -> FetchStepContext:
        if ctx.fatal:
            return ctx
        if ctx.artifacts.fetch_result is None:
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

        collect_links = bool(
            ctx.enable_others_and_subpages
            and (
                ctx.runtime.max_links is not None
                or ctx.runtime.max_links_for_subpages is not None
            )
        )
        collect_images = bool(
            ctx.enable_others_and_subpages and ctx.runtime.max_image_links is not None
        )

        try:
            assert ctx.artifacts.fetch_result is not None
            extracted = await self._extractor.extract(
                url=ctx.artifacts.fetch_result.url,
                content=ctx.artifacts.fetch_result.content,
                content_type=ctx.artifacts.fetch_result.content_type,
                content_options=ctx.resolved.content_options,
                collect_links=collect_links,
                collect_images=collect_images,
            )
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

        ctx.artifacts.extracted = extracted
        if ctx.enable_others_and_subpages:
            ctx.output.others.links = _prepare_links(
                values=[
                    str(item.url or "").strip().split("#")[0]
                    for item in list(extracted.links or [])
                ],
                limit=ctx.runtime.max_links,
            )
            ctx.subpages.links = _prepare_subpage_links(
                values=extracted.links,
                exclude=[ctx.url],
                limit=ctx.runtime.max_links_for_subpages,
            )
            ctx.output.others.image_links = _prepare_links(
                values=[
                    str(item.url or "").strip()
                    for item in list(extracted.image_links or [])
                ],
                limit=ctx.runtime.max_image_links,
            )
        else:
            ctx.output.others.links = []
            ctx.output.others.image_links = []
        markdown = str(extracted.markdown or "")
        text_chars = len(markdown_to_text(markdown))
        min_text_chars = int(self.settings.fetch.extract.min_text_chars)
        if not markdown.strip():
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
        elif text_chars < min_text_chars:
            ctx.fatal = True
            ctx.errors.append(
                AppError(
                    code="fetch_extract_failed",
                    message="extracted content below min_text_chars",
                    details={
                        "url": ctx.url,
                        "url_index": ctx.url_index,
                        "stage": "extract",
                        "fatal": True,
                        "crawl_mode": ctx.runtime.crawl_mode,
                        "text_chars": int(text_chars),
                        "min_text_chars": int(min_text_chars),
                    },
                )
            )
        return ctx

def _prepare_links(*, values: list[str], limit: int | None) -> list[str]:
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
    *, values: list[ExtractedLink], exclude: list[str], limit: int | None
) -> list[ExtractedLink]:
    if limit is None:
        return []
    out: list[ExtractedLink] = []
    seen: set[str] = set()
    max_items = max(1, int(limit))
    for item in values:
        url = str(item.url or "").strip().split("#")[0]
        if not url or url in seen:
            continue
        if url in exclude:
            continue
        seen.add(url)
        new_item = ExtractedLink(**item.model_dump() | {"url": url})
        out.append(new_item)
        if len(out) >= max_items:
            break
    return out

__all__ = ["FetchExtractStep"]

