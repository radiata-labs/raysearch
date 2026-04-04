from __future__ import annotations

from typing import cast
from typing_extensions import override
from urllib.parse import urlsplit

from raysearch.models.app.request import (
    FetchAbstractsRequest,
    FetchContentRequest,
    FetchOverviewRequest,
)
from raysearch.models.components.extract import ExtractContentTag, ExtractSpec
from raysearch.models.steps.fetch import FetchStepContext
from raysearch.steps.base import StepBase
from raysearch.utils import clean_whitespace

_DETAIL_SECTION_ORDER: tuple[ExtractContentTag, ...] = (
    "metadata",
    "header",
    "navigation",
    "banner",
    "body",
    "sidebar",
    "footer",
)

_DETAIL_SECTIONS: dict[str, tuple[ExtractContentTag, ...]] = {
    "concise": ("metadata", "body"),
    "standard": ("metadata", "header", "body"),
    "full": _DETAIL_SECTION_ORDER,
}


class FetchPrepareStep(StepBase[FetchStepContext]):
    @override
    async def should_run(self, ctx: FetchStepContext) -> bool:
        """Prepare always runs (validates URL, initializes context)."""
        _ = ctx
        return True

    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        url = clean_whitespace(ctx.url or "")
        if not url:
            ctx.error.failed = True
            ctx.error.tag = "SOURCE_NOT_AVAILABLE"
            ctx.error.detail = "empty url"
            await self.tracker.error(
                name="fetch.prepare.failed",
                request_id=ctx.request_id,
                step="fetch.prepare",
                error_code="fetch_load_failed",
                error_message="empty url",
                data={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "fatal": True,
                },
            )
            return ctx
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            ctx.error.failed = True
            ctx.error.tag = "UNSUPPORTED_URL"
            ctx.error.detail = "unsupported url format"
            await self.tracker.error(
                name="fetch.prepare.failed",
                request_id=ctx.request_id,
                step="fetch.prepare",
                error_code="fetch_load_failed",
                error_message="unsupported url format",
                data={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "fatal": True,
                },
            )
            return ctx
        raw_abstracts = ctx.request.abstracts
        if isinstance(raw_abstracts, bool):
            abstracts_request = FetchAbstractsRequest() if raw_abstracts else None
        else:
            query = clean_whitespace(raw_abstracts.query or "")
            abstracts_request = raw_abstracts.model_copy(
                update={"query": query or None}
            )
        raw_overview = ctx.request.overview
        if isinstance(raw_overview, bool):
            overview_request = FetchOverviewRequest() if raw_overview else None
        else:
            query = clean_whitespace(raw_overview.query or "")
            overview_request = raw_overview.model_copy(update={"query": query or None})
        raw_content = ctx.request.content
        if isinstance(raw_content, bool):
            return_content = bool(raw_content)
            content_request = FetchContentRequest()
        else:
            return_content = True
            content_request = raw_content
        ctx.url = url
        ctx.page.return_content = bool(return_content)
        ctx.page.content_request = content_request
        ctx.page.extract = ExtractSpec(
            detail=content_request.detail,
            keep_html=bool(content_request.include_html_tags),
            sections=_resolve_sections(content_request),
            emit_output=bool(return_content),
            keep_markdown_links=bool(content_request.include_markdown_links),
            output_max_chars=content_request.max_chars,
        )
        ctx.page.raw = None
        ctx.page.doc = None
        ctx.analysis.abstracts.request = abstracts_request
        ctx.analysis.abstracts.prepared = []
        ctx.analysis.abstracts.ranked = []
        ctx.analysis.overview.request = overview_request
        ctx.analysis.overview.ranked = []
        ctx.analysis.overview.output = None
        fetch_cfg = self.settings.fetch
        subpages_enabled = False
        subpages_limit = 0
        subpages_keywords: list[str] = []
        if ctx.related.enabled:
            subpages_request = ctx.request.subpages
            ctx.related.link_limit = (
                ctx.request.others.max_links if ctx.request.others is not None else None
            )
            ctx.related.image_limit = (
                ctx.request.others.max_image_links
                if ctx.request.others is not None
                else None
            )
            if (
                subpages_request is not None
                and subpages_request.max_subpages is not None
            ):
                subpages_keywords = list(subpages_request.subpage_keywords or [])
                subpages_limit = int(subpages_request.max_subpages)
                subpages_enabled = bool(subpages_limit > 0)
            if subpages_enabled:
                link_cap = int(fetch_cfg.extract.link_max_count)
                auto_limit = min(link_cap, max(20, int(subpages_limit) * 5))
                preset_limit = (
                    int(ctx.related.subpages.candidate_limit)
                    if ctx.related.subpages.candidate_limit is not None
                    else 0
                )
                ctx.related.subpages.candidate_limit = (
                    int(min(link_cap, max(1, preset_limit)))
                    if preset_limit > 0
                    else int(auto_limit)
                )
        else:
            ctx.related.link_limit = None
            ctx.related.image_limit = None
            ctx.related.others.links = []
            ctx.related.others.image_links = []
        ctx.related.subpages.enabled = bool(subpages_enabled)
        ctx.related.subpages.limit = int(subpages_limit) if subpages_enabled else 0
        ctx.related.subpages.keywords = list(subpages_keywords)
        ctx.related.subpages.candidates = []
        ctx.related.subpages.items = []
        return ctx


def _resolve_sections(content_request: FetchContentRequest) -> list[ExtractContentTag]:
    if content_request.include_tags:
        selected = [
            "metadata",
            *cast("list[ExtractContentTag]", list(content_request.include_tags)),
        ]
    else:
        selected = list(
            _DETAIL_SECTIONS.get(content_request.detail, ("metadata", "body"))
        )
    excluded = {item for item in content_request.exclude_tags if item != "metadata"}
    out: list[ExtractContentTag] = []
    seen: set[ExtractContentTag] = set()
    for item in _DETAIL_SECTION_ORDER:
        if item not in selected or item in excluded or item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


__all__ = ["FetchPrepareStep"]
