from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override
from urllib.parse import urlsplit

from serpsage.app.request import (
    FetchAbstractsRequest,
    FetchContentRequest,
    FetchOverviewRequest,
)
from serpsage.components.extract.models import ExtractContentOptions
from serpsage.steps.base import StepBase
from serpsage.steps.models import FetchStepContext
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


class FetchPrepareStep(StepBase[FetchStepContext]):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        url = clean_whitespace(ctx.url or "")
        if not url:
            ctx.fatal = True
            ctx.error_tag = "SOURCE_NOT_AVAILABLE"
            ctx.error_detail = "empty url"
            await self.emit_tracking_event(
                event_name="fetch.prepare.error",
                request_id=ctx.request_id,
                stage="prepare",
                status="error",
                error_code="fetch_load_failed",
                attrs={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "fatal": True,
                    "crawl_mode": str(ctx.runtime.crawl_mode),
                    "message": "empty url",
                },
            )
            return ctx
        parsed = urlsplit(url)
        if parsed.scheme.lower() not in {"http", "https"} or not parsed.netloc:
            ctx.fatal = True
            ctx.error_tag = "UNSUPPORTED_URL"
            ctx.error_detail = "unsupported url format"
            await self.emit_tracking_event(
                event_name="fetch.prepare.error",
                request_id=ctx.request_id,
                stage="prepare",
                status="error",
                error_code="fetch_load_failed",
                attrs={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "fatal": True,
                    "crawl_mode": str(ctx.runtime.crawl_mode),
                    "message": "unsupported url format",
                },
            )
            return ctx
        raw_abstracts = ctx.request.abstracts
        abstracts_request: FetchAbstractsRequest | None
        if isinstance(raw_abstracts, bool):
            abstracts_request = FetchAbstractsRequest() if raw_abstracts else None
        else:
            query = clean_whitespace(raw_abstracts.query or "")
            abstracts_request = raw_abstracts.model_copy(
                update={"query": query or None}
            )
        raw_overview = ctx.request.overview
        overview_request: FetchOverviewRequest | None
        if isinstance(raw_overview, bool):
            overview_request = FetchOverviewRequest() if raw_overview else None
        else:
            query = clean_whitespace(raw_overview.query or "")
            overview_request = raw_overview.model_copy(update={"query": query or None})
        raw_content = ctx.request.content
        content_request: FetchContentRequest
        return_content: bool
        if isinstance(raw_content, bool):
            return_content = bool(raw_content)
            content_request = FetchContentRequest()
        else:
            return_content = True
            content_request = raw_content
        content_options = ExtractContentOptions(
            detail=content_request.detail,
            include_html_tags=bool(content_request.include_html_tags),
            include_tags=list(content_request.include_tags),
            exclude_tags=list(content_request.exclude_tags),
        )
        ctx.url = url
        subpages_enabled = False
        subpages_max = 0
        subpages_keywords: list[str] = []
        subpages_query = ""
        if ctx.enable_others_and_subpages:
            subpages_request = ctx.request.subpages
            if (
                subpages_request is not None
                and subpages_request.max_subpages is not None
            ):
                subpages_keywords = _parse_subpage_keywords(
                    subpages_request.subpage_keywords
                )
                subpages_max = int(subpages_request.max_subpages)
                subpages_enabled = bool(subpages_max > 0 and subpages_keywords)
                if subpages_enabled:
                    subpages_query = ", ".join(subpages_keywords)
            if subpages_enabled:
                link_cap = int(self.settings.fetch.extract.link_max_count)
                auto_links_limit = min(link_cap, max(20, int(subpages_max) * 5))
                preset_limit = (
                    int(ctx.runtime.max_links_for_subpages)
                    if ctx.runtime.max_links_for_subpages is not None
                    else 0
                )
                if preset_limit > 0:
                    ctx.runtime.max_links_for_subpages = int(
                        min(link_cap, max(1, int(preset_limit)))
                    )
                else:
                    ctx.runtime.max_links_for_subpages = int(auto_links_limit)
        else:
            ctx.runtime.max_links = None
            ctx.runtime.max_image_links = None
        ctx.resolved.return_content = bool(return_content)
        ctx.resolved.content_request = content_request
        ctx.resolved.content_options = content_options
        ctx.resolved.abstracts_request = abstracts_request
        ctx.resolved.overview_request = overview_request
        ctx.subpages.enabled = bool(subpages_enabled)
        ctx.subpages.max_count = int(subpages_max) if subpages_enabled else 0
        ctx.subpages.keywords = list(subpages_keywords)
        ctx.subpages.query = subpages_query if subpages_enabled else ""
        ctx.subpages.results = []
        ctx.subpages.result_links = []
        ctx.subpages.md_for_abstract = []
        ctx.subpages.overview_scores = []
        return ctx


def _parse_subpage_keywords(value: str | None) -> list[str]:
    text = clean_whitespace(value or "").replace("\uff0c", ",")
    if not text:
        return []
    out: list[str] = []
    seen: set[str] = set()
    for raw in text.split(","):
        token = clean_whitespace(raw)
        if not token or token in seen:
            continue
        seen.add(token)
        out.append(token)
    return out


__all__ = ["FetchPrepareStep"]
