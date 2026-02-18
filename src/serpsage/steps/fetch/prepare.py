from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.app.request import (
    FetchAbstractsRequest,
    FetchContentRequest,
    FetchOverviewRequest,
)
from serpsage.models.errors import AppError
from serpsage.models.extract import ExtractContentOptions
from serpsage.models.pipeline import FetchStepContext
from serpsage.steps.base import StepBase
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase


class FetchPrepareStep(StepBase[FetchStepContext]):
    span_name = "step.fetch_prepare"

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        url = clean_whitespace(ctx.url or "")
        if not url:
            ctx.fatal = True
            ctx.errors.append(
                AppError(
                    code="fetch_load_failed",
                    message="empty url",
                    details={
                        "url": ctx.url,
                        "url_index": ctx.url_index,
                        "stage": "prepare",
                        "fatal": True,
                        "crawl_mode": ctx.others.crawl_mode,
                    },
                )
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
                ctx.others.max_links_for_subpages = int(auto_links_limit)
        else:
            ctx.others.max_links = None
            ctx.others.max_image_links = None

        ctx.return_content = bool(return_content)
        ctx.content_request = content_request
        ctx.content_options = content_options
        ctx.abstracts_request = abstracts_request
        ctx.overview_request = overview_request
        ctx.subpages.subpages_enabled = bool(subpages_enabled)
        ctx.subpages.subpages_max = int(subpages_max) if subpages_enabled else 0
        ctx.subpages.subpages_keywords = list(subpages_keywords)
        ctx.subpages.subpages_query = subpages_query if subpages_enabled else ""
        ctx.subpages_result = []

        span.set_attr("has_content_output", bool(return_content))
        span.set_attr("has_abstracts", bool(abstracts_request is not None))
        span.set_attr("has_overview", bool(overview_request is not None))
        span.set_attr("content_detail", str(content_request.detail))
        span.set_attr("crawl_mode", str(ctx.others.crawl_mode))
        span.set_attr("crawl_timeout_s", float(ctx.others.crawl_timeout_s))
        span.set_attr("subpages_enabled", bool(ctx.subpages.subpages_enabled))
        span.set_attr("subpages_max", int(ctx.subpages.subpages_max))
        span.set_attr(
            "subpages_keywords_count", int(len(ctx.subpages.subpages_keywords))
        )
        span.set_attr("url_index", int(ctx.url_index))
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
