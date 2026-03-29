from __future__ import annotations

from typing_extensions import override

from raysearch.models.app.response import FetchOthersResult, FetchResultItem
from raysearch.models.steps.fetch import FetchStepContext
from raysearch.steps.base import StepBase
from raysearch.utils import normalize_iso8601_string


class FetchFinalizeStep(StepBase[FetchStepContext]):
    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        if ctx.error.failed:
            return ctx
        if ctx.page.doc is None:
            ctx.error.failed = True
            ctx.error.tag = "SOURCE_NOT_AVAILABLE"
            ctx.error.detail = "missing extracted content"
            await self.tracker.error(
                name="fetch.finalize.failed",
                request_id=ctx.request_id,
                step="fetch.finalize",
                error_code="fetch_extract_failed",
                error_message="missing extracted content",
                data={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "fatal": True,
                },
            )
            return ctx
        content = str(ctx.page.doc.content.output_markdown or "")
        abstracts = [
            str(item.text) for item in list(ctx.analysis.abstracts.ranked or [])
        ]
        abstract_scores = [
            float(item.score) for item in list(ctx.analysis.abstracts.ranked or [])
        ]

        # Build excluded URL set for others.image_links only
        # Note: We don't exclude subpage URLs from others.links because
        # the first link should always point to the full text article
        page_url = ctx.url.strip().split("#")[0]
        page_image = (ctx.page.doc.meta.image or "").strip()
        page_favicon = (ctx.page.doc.meta.favicon or "").strip()
        excluded_urls: set[str] = {
            url for url in [page_url, page_image, page_favicon] if url
        }

        others_result = {}
        if not ctx.related.enabled:
            subpages_result = []
        else:
            if ctx.request.others is not None:
                filtered_others = _filter_others_result(
                    others=ctx.related.others,
                    excluded_urls=excluded_urls,
                    max_links=ctx.request.others.max_links,
                    max_image_links=ctx.request.others.max_image_links,
                )
                others_result = {"others": filtered_others}
            subpages_result = [
                item.result
                for item in list(ctx.related.subpages.items or [])
                if item.result is not None
            ]
        ctx.result = FetchResultItem(
            url=ctx.url,
            title=ctx.page.doc.meta.title or ctx.page.pre_fetched_title or "",
            published_date=_normalize_published_date(
                ctx.page.doc.meta.published_date or ""
            ),
            author=ctx.page.doc.meta.author or ctx.page.pre_fetched_author or "",
            image=ctx.page.doc.meta.image or "",
            favicon=ctx.page.doc.meta.favicon or "",
            content=content,
            abstracts=abstracts,
            abstract_scores=abstract_scores,
            overview=(
                ""
                if ctx.analysis.overview.output is None
                else ctx.analysis.overview.output
            ),
            subpages=subpages_result,
            **others_result,
        )
        return ctx


def _filter_others_result(
    *,
    others: FetchOthersResult,
    excluded_urls: set[str],
    max_links: int | None,
    max_image_links: int | None,
) -> FetchOthersResult:
    """Filter out excluded URLs from others result.

    Links are processed first, then image_links are filtered to exclude
    any URLs that appeared in links (to prevent duplicates across fields).
    """
    # Process links first
    filtered_links: list[str] = []
    links_seen: set[str] = set()
    for url in others.links:
        clean_url = url.strip().split("#")[0]
        if clean_url and clean_url not in excluded_urls and clean_url not in links_seen:
            filtered_links.append(clean_url)
            links_seen.add(clean_url)
            if max_links is not None and len(filtered_links) >= max_links:
                break

    # Process image_links, excluding URLs from links and excluded_urls
    filtered_image_links: list[str] = []
    all_excluded = excluded_urls | links_seen
    for url in others.image_links:
        clean_url = url.strip()
        if clean_url and clean_url not in all_excluded:
            filtered_image_links.append(clean_url)
            all_excluded.add(clean_url)
            if (
                max_image_links is not None
                and len(filtered_image_links) >= max_image_links
            ):
                break

    return FetchOthersResult(
        links=filtered_links,
        image_links=filtered_image_links,
    )


def _normalize_published_date(value: str) -> str:
    try:
        return normalize_iso8601_string(value, allow_blank=True)
    except ValueError:
        return ""


__all__ = ["FetchFinalizeStep"]
