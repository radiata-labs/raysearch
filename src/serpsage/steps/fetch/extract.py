from __future__ import annotations

from typing_extensions import override
from urllib.parse import urlparse

from serpsage.components.crawl.utils import is_reddit_thread_url
from serpsage.components.extract import ExtractorBase
from serpsage.components.extract.utils import markdown_to_text
from serpsage.dependencies import Depends
from serpsage.models.components.extract import ExtractRef
from serpsage.models.steps.fetch import FetchStepContext
from serpsage.steps.base import StepBase
from serpsage.utils import clean_whitespace


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
        fetch_cfg = self.settings.fetch
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
    prepared: list[tuple[int, int, ExtractRef]] = []
    seen: set[str] = set()
    excluded = {str(item or "").strip().split("#")[0] for item in exclude if item}
    for order, item in enumerate(values):
        url = str(item.url or "").strip().split("#")[0]
        if not url or url in seen:
            continue
        if url in excluded:
            continue
        seen.add(url)
        prepared.append(
            (
                _subpage_candidate_priority(
                    item=item.model_copy(update={"url": url}),
                    excluded=excluded,
                ),
                order,
                item.model_copy(update={"url": url}),
            )
        )
    max_items = max(1, int(limit))
    ranked = sorted(prepared, key=lambda item: (-item[0], item[1]))
    return [item for _, _, item in ranked[:max_items]]


def _subpage_candidate_priority(*, item: ExtractRef, excluded: set[str]) -> int:
    url = str(item.url or "").strip()
    text = clean_whitespace(str(item.text or ""))
    if not url:
        return -1000
    parsed = urlparse(url)
    host = parsed.netloc.lower().removeprefix("www.")
    path = parsed.path or "/"
    score = 0
    if host in {"reddit.com", "old.reddit.com"}:
        if is_reddit_thread_url(url):
            score += 120
            if "/comment/" in path:
                score += 15
            else:
                score += 45
        if path.startswith("/user/"):
            score -= 90
        if path.startswith("/login"):
            score -= 100
        if path.startswith("/search"):
            score -= 40
        if path.startswith("/r/") and "/comments/" not in path:
            score -= 35
        if "force-legacy-sct=1" in url:
            score += 5
    elif host == "reddit.app.link":
        score -= 120
    elif "redditinc.com" in host or "reddithelp.com" in host:
        score -= 110
    else:
        score += 25
    normalized_text = text.casefold()
    if normalized_text in {"log in", "view in app", "privacy policy", "user agreement"}:
        score -= 80
    if normalized_text == "continue this thread":
        score += 20
    if text and len(text) >= 12:
        score += min(30, len(text) // 6)
    if text.endswith("ago"):
        score -= 25
    if url in excluded:
        score -= 200
    return score


__all__ = ["FetchExtractStep"]
