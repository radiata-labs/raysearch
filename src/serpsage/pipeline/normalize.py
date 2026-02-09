from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from serpsage.app.container import Container
from serpsage.app.response import ResultItem
from serpsage.pipeline.steps import StepContext
from serpsage.text.normalize import clean_whitespace, strip_html


class NormalizeStep:
    def __init__(self, container: Container) -> None:
        self._c = container

    async def run(self, ctx: StepContext) -> StepContext:
        span = self._c.telemetry.start_span("step.normalize")
        try:
            out: list[ResultItem] = []
            include_raw = bool(ctx.settings.pipeline.include_raw)
            for raw in ctx.raw_results:
                item = _normalize_one(raw, include_raw=include_raw)
                out.append(item)
            ctx.results = out
            span.set_attr("n", len(out))
            return ctx
        finally:
            span.end()


def _normalize_one(raw: dict[str, Any], *, include_raw: bool) -> ResultItem:
    url = str(raw.get("url") or "").strip()
    title_raw = str(raw.get("title") or "").strip()

    snippet_raw = raw.get("snippet")
    if snippet_raw is None:
        snippet_raw = raw.get("content")
    if snippet_raw is None:
        snippet_raw = raw.get("description")
    snippet_text = str(snippet_raw or "").strip()

    published = raw.get("publishedDate")
    published_str = "" if published in (None, "null") else str(published).strip()
    engine = str(raw.get("engine") or "").strip()

    title = clean_whitespace(strip_html(title_raw))
    snippet = clean_whitespace(strip_html(snippet_text))
    domain = _extract_domain(url)

    return ResultItem(
        url=url,
        title=title,
        snippet=snippet,
        domain=domain,
        published_date=published_str,
        engine=engine,
        raw=dict(raw) if include_raw else None,
    )


def _extract_domain(url: str) -> str:
    if not url:
        return ""
    parsed = urlparse(url)
    host = parsed.netloc
    if not host and parsed.path:
        host = parsed.path.split("/")[0]
    return host.lower()


__all__ = ["NormalizeStep"]

