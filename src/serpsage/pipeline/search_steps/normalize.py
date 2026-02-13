from __future__ import annotations

from typing import TYPE_CHECKING, Any
from typing_extensions import override
from urllib.parse import urlparse

from serpsage.models.pipeline import SearchStepContext
from serpsage.pipeline.step import PipelineStep
from serpsage.text.normalize import clean_whitespace, strip_html
from serpsage.text.tokenize import tokenize_for_query
from serpsage.text.utils import extract_intent_tokens

if TYPE_CHECKING:
    from serpsage.app.response import ResultItem
    from serpsage.contracts.lifecycle import SpanBase
    from serpsage.core.runtime import Runtime


class NormalizeStep(PipelineStep[SearchStepContext]):
    span_name = "step.normalize"

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: SearchStepContext, *, span: SpanBase
    ) -> SearchStepContext:
        span.set_attr("raw_results_count", int(len(ctx.raw_results or [])))
        ctx.results = self._normalize_many(ctx.raw_results)
        ctx.query_tokens = tokenize_for_query(ctx.request.query)
        ctx.profile_name, ctx.profile = self.settings.select_profile(
            query=ctx.request.query, explicit=ctx.request.profile
        )
        ctx.intent_tokens = extract_intent_tokens(
            ctx.request.query, ctx.profile.intent_terms
        )
        span.set_attr("profile_name", str(ctx.profile_name or ""))
        span.set_attr("results_count", int(len(ctx.results or [])))
        return ctx

    def _normalize_many(self, raw_results: list[dict[str, object]]) -> list[ResultItem]:
        from serpsage.app.response import ResultItem  # noqa: PLC0415

        include_raw = bool(self.settings.search.include_raw)
        return [
            self._normalize_one(raw, include_raw=include_raw, result_type=ResultItem)
            for raw in raw_results
        ]

    def _normalize_one(
        self,
        raw: dict[str, Any],
        *,
        include_raw: bool,
        result_type: type[ResultItem],
    ) -> ResultItem:
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
        domain = self._extract_domain(url)

        return result_type(
            url=url,
            title=title,
            snippet=snippet,
            domain=domain,
            published_date=published_str,
            engine=engine,
            raw=dict(raw) if include_raw else None,
        )

    def _extract_domain(self, url: str) -> str:
        if not url:
            return ""
        parsed = urlparse(url)
        host = parsed.netloc
        if not host and parsed.path:
            host = parsed.path.split("/")[0]
        return host.lower()


__all__ = ["NormalizeStep"]
