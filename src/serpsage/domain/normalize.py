from __future__ import annotations

from typing import Any
from urllib.parse import urlparse

from serpsage.app.response import ResultItem
from serpsage.core.workunit import WorkUnit
from serpsage.text.normalize import clean_whitespace, strip_html


class ResultNormalizer(WorkUnit):
    def normalize_many(self, raw_results: list[dict[str, Any]]) -> list[ResultItem]:
        include_raw = bool(self.settings.pipeline.include_raw)
        return [
            self._normalize_one(raw, include_raw=include_raw) for raw in raw_results
        ]

    def _normalize_one(self, raw: dict[str, Any], *, include_raw: bool) -> ResultItem:
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

        return ResultItem(
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


__all__ = ["ResultNormalizer"]
