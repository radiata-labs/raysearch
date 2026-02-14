from __future__ import annotations

import time
from pathlib import Path

from serpsage.components.extract.markdown import MarkdownExtractor
from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import Runtime
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry

BASE_URL = "https://example.com/fixture/"
FIXTURES: list[tuple[str, int]] = [
    ("complex_article.html", 180),
    ("code_table_page.html", 120),
    ("github_like_page.html", 90),
    ("complex_list_page.html", 120),
]


class _Clock(ClockBase):
    def now_ms(self) -> int:
        return int(time.time() * 1000)


def _build_extractor() -> MarkdownExtractor:
    settings = AppSettings()
    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_Clock())
    return MarkdownExtractor(rt=rt)


def test_regression_corpus_has_non_empty_markdown_and_plain_text() -> None:
    extractor = _build_extractor()
    for filename, min_chars in FIXTURES:
        html_doc = Path("tests/fixtures/extract", filename).read_text(encoding="utf-8")
        out = extractor.extract(
            url=f"{BASE_URL}{filename}",
            content=html_doc.encode("utf-8"),
            content_type="text/html",
            include_secondary_content=False,
            collect_links=False,
        )
        assert out.content_kind == "html"
        assert len(out.markdown) > 0
        assert len(out.plain_text) >= min_chars
        assert float(out.quality_score) > 0.0
        assert "<table" not in out.markdown.lower()
        assert "<pre" not in out.markdown.lower()
