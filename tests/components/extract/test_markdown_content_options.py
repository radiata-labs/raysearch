from __future__ import annotations

import time
from pathlib import Path

from serpsage.components.extract.markdown import MarkdownExtractor
from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import Runtime
from serpsage.models.extract import ExtractContentOptions
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry

FIXTURE = Path("tests/fixtures/extract/complex_article.html")
BASE_URL = "https://example.com/articles/complex"


class _Clock(ClockBase):
    def now_ms(self) -> int:
        return int(time.time() * 1000)


def _build_extractor() -> MarkdownExtractor:
    settings = AppSettings()
    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_Clock())
    return MarkdownExtractor(rt=rt)


def test_depth_low_and_high_follow_secondary_semantics() -> None:
    extractor = _build_extractor()
    html_doc = FIXTURE.read_text(encoding="utf-8").encode("utf-8")

    low_doc = extractor.extract(
        url=BASE_URL,
        content=html_doc,
        content_type="text/html",
        content_options=ExtractContentOptions(depth="low"),
    )
    assert "Main body phrase Alpha" in low_doc.markdown
    assert "Related recommendation block Omega" not in low_doc.markdown

    high_doc = extractor.extract(
        url=BASE_URL,
        content=html_doc,
        content_type="text/html",
        content_options=ExtractContentOptions(depth="high"),
    )
    assert "Main body phrase Alpha" in high_doc.markdown
    assert "Related recommendation block Omega" in high_doc.markdown


def test_depth_medium_adapts_to_short_primary_content() -> None:
    extractor = _build_extractor()
    html_doc = (
        "<html><body><main><article><p>short body</p></article></main>"
        "<aside><p>Secondary block should be included in medium mode.</p></aside>"
        "</body></html>"
    ).encode("utf-8")

    medium_doc = extractor.extract(
        url="https://example.com/medium-short",
        content=html_doc,
        content_type="text/html",
        content_options=ExtractContentOptions(depth="medium"),
    )
    assert "Secondary block should be included in medium mode." in medium_doc.markdown


def test_include_and_exclude_tags_filter_output() -> None:
    extractor = _build_extractor()
    html_doc = FIXTURE.read_text(encoding="utf-8").encode("utf-8")
    doc = extractor.extract(
        url=BASE_URL,
        content=html_doc,
        content_type="text/html",
        content_options=ExtractContentOptions(
            depth="high",
            include_tags=["body", "sidebar"],
            exclude_tags=["sidebar"],
        ),
    )
    assert "Main body phrase Alpha" in doc.markdown
    assert "Related recommendation block Omega" not in doc.markdown


def test_include_html_tags_keeps_html_markup_in_markdown_payload() -> None:
    extractor = _build_extractor()
    html_doc = FIXTURE.read_text(encoding="utf-8").encode("utf-8")
    doc = extractor.extract(
        url=BASE_URL,
        content=html_doc,
        content_type="text/html",
        content_options=ExtractContentOptions(
            depth="low",
            include_html_tags=True,
            include_tags=["body"],
        ),
    )
    assert "<section data-serpsage-tag=\"body\">" in doc.markdown
    assert "<h1>" in doc.markdown or "<p>" in doc.markdown
