from __future__ import annotations

import time

import pytest

from serpsage.components.extract.markdown.extractor import (
    MarkdownExtractor,
    build_extract_profile,
)
from serpsage.components.extract.markdown.postprocess import markdown_to_text
from serpsage.core.runtime import Runtime
from serpsage.models.extract import ExtractContentOptions
from serpsage.settings.models import AppSettings
from serpsage.telemetry.base import ClockBase
from serpsage.telemetry.trace import NoopTelemetry


class _TestClock(ClockBase):
    def now_ms(self) -> int:
        return int(time.time() * 1000)


def _build_extractor() -> MarkdownExtractor:
    rt = Runtime(settings=AppSettings(), telemetry=NoopTelemetry(), clock=_TestClock())
    return MarkdownExtractor(rt=rt)


def test_build_extract_profile_keeps_large_html_capture_floor() -> None:
    settings = AppSettings(fetch={"extract": {"max_markdown_chars": 160_000}})
    profile = build_extract_profile(settings=settings)
    assert profile.max_html_chars >= 1_800_000


@pytest.mark.anyio
async def test_markdown_extractor_full_detail_keeps_tail_content_for_large_html() -> None:
    extractor = _build_extractor()
    tail_text = (
        "This section contains meaningful repository content for machine learning "
        "papers and weekly highlights. "
    ) * 8
    script_filler = "x" * 700_000
    html_doc = (
        "<!DOCTYPE html><html><head><meta charset='utf-8'>"
        "<title>Synthetic Repository</title></head><body>"
        f"<script>{script_filler}</script>"
        "<main><article><h1>Weekly Papers</h1>"
        f"<p>{tail_text}</p>"
        "</article></main></body></html>"
    )

    doc = await extractor.extract(
        url="https://example.com/repo",
        content=html_doc.encode("utf-8"),
        content_type="text/html",
        content_options=ExtractContentOptions(detail="full"),
    )
    text = markdown_to_text(doc.markdown)

    assert "Weekly Papers" in text
    assert "meaningful repository content" in text
    assert len(text) >= 220
