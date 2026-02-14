from __future__ import annotations

import time
from pathlib import Path

from serpsage.components.extract.markdown import MarkdownExtractor
from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import Runtime
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry

FIXTURE = Path("tests/fixtures/extract/complex_article.html")


class _Clock(ClockBase):
    def now_ms(self) -> int:
        return int(time.time() * 1000)


def _build_extractor() -> MarkdownExtractor:
    settings = AppSettings()
    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_Clock())
    return MarkdownExtractor(rt=rt)


def test_secondary_content_switch_respects_bool_semantics() -> None:
    extractor = _build_extractor()
    html_doc = FIXTURE.read_text(encoding="utf-8").encode("utf-8")
    base_url = "https://example.com/articles/complex"

    primary_only = extractor.extract(
        url=base_url,
        content=html_doc,
        content_type="text/html; charset=utf-8",
        include_secondary_content=False,
        collect_links=False,
    )
    assert "Main body phrase Alpha" in primary_only.markdown
    assert "Related recommendation block Omega" not in primary_only.markdown
    assert "Comments thread Sigma" not in primary_only.markdown

    full = extractor.extract(
        url=base_url,
        content=html_doc,
        content_type="text/html; charset=utf-8",
        include_secondary_content=True,
        collect_links=False,
    )
    assert "Main body phrase Alpha" in full.markdown
    assert "Related recommendation block Omega" in full.markdown
    assert "Comments thread Sigma" in full.markdown
    assert "## Secondary Content" in full.markdown
