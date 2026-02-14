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


def test_links_inventory_normalizes_and_filters_by_section() -> None:
    extractor = _build_extractor()
    html_doc = FIXTURE.read_text(encoding="utf-8").encode("utf-8")
    base_url = "https://example.com/articles/complex"

    primary_only = extractor.extract(
        url=base_url,
        content=html_doc,
        content_type="text/html",
        include_secondary_content=False,
        collect_links=True,
    )
    assert primary_only.links
    assert all(link.section == "primary" for link in primary_only.links)
    assert all("utm_" not in link.url for link in primary_only.links)

    with_secondary = extractor.extract(
        url=base_url,
        content=html_doc,
        content_type="text/html",
        include_secondary_content=True,
        collect_links=True,
    )
    sections = {link.section for link in with_secondary.links}
    assert "primary" in sections
    assert "secondary" in sections
    assert any(link.same_page for link in with_secondary.links)
    assert all("utm_" not in link.url for link in with_secondary.links)
