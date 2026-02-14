from __future__ import annotations

import time

from serpsage.components.extract.markdown import MarkdownExtractor
from serpsage.components.extract.markdown.types import CandidateDoc
from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import Runtime
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class _Clock(ClockBase):
    def now_ms(self) -> int:
        return int(time.time() * 1000)


def _build_extractor() -> MarkdownExtractor:
    settings = AppSettings()
    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=_Clock())
    return MarkdownExtractor(rt=rt)


def test_enhance_for_missing_features_backfills_code_and_table() -> None:
    extractor = _build_extractor()
    best = CandidateDoc(
        markdown="# Intro\n\nPlain body paragraph only.",
        plain_text="Intro Plain body paragraph only.",
        extractor_used="fastdom",
        quality_score=0.45,
        warnings=[],
        stats={
            "heading_count": 1,
            "table_count": 0,
            "code_block_count": 0,
            "link_count": 0,
        },
        primary_chars=40,
        secondary_chars=0,
    )
    donor = CandidateDoc(
        markdown=(
            "## Example\n\n```python\nprint('hi')\n```\n\n"
            "| key | value |\n| --- | --- |\n| retries | 3 |"
        ),
        plain_text="Example print hi key value retries 3",
        extractor_used="readability",
        quality_score=0.9,
        warnings=[],
        stats={
            "heading_count": 1,
            "table_count": 1,
            "code_block_count": 1,
            "link_count": 0,
        },
        primary_chars=60,
        secondary_chars=0,
    )

    enhanced = extractor._enhance_for_missing_features(
        best=best,
        candidates=[best, donor],
        profile=extractor._profile,
    )
    assert "```" in enhanced.markdown
    assert "| --- |" in enhanced.markdown
    assert int(enhanced.stats.get("table_count", 0)) >= 1
    assert int(enhanced.stats.get("code_block_count", 0)) >= 1
