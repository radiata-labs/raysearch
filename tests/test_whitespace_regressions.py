from __future__ import annotations

import ast
import re
from pathlib import Path

from serpsage.models.pipeline import ResearchSource, ResearchTrackResult
from serpsage.steps.answer.generate import (
    _sanitize_output_text,
    _strip_citation_markers_in_text,
)
from serpsage.steps.fetch.overview import _coerce_json_output
from serpsage.steps.research.content import ResearchContentStep
from serpsage.steps.research.render import ResearchRenderStep
from serpsage.steps.research.subreport import ResearchSubreportStep

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"


def test_research_render_subreport_excerpt_keeps_newlines() -> None:
    step = ResearchRenderStep.__new__(ResearchRenderStep)
    track = ResearchTrackResult(
        question_id="q1",
        question="test",
        subreport_markdown="line1\n\nline2",
    )

    packet = step._build_track_result_packet([track])

    assert packet[0]["subreport_excerpt"] == "line1\n\nline2"


def test_research_subreport_source_excerpt_keeps_newlines() -> None:
    step = ResearchSubreportStep.__new__(ResearchSubreportStep)
    source = ResearchSource(
        source_id=1,
        url="https://example.com",
        title="example",
        content="alpha\n\nbeta",
    )

    packet = step._build_source_evidence_packet([source])

    assert packet[0]["content_excerpt"] == "alpha\n\nbeta"


def test_research_content_packet_keeps_newlines() -> None:
    step = ResearchContentStep.__new__(ResearchContentStep)
    source = ResearchSource(
        source_id=1,
        url="https://example.com",
        title="example",
        content="first\nsecond",
    )

    packet = step._build_content_packet(
        sources=[source],
        source_ids=[1],
        max_chars=200,
    )

    assert "first\nsecond" in packet


def test_answer_text_sanitize_preserves_paragraphs() -> None:
    raw = "line1[citation:1]\n\nline2"

    stripped = _strip_citation_markers_in_text(raw)
    cleaned = _sanitize_output_text(stripped)

    assert "\n\n" in cleaned
    assert "line1" in cleaned
    assert "line2" in cleaned


def test_overview_json_coerce_does_not_flatten_string_whitespace() -> None:
    raw_text = '{"summary":"line1\\n\\nline2"}'

    parsed = _coerce_json_output(result_data=None, raw_text=raw_text)

    assert isinstance(parsed, dict)
    assert parsed["summary"] == "line1\n\nline2"


def test_static_no_clean_whitespace_on_llm_result_text() -> None:
    pattern = re.compile(
        r"clean_whitespace\(\s*(?:str\(\s*)?(?:result|res)\.text\b",
        re.MULTILINE,
    )
    offenders: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        text = path.read_text(encoding="utf-8-sig")
        if pattern.search(text):
            offenders.append(str(path.relative_to(PROJECT_ROOT)))

    assert not offenders, f"forbidden clean_whitespace(result/res.text) in: {offenders}"


def test_static_no_duplicate_clean_whitespace_in_comprehensions() -> None:
    offenders: list[str] = []
    for path in SRC_ROOT.rglob("*.py"):
        source = path.read_text(encoding="utf-8-sig")
        tree = ast.parse(source)
        for node in ast.walk(tree):
            if not isinstance(node, ast.ListComp | ast.SetComp | ast.GeneratorExp):
                continue
            elt_expr = ast.unparse(node.elt)
            if "clean_whitespace(" not in elt_expr:
                continue
            conds = [
                ast.unparse(cond)
                for generator in node.generators
                for cond in generator.ifs
            ]
            if any("clean_whitespace(" in cond for cond in conds):
                offenders.append(f"{path.relative_to(PROJECT_ROOT)}:{node.lineno}")

    assert not offenders, f"duplicate clean_whitespace in comprehensions: {offenders}"
