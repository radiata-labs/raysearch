from __future__ import annotations

from serpsage.app.bootstrap import build_runtime
from serpsage.settings.models import AppSettings
from serpsage.steps.fetch.abstracts import FetchAbstractBuildStep


def _build_step() -> FetchAbstractBuildStep:
    rt = build_runtime(settings=AppSettings())
    return FetchAbstractBuildStep(rt=rt)


def test_split_line_sentences_keeps_decimal_and_splits_english_dot_on_space() -> None:
    step = _build_step()
    text = "DeepSeek V3.2 is stable. It improves tools."
    out = step._split_line_sentences(text)

    assert out == ["DeepSeek V3.2 is stable.", "It improves tools."]


def test_split_line_sentences_keeps_url_and_pdf_path() -> None:
    step = _build_step()
    text = (
        "See the technical report at "
        "https://modelscope.cn/models/deepseek-ai/DeepSeek-V3.2/resolve/master/assets/paper.pdf"
    )
    out = step._split_line_sentences(text)

    assert out == [text]


def test_split_line_sentences_splits_cjk_sentence_end_without_space() -> None:
    step = _build_step()
    text = "\u8fd9\u662f\u7b2c\u4e00\u53e5\u3002\u8fd9\u662f\u7b2c\u4e8c\u53e5\u3002"
    out = step._split_line_sentences(text)

    assert out == [
        "\u8fd9\u662f\u7b2c\u4e00\u53e5\u3002",
        "\u8fd9\u662f\u7b2c\u4e8c\u53e5\u3002",
    ]


def test_extract_abstracts_does_not_emit_v3_dot_fragments() -> None:
    step = _build_step()
    markdown = (
        "Today we released two models: DeepSeek-V3.2 and DeepSeek-V3.2-Speciale.\n"
        "Technical report: "
        "<https://modelscope.cn/models/deepseek-ai/DeepSeek-V3.2/resolve/master/assets/paper.pdf>"
    )

    out = step._extract_abstracts(
        markdown=markdown,
        max_abstracts=10,
        min_abstract_chars=1,
    )
    texts = [item.text for item in out]

    assert "DeepSeek-V3." not in texts
    assert "2-Speciale is DeepSeek-V3." not in texts
    assert any(
        "DeepSeek-V3.2 and DeepSeek-V3.2-Speciale." in item for item in texts
    )
