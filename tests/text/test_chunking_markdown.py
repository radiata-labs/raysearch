from __future__ import annotations

from serpsage.text.chunking import chunk_segments, markdown_to_segments


def test_markdown_to_segments_preserves_structure() -> None:
    md = (
        "# Title\n\n"
        "This is paragraph one.\n\n"
        "- item a\n"
        "- item b\n\n"
        "> quote line\n\n"
        "```python\nprint('x')\n```\n"
    )
    segs = markdown_to_segments(
        md,
        max_markdown_chars=10_000,
        max_segments=100,
        max_sentence_chars=120,
    )
    kinds = [s.kind for s in segs]
    assert "heading" in kinds
    assert "list" in kinds
    assert "quote" in kinds
    assert "code" in kinds


def test_chunk_segments_with_overlap() -> None:
    md = "\n\n".join(f"Paragraph {i}: " + ("x " * 50) for i in range(1, 7))
    segs = markdown_to_segments(
        md,
        max_markdown_chars=10_000,
        max_segments=100,
        max_sentence_chars=600,
    )
    chunks = chunk_segments(
        segs,
        target_chars=260,
        overlap_segments=1,
        min_chunk_chars=80,
    )
    assert len(chunks) >= 2
    assert all(len(c) >= 80 for c in chunks)

