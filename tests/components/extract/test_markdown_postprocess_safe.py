from __future__ import annotations

from serpsage.components.extract.markdown.postprocess import (
    finalize_markdown,
    merge_markdown,
)


def _fences_balanced(markdown: str) -> bool:
    in_code = False
    active_len = 0
    for line in markdown.splitlines():
        stripped = line.strip()
        if not stripped.startswith("```"):
            continue
        ticks = len(stripped) - len(stripped.lstrip("`"))
        if not in_code:
            in_code = True
            active_len = ticks
            continue
        if ticks >= active_len:
            in_code = False
            active_len = 0
    return not in_code


def test_finalize_markdown_keeps_code_fence_balanced_after_clip() -> None:
    prefix = "\n\n".join(f"paragraph {i}" for i in range(24))
    markdown = (
        f"{prefix}\n\n"
        "```python\n"
        "line1\n"
        "line2\n"
        "line3\n"
        "```\n\n"
        "tail paragraph"
    )

    clipped = finalize_markdown(markdown=markdown, max_chars=160)

    assert _fences_balanced(clipped)


def test_finalize_markdown_does_not_dedupe_code_lines() -> None:
    markdown = """```python
print('same')
print('same')
```
"""
    out = finalize_markdown(markdown=markdown, max_chars=500)

    assert "print('same')\nprint('same')" in out
    assert _fences_balanced(out)


def test_merge_markdown_preserves_table_and_fence_integrity() -> None:
    base = "| Key | Value |\n| --- | --- |\n| retries | 3 |"
    extra = "```bash\necho ok\n```"
    merged = merge_markdown(base=base, extra=extra, max_chars=1000)

    assert "| --- | --- |" in merged
    assert "```bash" in merged
    assert _fences_balanced(merged)
