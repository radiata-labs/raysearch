from __future__ import annotations

import re

from serpsage.utils import clean_whitespace

_MD_PREFIX_RE = re.compile(r"^(\s*[-*+]\s+|\s*\d+[.)]\s+|#{1,6}\s+|>\s+)")
_NOISE_LINE_RE = re.compile(
    r"(privacy policy|cookie policy|terms of service|all rights reserved|"
    r"sign up|subscribe|advertisement|sponsored content|related posts)",
    re.IGNORECASE,
)
_FENCE_LINE_RE = re.compile(r"^\s*(`{3,})(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
_SPECIAL_BLOCK_RE = re.compile(r"^\s*(#{1,6}\s+|>|[-*+]\s+|\d+[.)]\s+|\|)")
_IMAGE_RE = re.compile(r"!\[[^\]]*]\([^)]+\)")
_MD_LINK_RE = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_AUTO_URL_RE = re.compile(r"<https?://[^>\s]+>", re.IGNORECASE)
_BARE_URL_RE = re.compile(r"(?<!<)https?://\S+", re.IGNORECASE)
_ABSTRACT_NOISE_RE = re.compile(
    r"^\s*(跳到主要内容|skip to main content)\s*$", re.IGNORECASE
)
_SETEXT_HEADING_LINE_RE = re.compile(r"^\s*[=-]{2,}\s*$")
_HR_LINE_RE = re.compile(r"^\s*(?:[-*_]\s*){3,}\s*$")


def finalize_markdown(*, markdown: str, max_chars: int) -> str:
    if not markdown:
        return ""
    source_lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    out: list[str] = []
    paragraph_buf: list[str] = []
    seen_paragraphs: set[str] = set()
    table_buf: list[str] = []
    mode = "normal"
    active_fence = ""

    def append_blank_once() -> None:
        if out and out[-1] != "":
            out.append("")

    def flush_paragraph() -> None:
        nonlocal paragraph_buf
        if not paragraph_buf:
            return
        block = "\n".join(paragraph_buf).strip()
        paragraph_buf = []
        if not block:
            return
        key = _paragraph_key(block)
        if key and key in seen_paragraphs:
            return
        if key:
            seen_paragraphs.add(key)
        out.append(block)

    def flush_table() -> None:
        nonlocal table_buf
        if not table_buf:
            return
        table_block = "\n".join(table_buf).strip("\n")
        table_buf = []
        if table_block:
            out.append(table_block)

    for raw_line in source_lines:
        fence = _fence_delimiter(raw_line)
        if mode == "fenced_code":
            out.append(raw_line)
            if fence and len(fence) >= len(active_fence):
                mode = "normal"
                active_fence = ""
            continue
        if mode == "table":
            if _is_table_line(raw_line):
                table_buf.append(_normalize_table_line(raw_line))
                continue
            flush_table()
            mode = "normal"
        if fence:
            flush_paragraph()
            append_blank_once()
            out.append(raw_line.rstrip())
            mode = "fenced_code"
            active_fence = fence
            continue
        if _is_table_line(raw_line):
            flush_paragraph()
            append_blank_once()
            mode = "table"
            table_buf.append(_normalize_table_line(raw_line))
            continue
        normalized = _normalize_normal_line(raw_line)
        if not normalized:
            flush_paragraph()
            append_blank_once()
            continue
        if _NOISE_LINE_RE.search(_collapse_ws_outside_inline_code(normalized).strip()):
            continue
        if _SPECIAL_BLOCK_RE.match(normalized):
            flush_paragraph()
            out.append(normalized)
            continue
        paragraph_buf.append(normalized)
    flush_table()
    flush_paragraph()
    compact = _compact_blank_lines(out)
    result = "\n".join(compact).strip()
    if len(result) <= int(max_chars):
        return result
    return _clip_with_structure(result=result, max_chars=int(max_chars))


def markdown_to_text(markdown: str) -> str:
    out: list[str] = []
    in_code = False
    active_fence = ""
    for raw in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        fence = _fence_delimiter(raw)
        if fence:
            if not in_code:
                in_code = True
                active_fence = fence
            elif len(fence) >= len(active_fence):
                in_code = False
                active_fence = ""
            continue
        if in_code:
            if raw:
                out.append(raw)
            continue
        line = _MD_PREFIX_RE.sub("", raw.strip())
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = re.sub(r"\[([^\]]+)\]\(([^)]+)\)", r"\1", line)
        line = line.replace("|", " ")
        line = clean_whitespace(line)
        if line:
            out.append(line)
    return "\n".join(out).strip()


def markdown_to_abstract_text(markdown: str) -> str:
    out: list[str] = []
    in_code = False
    active_fence = ""
    for raw in markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        fence = _fence_delimiter(raw)
        if fence:
            if not in_code:
                in_code = True
                active_fence = fence
            elif len(fence) >= len(active_fence):
                in_code = False
                active_fence = ""
            continue
        if in_code:
            continue
        line = raw.strip()
        if not line:
            continue
        if _ABSTRACT_NOISE_RE.match(line):
            continue
        if _SETEXT_HEADING_LINE_RE.match(line):
            continue
        if _HR_LINE_RE.match(line):
            continue
        if _TABLE_SEP_RE.match(line):
            continue
        line = _MD_PREFIX_RE.sub("", line)
        line = re.sub(r"`([^`]+)`", r"\1", line)
        line = _IMAGE_RE.sub("", line)
        line = _MD_LINK_RE.sub(r"\1", line)
        line = _AUTO_URL_RE.sub("", line)
        line = _BARE_URL_RE.sub("", line)
        line = line.replace("\\*", "*").replace("\\_", "_")
        line = line.replace("**", "").replace("__", "")
        line = line.replace("|", " ")
        line = clean_whitespace(line)
        if not line:
            continue
        if _ABSTRACT_NOISE_RE.match(line):
            continue
        out.append(line)
    return "\n".join(out).strip()


def merge_markdown(*, base: str, extra: str, max_chars: int) -> str:
    if not base.strip():
        return finalize_markdown(markdown=extra, max_chars=max_chars)
    if not extra.strip():
        return finalize_markdown(markdown=base, max_chars=max_chars)
    merged = f"{base.rstrip()}\n\n{extra.lstrip()}"
    return finalize_markdown(markdown=merged, max_chars=max_chars)


def extract_feature_snippets(*, markdown: str, feature: str) -> str:
    if not markdown.strip():
        return ""
    if feature == "code_block_count":
        snippets = _extract_fenced_blocks(markdown)
        return "\n\n".join(snippets[:6]).strip()
    if feature == "table_count":
        blocks: list[str] = []
        current: list[str] = []
        for line in markdown.splitlines():
            if _is_table_line(line):
                current.append(line.rstrip())
                continue
            if current:
                block = "\n".join(current).strip()
                if _table_has_separator(current):
                    blocks.append(block)
                current = []
        if current:
            block = "\n".join(current).strip()
            if _table_has_separator(current):
                blocks.append(block)
        return "\n\n".join(blocks[:6]).strip()
    if feature == "heading_count":
        heads = [
            ln.strip() for ln in markdown.splitlines() if ln.strip().startswith("#")
        ]
        return "\n".join(heads[:14]).strip()
    if feature == "ordered_list_count":
        items = [
            ln.rstrip()
            for ln in markdown.splitlines()
            if re.match(r"^\s*\d+[.)]\s+", ln)
        ]
        return "\n".join(items[:24]).strip()
    return ""


def _paragraph_key(block: str) -> str:
    compact = _collapse_ws_outside_inline_code(block).lower().strip()
    if len(compact) <= 10:
        return ""
    return compact


def _normalize_normal_line(line: str) -> str:
    if not line.strip():
        return ""
    leading = line[: len(line) - len(line.lstrip(" \t"))]
    leading = leading.replace("\t", "    ")
    body = _collapse_ws_outside_inline_code(line[len(leading) :]).strip()
    if not body:
        return ""
    if _SPECIAL_BLOCK_RE.match(body):
        return f"{leading}{body}".rstrip()
    return body


def _normalize_table_line(line: str) -> str:
    return line.rstrip()


def _compact_blank_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    blank = 0
    for line in lines:
        if line == "":
            blank += 1
            if blank <= 1:
                out.append(line)
            continue
        blank = 0
        out.append(line)
    return out


def _fence_delimiter(line: str) -> str | None:
    match = _FENCE_LINE_RE.match(line)
    if match is None:
        return None
    return match.group(1)


def _is_table_line(line: str) -> bool:
    stripped = line.strip()
    if not stripped:
        return False
    if "|" not in stripped:
        return False
    if _TABLE_SEP_RE.match(stripped):
        return True
    return stripped.count("|") >= 2


def _table_has_separator(lines: list[str]) -> bool:
    return any(_TABLE_SEP_RE.match(line.strip()) for line in lines)


def _collapse_ws_outside_inline_code(text: str) -> str:
    out: list[str] = []
    in_code = False
    code_fence_len = 0
    idx = 0
    while idx < len(text):
        ch = text[idx]
        if ch == "`":
            run = 1
            while idx + run < len(text) and text[idx + run] == "`":
                run += 1
            ticks = "`" * run
            out.append(ticks)
            if not in_code:
                in_code = True
                code_fence_len = run
            elif run >= code_fence_len:
                in_code = False
                code_fence_len = 0
            idx += run
            continue
        if in_code:
            out.append(ch)
            idx += 1
            continue
        if ch.isspace():
            if not out or out[-1] != " ":
                out.append(" ")
            idx += 1
            continue
        out.append(ch)
        idx += 1
    return "".join(out)


def _extract_fenced_blocks(markdown: str) -> list[str]:
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    blocks: list[str] = []
    buf: list[str] = []
    active_fence = ""
    for line in lines:
        fence = _fence_delimiter(line)
        if active_fence:
            buf.append(line)
            if fence and len(fence) >= len(active_fence):
                blocks.append("\n".join(buf).strip())
                buf = []
                active_fence = ""
            continue
        if fence:
            active_fence = fence
            buf = [line]
    return [b for b in blocks if b]


def _clip_with_structure(*, result: str, max_chars: int) -> str:
    if len(result) <= max_chars:
        return result
    lines = result.splitlines(keepends=True)
    total = 0
    last_safe = 0
    open_fence = ""
    for line in lines:
        next_total = total + len(line)
        stripped = line.strip()
        fence = _fence_delimiter(stripped)
        if fence:
            if not open_fence:
                open_fence = fence
            elif len(fence) >= len(open_fence):
                open_fence = ""
        if next_total <= max_chars and not open_fence:
            if (
                stripped == ""
                or stripped.startswith(("#", "|"))
                or bool(re.match(r"^[-*+]\s+", stripped))
                or bool(re.match(r"^\d+[.)]\s+", stripped))
            ):
                last_safe = next_total
        if next_total > max_chars:
            break
        total = next_total
    cut = max_chars
    if last_safe > int(max_chars * 0.55):
        cut = last_safe
    clipped = result[:cut].rstrip()
    unclosed_start = _find_unclosed_fence_start(clipped)
    if unclosed_start >= 0:
        clipped = clipped[:unclosed_start].rstrip()
    return clipped


def _find_unclosed_fence_start(markdown: str) -> int:
    offset = 0
    open_fence = ""
    open_pos = -1
    for line in markdown.splitlines(keepends=True):
        stripped = line.strip()
        fence = _fence_delimiter(stripped)
        if fence:
            if not open_fence:
                open_fence = fence
                open_pos = offset
            elif len(fence) >= len(open_fence):
                open_fence = ""
                open_pos = -1
        offset += len(line)
    return open_pos


__all__ = [
    "extract_feature_snippets",
    "finalize_markdown",
    "markdown_to_abstract_text",
    "markdown_to_text",
    "merge_markdown",
]
