from __future__ import annotations

import re

from serpsage.components.extract.markdown.types import StatsMap

_PUNCT_RE = re.compile(r"[,.!?;:\u3002\uff01\uff1f\uff1b]")
_FENCE_LINE_RE = re.compile(r"^\s*(`{3,})(.*)$")
_TABLE_SEP_RE = re.compile(r"^\s*\|?(?:\s*:?-{3,}:?\s*\|)+\s*:?-{3,}:?\s*\|?\s*$")
_ORDERED_LIST_RE = re.compile(r"^\s*\d+[.)]\s+")
_BULLET_LIST_RE = re.compile(r"^\s*[-*+]\s+")
_INLINE_CODE_RE = re.compile(r"(?<!`)`([^`\n]+?)`(?!`)")
_LINK_RE = re.compile(r"\[[^\]]+\]\([^)]+\)")


def score_candidate(
    *,
    markdown: str,
    plain_text: str,
    stats: StatsMap,
    warnings: list[str],
) -> float:
    if not plain_text.strip():
        return 0.0

    chars = len(plain_text)
    paragraphs = [ln for ln in markdown.split("\n\n") if ln.strip()]
    para_count = len(paragraphs)

    heading_count = int(stats.get("heading_count", 0))
    table_count = int(stats.get("table_count", 0))
    table_rows = int(stats.get("table_row_count", 0))
    code_count = int(stats.get("code_block_count", 0))
    inline_code_count = int(stats.get("inline_code_count", 0))
    ordered_list_count = int(stats.get("ordered_list_count", 0))
    link_count = int(stats.get("link_count", 0))
    fence_ok = bool(stats.get("fence_pairs_ok", True))

    punct_density = len(_PUNCT_RE.findall(plain_text)) / max(1, chars)
    link_density = float(link_count) / max(1.0, float(para_count))
    noise_penalty = min(0.25, float(len(warnings)) * 0.03)

    score = (
        min(1.0, chars / 3600.0) * 0.46
        + min(1.0, para_count / 26.0) * 0.15
        + min(1.0, punct_density * 120.0) * 0.10
        + min(1.0, heading_count / 8.0) * 0.08
        + min(1.0, table_count / 4.0) * 0.04
        + min(1.0, table_rows / 12.0) * 0.04
        + min(1.0, code_count / 4.0) * 0.04
        + min(1.0, inline_code_count / 12.0) * 0.03
        + min(1.0, ordered_list_count / 10.0) * 0.02
        - min(0.15, link_density / 8.0) * 0.04
        - (0.18 if not fence_ok else 0.0)
        - noise_penalty
    )
    return max(0.0, min(1.0, float(score)))


def infer_markdown_stats(markdown: str) -> dict[str, int | bool]:
    lines = markdown.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    heading_count = 0
    list_count = 0
    ordered_list_count = 0
    link_count = 0
    inline_code_count = 0

    in_code = False
    active_fence = ""
    code_block_count = 0
    open_block = False

    table_count = 0
    table_row_count = 0
    table_buf: list[str] = []

    for line in lines:
        stripped = line.strip()
        fence = _fence_delimiter(stripped)

        if in_code:
            if fence and len(fence) >= len(active_fence):
                in_code = False
                active_fence = ""
                open_block = False
            continue

        if fence:
            in_code = True
            active_fence = fence
            code_block_count += 1
            open_block = True
            continue

        if stripped.startswith("#"):
            heading_count += 1

        if _ORDERED_LIST_RE.match(line):
            ordered_list_count += 1
            list_count += 1
        elif _BULLET_LIST_RE.match(line):
            list_count += 1

        link_count += len(_LINK_RE.findall(line))
        inline_code_count += len(_INLINE_CODE_RE.findall(line))

        if _is_table_line(line):
            table_buf.append(line)
            continue

        if table_buf:
            tc, rows = _finalize_table_block(table_buf)
            table_count += tc
            table_row_count += rows
            table_buf = []

    if table_buf:
        tc, rows = _finalize_table_block(table_buf)
        table_count += tc
        table_row_count += rows

    return {
        "heading_count": int(heading_count),
        "table_count": int(table_count),
        "table_row_count": int(table_row_count),
        "code_block_count": int(code_block_count),
        "inline_code_count": int(inline_code_count),
        "list_count": int(list_count),
        "ordered_list_count": int(ordered_list_count),
        "link_count": int(link_count),
        "fence_pairs_ok": bool(not in_code and not open_block),
    }


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
    return stripped.count("|") >= 2 or _TABLE_SEP_RE.match(stripped) is not None


def _finalize_table_block(lines: list[str]) -> tuple[int, int]:
    if len(lines) < 2:
        return 0, 0
    has_sep = any(_TABLE_SEP_RE.match(line.strip()) for line in lines)
    if not has_sep:
        return 0, 0
    body_rows = max(0, len(lines) - 2)
    return 1, body_rows


__all__ = ["infer_markdown_stats", "score_candidate"]
