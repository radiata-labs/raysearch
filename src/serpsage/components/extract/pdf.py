from __future__ import annotations

import re
from collections import Counter
from io import BytesIO

from pypdf import PdfReader

from serpsage.models.extract import ExtractedDocument
from serpsage.text.normalize import clean_whitespace

_LINE_SPLIT_RE = re.compile(r"\r?\n+")


def extract_pdf_document(*, content: bytes) -> ExtractedDocument:
    if not content:
        return ExtractedDocument(content_kind="pdf")

    reader = PdfReader(BytesIO(content))
    pages: list[list[str]] = []
    for page in reader.pages:
        raw = page.extract_text() or ""
        lines = [clean_whitespace(x) for x in _LINE_SPLIT_RE.split(raw) if x.strip()]
        pages.append(lines)

    header_footer = _detect_repeated_header_footer(pages)

    md_parts: list[str] = []
    plain_parts: list[str] = []
    for idx, lines in enumerate(pages, 1):
        clean_lines = [ln for ln in lines if ln not in header_footer]
        paras = _merge_paragraph_lines(clean_lines)
        if not paras:
            continue
        md_parts.append(f"## Page {idx}")
        md_parts.extend(paras)
        plain_parts.extend(paras)

    markdown = "\n\n".join(md_parts).strip()
    plain_text = "\n\n".join(plain_parts).strip()
    return ExtractedDocument(
        markdown=markdown,
        plain_text=plain_text,
        title="",
        content_kind="pdf",
        stats={
            "pages_total": len(pages),
            "pages_kept": len([1 for p in pages if p]),
            "chars": len(plain_text),
        },
    )


def _detect_repeated_header_footer(pages: list[list[str]]) -> set[str]:
    if len(pages) < 2:
        return set()
    cand: list[str] = []
    for lines in pages:
        if not lines:
            continue
        cand.extend(lines[:2])
        cand.extend(lines[-2:])
    counts = Counter(cand)
    threshold = max(2, int(len(pages) * 0.5))
    return {
        line for line, cnt in counts.items() if cnt >= threshold and len(line) < 120
    }


def _merge_paragraph_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    buf = ""
    for line in lines:
        if not line:
            continue
        if buf.endswith("-"):
            buf = f"{buf[:-1]}{line.lstrip()}"
            continue
        if _is_list_item(line):
            if buf:
                out.append(buf.strip())
                buf = ""
            out.append(line)
            continue
        if _looks_like_heading(line):
            if buf:
                out.append(buf.strip())
                buf = ""
            out.append(f"### {line}")
            continue
        if not buf:
            buf = line
            continue
        if _is_sentence_end(buf) or line[:1].isupper():
            out.append(buf.strip())
            buf = line
        else:
            buf = f"{buf} {line}"
    if buf:
        out.append(buf.strip())
    return [x for x in out if x]


def _looks_like_heading(line: str) -> bool:
    if len(line) > 100:
        return False
    if line.endswith(":"):
        return True
    words = line.split()
    return bool(0 < len(words) <= 8 and line == line.title())


def _is_list_item(line: str) -> bool:
    return bool(re.match(r"^(\d+[\.\)]|[-*•])\s+", line))


def _is_sentence_end(text: str) -> bool:
    return text.rstrip().endswith((".", "!", "?", "。", "！", "？", ":", "："))


__all__ = ["extract_pdf_document"]
