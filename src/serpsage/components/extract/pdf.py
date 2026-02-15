from __future__ import annotations

import re
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO

from pypdf import PdfReader

from serpsage.models.extract import ExtractedDocument
from serpsage.utils import clean_whitespace

try:
    import fitz  # type: ignore[import-not-found]

    PYMUPDF_AVAILABLE = True
except Exception:  # noqa: BLE001
    fitz = None
    PYMUPDF_AVAILABLE = False

_LINE_SPLIT_RE = re.compile(r"\r?\n+")


def extract_pdf_document(*, content: bytes) -> ExtractedDocument:
    if not content:
        return ExtractedDocument(
            content_kind="pdf",
            extractor_used="pdf:none",
            quality_score=0.0,
            warnings=["empty pdf content"],
        )

    warnings: list[str] = []
    pages_pypdf: list[list[str]] = []
    pages_pymupdf: list[list[str]] = []

    try:
        pages_pypdf = _extract_lines_pypdf(content)
    except Exception as exc:  # noqa: BLE001
        warnings.append(f"pypdf_failed:{type(exc).__name__}")

    if PYMUPDF_AVAILABLE:
        try:
            pages_pymupdf = _extract_lines_pymupdf(content)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"pymupdf_failed:{type(exc).__name__}")

    if not pages_pypdf and not pages_pymupdf:
        return ExtractedDocument(
            content_kind="pdf",
            extractor_used="pdf:none",
            quality_score=0.0,
            warnings=warnings or ["pdf extraction failed"],
        )

    chosen_pages, engine = _pick_pages(pages_pypdf, pages_pymupdf)
    header_footer = _detect_repeated_header_footer(chosen_pages)

    md_parts: list[str] = []
    plain_parts: list[str] = []
    pages_kept = 0
    for idx, lines in enumerate(chosen_pages, 1):
        clean_lines = [ln for ln in lines if ln and ln not in header_footer]
        paras = _merge_paragraph_lines(clean_lines)
        if not paras:
            continue
        pages_kept += 1
        md_parts.append(f"## Page {idx}")
        md_parts.extend(paras)
        plain_parts.extend(paras)

    markdown = "\n\n".join(md_parts).strip()
    plain_text = "\n\n".join(plain_parts).strip()
    text_chars = len(plain_text)
    if text_chars < 180:
        warnings.append("very_low_text_pdf; likely scanned or encrypted")
    quality = min(1.0, text_chars / 3200.0)

    return ExtractedDocument(
        markdown=markdown,
        plain_text=plain_text,
        title="",
        content_kind="pdf",
        extractor_used=f"pdf:{engine}",
        quality_score=float(quality),
        warnings=warnings,
        stats={
            "pages_total": len(chosen_pages),
            "pages_kept": pages_kept,
            "chars": text_chars,
            "engine": engine,
        },
    )


def _extract_lines_pypdf(content: bytes) -> list[list[str]]:
    reader = PdfReader(BytesIO(content))
    pages = list(reader.pages)
    if not pages:
        return []

    def _read_page(i: int) -> tuple[int, list[str]]:
        page = pages[i]
        raw = page.extract_text() or ""
        lines = [clean_whitespace(x) for x in _LINE_SPLIT_RE.split(raw) if x.strip()]
        return i, lines

    if len(pages) <= 3:
        return [_read_page(i)[1] for i in range(len(pages))]

    out: list[list[str]] = [[] for _ in range(len(pages))]
    workers = min(6, len(pages))
    with ThreadPoolExecutor(max_workers=max(2, workers)) as pool:
        for idx, lines in pool.map(_read_page, range(len(pages))):
            out[idx] = lines
    return out


def _extract_lines_pymupdf(content: bytes) -> list[list[str]]:
    if not PYMUPDF_AVAILABLE or fitz is None:
        return []
    doc = fitz.open(stream=content, filetype="pdf")
    pages: list[list[str]] = []
    for page in doc:
        raw = page.get_text("text") or ""
        lines = [clean_whitespace(x) for x in _LINE_SPLIT_RE.split(raw) if x.strip()]
        pages.append(lines)
    doc.close()
    return pages


def _pick_pages(
    pages_pypdf: list[list[str]],
    pages_pymupdf: list[list[str]],
) -> tuple[list[list[str]], str]:
    chars_pypdf = sum(len(" ".join(p)) for p in pages_pypdf)
    chars_pymupdf = sum(len(" ".join(p)) for p in pages_pymupdf)
    if chars_pypdf >= chars_pymupdf and pages_pypdf:
        return pages_pypdf, "pypdf"
    if pages_pymupdf:
        return pages_pymupdf, "pymupdf"
    return pages_pypdf, "pypdf"


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
    threshold = max(2, int(len(pages) * 0.45))
    return {
        line
        for line, cnt in counts.items()
        if cnt >= threshold and len(line) < 140 and not _looks_like_heading(line)
    }


def _merge_paragraph_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    buf = ""
    for line in lines:
        line = _normalize_pdf_line(line)
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
        if _is_sentence_end(buf):
            out.append(buf.strip())
            buf = line
        else:
            buf = f"{buf} {line}"
    if buf:
        out.append(buf.strip())
    return [x for x in _dedupe_lines(out) if x]


def _normalize_pdf_line(line: str) -> str:
    line = clean_whitespace(line)
    if not line:
        return ""
    line = line.replace("\u00ad", "")
    line = re.sub(r"\s+", " ", line)
    return line.strip()


def _dedupe_lines(lines: list[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for line in lines:
        norm = line.lower().strip()
        if len(norm) > 12 and norm in seen:
            continue
        seen.add(norm)
        out.append(line)
    return out


def _looks_like_heading(line: str) -> bool:
    if len(line) > 120:
        return False
    if line.endswith(":"):
        return True
    words = line.split()
    if not words:
        return False
    if len(words) <= 12 and line == line.upper():
        return True
    if len(words) <= 10 and line == line.title():
        return True
    return bool(re.match(r"^\d+(\.\d+){0,3}\s+\w+", line))


def _is_list_item(line: str) -> bool:
    return bool(re.match(r"^(\d+[\.\)]|[-*•])\s+", line))


def _is_sentence_end(text: str) -> bool:
    return text.rstrip().endswith(
        (".", "!", "?", ";", ":", "\u3002", "\uff01", "\uff1f")
    )


__all__ = ["extract_pdf_document"]
