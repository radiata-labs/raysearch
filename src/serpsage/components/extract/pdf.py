from __future__ import annotations

import contextlib
import io
import re
from collections import Counter
from dataclasses import dataclass
from io import BytesIO
from typing import cast
from typing_extensions import override

import anyio
import fitz
import pymupdf4llm
from anyio import to_thread
from pypdf import PdfReader
from pypdf.errors import PdfReadError, PdfStreamError

from serpsage.components.extract.base import ExtractorBase
from serpsage.components.extract.html.postprocess import (
    finalize_markdown,
    markdown_to_abstract_text,
    markdown_to_text,
)
from serpsage.components.fetch.utils import classify_content_kind
from serpsage.core.runtime import Runtime
from serpsage.models.extract import ExtractContentOptions, ExtractedDocument
from serpsage.utils import clean_whitespace

_LINE_SPLIT_RE = re.compile(r"\r?\n+")
_PAGE_SEPARATOR = "\n\n---\n\n"


@dataclass(slots=True)
class PdfExtractionResult:
    markdown: str
    text_chars: int
    pages_total: int
    pages_kept: int
    title: str
    author: str
    published_date: str
    stats: dict[str, int | float | str | bool]


class PdfExtractor(ExtractorBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def extract(
        self,
        *,
        url: str,
        content: bytes,
        content_type: str | None,
        content_options: ExtractContentOptions | None = None,
        include_secondary_content: bool = False,
        collect_links: bool = False,
        collect_images: bool = False,
    ) -> ExtractedDocument:
        del (
            url,
            content_options,
            include_secondary_content,
            collect_links,
            collect_images,
        )
        kind = classify_content_kind(
            content_type=content_type,
            url="https://pdf.local/document.pdf",
            content=content,
        )
        if kind != "pdf":
            raise ValueError("PdfExtractor only handles PDF content")
        if not content:
            return self._attach_abstract_markdown(
                ExtractedDocument(
                    content_kind="pdf",
                    extractor_used="pdf:none",
                    warnings=["empty pdf content"],
                )
            )

        warnings: list[str] = []
        try:
            extracted = await self._extract_pymupdf4llm(content)
            warnings.extend(self._quality_warnings(extracted.text_chars))
            return self._attach_abstract_markdown(
                ExtractedDocument(
                    markdown=extracted.markdown,
                    title=extracted.title,
                    published_date=extracted.published_date,
                    author=extracted.author,
                    content_kind="pdf",
                    extractor_used="pdf:pymupdf4llm",
                    warnings=warnings,
                    stats=extracted.stats | {"engine": "pymupdf4llm"},
                )
            )
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"pymupdf4llm_failed:{type(exc).__name__}")

        pages_pypdf: list[list[str]] = []
        pages_pymupdf: list[list[str]] = []
        try:
            pages_pypdf = await self._extract_lines_pypdf(content)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"pypdf_failed:{type(exc).__name__}")
        try:
            pages_pymupdf = await self._extract_lines_pymupdf(content)
        except Exception as exc:  # noqa: BLE001
            warnings.append(f"pymupdf_failed:{type(exc).__name__}")
        if not pages_pypdf and not pages_pymupdf:
            return self._attach_abstract_markdown(
                ExtractedDocument(
                    content_kind="pdf",
                    extractor_used="pdf:none",
                    warnings=warnings or ["pdf extraction failed"],
                )
            )

        markdown, text_chars, pages_kept, pages_total, engine = (
            self._build_fallback_markdown(
                pages_pypdf=pages_pypdf,
                pages_pymupdf=pages_pymupdf,
            )
        )
        warnings.extend(self._quality_warnings(text_chars))
        metadata = await self._extract_pdf_metadata(content)
        return self._attach_abstract_markdown(
            ExtractedDocument(
                markdown=markdown,
                title=metadata["title"],
                published_date=metadata["published_date"],
                author=metadata["author"],
                content_kind="pdf",
                extractor_used=f"pdf:{engine}",
                warnings=warnings,
                stats={
                    "pages_total": pages_total,
                    "pages_kept": pages_kept,
                    "text_chars": text_chars,
                    "engine": engine,
                },
            )
        )

    def _attach_abstract_markdown(self, doc: ExtractedDocument) -> ExtractedDocument:
        return doc.model_copy(
            update={
                "md_for_abstract": markdown_to_abstract_text(str(doc.markdown or ""))
            }
        )

    async def _extract_pymupdf4llm(self, content: bytes) -> PdfExtractionResult:
        def _do_extract() -> PdfExtractionResult:
            with fitz.open(stream=content, filetype="pdf") as doc:
                raw_metadata = dict(doc.metadata or {})
                page_chunks = self._run_pymupdf4llm(doc)
            page_texts = [
                self._normalize_markdown_chunk(str(chunk.get("text", "") or ""))
                for chunk in page_chunks
            ]
            kept_chunks = [text for text in page_texts if text]
            markdown = finalize_markdown(
                markdown=_PAGE_SEPARATOR.join(kept_chunks).strip(),
                max_chars=max(8_000, self.settings.fetch.extract.max_markdown_chars),
            )
            text_chars = len(markdown_to_text(markdown))
            pages_total = len(page_chunks)
            pages_kept = len(kept_chunks)
            stats: dict[str, int | float | str | bool] = {
                "pages_total": pages_total,
                "pages_kept": pages_kept,
                "text_chars": text_chars,
                "table_count": sum(
                    len(cast("list[object]", chunk.get("tables", [])))
                    for chunk in page_chunks
                    if isinstance(chunk, dict)
                ),
                "image_count": sum(
                    len(cast("list[object]", chunk.get("images", [])))
                    for chunk in page_chunks
                    if isinstance(chunk, dict)
                ),
                "graphics_count": sum(
                    len(cast("list[object]", chunk.get("graphics", [])))
                    for chunk in page_chunks
                    if isinstance(chunk, dict)
                ),
            }
            return PdfExtractionResult(
                markdown=markdown,
                text_chars=text_chars,
                pages_total=pages_total,
                pages_kept=pages_kept,
                title=clean_whitespace(str(raw_metadata.get("title") or "")),
                author=clean_whitespace(str(raw_metadata.get("author") or "")),
                published_date=self._normalize_pdf_date(
                    str(
                        raw_metadata.get("creationDate")
                        or raw_metadata.get("modDate")
                        or ""
                    )
                ),
                stats=stats,
            )

        return await to_thread.run_sync(_do_extract)

    def _run_pymupdf4llm(self, doc: fitz.Document) -> list[dict[str, object]]:
        with contextlib.redirect_stdout(io.StringIO()):
            chunks = pymupdf4llm.to_markdown(
                doc,
                page_chunks=True,
                write_images=False,
                embed_images=False,
                ignore_images=True,
                ignore_graphics=False,
                force_text=True,
                page_separators=False,
                show_progress=False,
            )
        if not isinstance(chunks, list):
            raise TypeError("pymupdf4llm did not return page chunks")
        return [chunk for chunk in chunks if isinstance(chunk, dict)]

    async def _extract_pdf_metadata(self, content: bytes) -> dict[str, str]:
        def _do_extract() -> dict[str, str]:
            with fitz.open(stream=content, filetype="pdf") as doc:
                metadata = dict(doc.metadata or {})
            return {
                "title": clean_whitespace(str(metadata.get("title") or "")),
                "author": clean_whitespace(str(metadata.get("author") or "")),
                "published_date": self._normalize_pdf_date(
                    str(metadata.get("creationDate") or metadata.get("modDate") or "")
                ),
            }

        return await to_thread.run_sync(_do_extract)

    async def _extract_lines_pypdf(
        self,
        content: bytes,
        *,
        timeout_per_page: float = 5.0,
        total_timeout: float = 30.0,
    ) -> list[list[str]]:
        try:
            reader = PdfReader(BytesIO(content))
            pages = list(reader.pages)
        except (PdfReadError, PdfStreamError) as exc:
            raise ValueError(f"Invalid or corrupted PDF: {type(exc).__name__}") from exc
        except (KeyboardInterrupt, SystemExit):
            raise
        except Exception as exc:
            raise ValueError(f"Failed to read PDF: {type(exc).__name__}") from exc
        if not pages:
            return []

        def _read_page(i: int) -> tuple[int, list[str]]:
            page = pages[i]
            raw = str(page.extract_text() or "")
            lines = [
                clean_whitespace(x) for x in _LINE_SPLIT_RE.split(raw) if x.strip()
            ]
            return i, lines

        if len(pages) <= 3:
            results = [
                await to_thread.run_sync(_read_page, i) for i in range(len(pages))
            ]
            return [r[1] for r in results]

        out: list[list[str]] = [[] for _ in range(len(pages))]
        results_dict: dict[int, list[str]] = {}
        try:
            with anyio.fail_after(total_timeout):
                async with anyio.create_task_group() as tg:

                    async def extract_page(i: int) -> None:
                        idx, lines = await to_thread.run_sync(_read_page, i)
                        results_dict[idx] = lines

                    for i in range(len(pages)):
                        tg.start_soon(extract_page, i)
            for idx, lines in results_dict.items():
                out[idx] = lines
        except TimeoutError:
            for i in range(len(pages)):
                try:
                    with anyio.fail_after(timeout_per_page):
                        idx, lines = await to_thread.run_sync(_read_page, i)
                        out[idx] = lines
                except (TimeoutError, Exception):  # noqa: BLE001
                    out[i] = []
        except Exception:  # noqa: BLE001
            for i in range(len(pages)):
                try:
                    with anyio.fail_after(timeout_per_page):
                        idx, lines = await to_thread.run_sync(_read_page, i)
                        out[idx] = lines
                except (TimeoutError, Exception):  # noqa: BLE001
                    out[i] = []
        return out

    async def _extract_lines_pymupdf(self, content: bytes) -> list[list[str]]:
        def _do_extract() -> list[list[str]]:
            with fitz.open(stream=content, filetype="pdf") as doc:
                pages: list[list[str]] = []
                for page in doc:
                    raw = str(page.get_text("text") or "")
                    lines = [
                        clean_whitespace(x)
                        for x in _LINE_SPLIT_RE.split(raw)
                        if x.strip()
                    ]
                    pages.append(lines)
            return pages

        return await to_thread.run_sync(_do_extract)

    def _build_fallback_markdown(
        self,
        *,
        pages_pypdf: list[list[str]],
        pages_pymupdf: list[list[str]],
    ) -> tuple[str, int, int, int, str]:
        chosen_pages, engine = self._pick_pages(pages_pypdf, pages_pymupdf)
        header_footer = self._detect_repeated_header_footer(chosen_pages)
        md_parts: list[str] = []
        text_parts: list[str] = []
        pages_kept = 0
        for idx, lines in enumerate(chosen_pages, 1):
            clean_lines = [ln for ln in lines if ln and ln not in header_footer]
            paras = self._merge_paragraph_lines(clean_lines)
            if not paras:
                continue
            pages_kept += 1
            md_parts.append(f"## Page {idx}")
            md_parts.extend(paras)
            text_parts.extend(paras)
        markdown = "\n\n".join(md_parts).strip()
        text_chars = len("\n\n".join(text_parts).strip())
        return markdown, text_chars, pages_kept, len(chosen_pages), engine

    def _pick_pages(
        self,
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

    def _detect_repeated_header_footer(self, pages: list[list[str]]) -> set[str]:
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
            if cnt >= threshold
            and len(line) < 140
            and not self._looks_like_heading(line)
        }

    def _merge_paragraph_lines(self, lines: list[str]) -> list[str]:
        out: list[str] = []
        buf = ""
        for line in lines:
            line = self._normalize_pdf_line(line)
            if not line:
                continue
            if buf.endswith("-"):
                buf = f"{buf[:-1]}{line.lstrip()}"
                continue
            if self._is_list_item(line):
                if buf:
                    out.append(buf.strip())
                    buf = ""
                out.append(line)
                continue
            if self._looks_like_heading(line):
                if buf:
                    out.append(buf.strip())
                    buf = ""
                out.append(f"### {line}")
                continue
            if not buf:
                buf = line
                continue
            if self._is_sentence_end(buf):
                out.append(buf.strip())
                buf = line
            else:
                buf = f"{buf} {line}"
        if buf:
            out.append(buf.strip())
        return [x for x in self._dedupe_lines(out) if x]

    def _normalize_pdf_line(self, line: str) -> str:
        line = clean_whitespace(line)
        if not line:
            return ""
        line = line.replace("\u00ad", "")
        line = re.sub(r"\s+", " ", line)
        return line.strip()

    def _normalize_markdown_chunk(self, chunk: str) -> str:
        return chunk.replace("\r\n", "\n").replace("\r", "\n").strip()

    def _normalize_pdf_date(self, value: str) -> str:
        raw = clean_whitespace(value)
        if not raw:
            return ""
        match = re.match(
            r"^D:(\d{4})(\d{2})?(\d{2})?(\d{2})?(\d{2})?(\d{2})?",
            raw,
        )
        if match is None:
            return raw
        year, month, day, hour, minute, second = match.groups()
        out = [year]
        if month:
            out.append(month)
        if day:
            out.append(day)
        date_part = "-".join(out[:3]) if len(out) >= 3 else "-".join(out)
        if hour and minute and second and len(out) >= 3:
            return f"{date_part}T{hour}:{minute}:{second}"
        return date_part

    def _quality_warnings(self, text_chars: int) -> list[str]:
        warnings: list[str] = []
        if text_chars < 180:
            warnings.append("very_low_text_pdf; likely scanned or encrypted")
        return warnings

    def _dedupe_lines(self, lines: list[str]) -> list[str]:
        out: list[str] = []
        seen: set[str] = set()
        for line in lines:
            norm = line.lower().strip()
            if len(norm) > 12 and norm in seen:
                continue
            seen.add(norm)
            out.append(line)
        return out

    def _looks_like_heading(self, line: str) -> bool:
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

    def _is_list_item(self, line: str) -> bool:
        return bool(re.match(r"^(\d+[\.\)]|[-*])\s+", line))

    def _is_sentence_end(self, text: str) -> bool:
        return text.rstrip().endswith(
            (".", "!", "?", ";", ":", "\u3002", "\uff01", "\uff1f")
        )


__all__ = ["PdfExtractor"]
