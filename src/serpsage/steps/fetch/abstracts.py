from __future__ import annotations

import re
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.models.errors import AppError
from serpsage.models.pipeline import FetchStepContext, PreparedAbstract
from serpsage.steps.base import StepBase
from serpsage.utils import clean_whitespace

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime
    from serpsage.telemetry.base import SpanBase

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")
_INLINE_CODE_ONLY_RE = re.compile(r"^\s*`[^`]+`\s*$")
_CJK_SENTENCE_END = {"。", "！", "？", "；"}
_GENERAL_SENTENCE_END = {"!", "?", ";"}


class FetchAbstractBuildStep(StepBase[FetchStepContext]):
    span_name = "step.fetch_abstract_build"

    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def run_inner(
        self, ctx: FetchStepContext, *, span: SpanBase
    ) -> FetchStepContext:
        if ctx.fatal:
            return ctx
        req = ctx.abstracts_request
        span.set_attr("has_abstracts", bool(req is not None))
        if req is None:
            return ctx
        if ctx.extracted is None:
            ctx.errors.append(
                AppError(
                    code="fetch_abstract_build_failed",
                    message="missing extracted content",
                    details={
                        "url": ctx.url,
                        "url_index": ctx.url_index,
                        "stage": "abstract_build",
                        "fatal": False,
                        "crawl_mode": ctx.others.crawl_mode,
                    },
                )
            )
            return ctx

        markdown = str(ctx.extracted.md_for_abstract or ctx.extracted.markdown or "")
        cfg = self.settings.fetch.abstract
        prepared = self._extract_abstracts(
            markdown=markdown,
            min_abstract_chars=int(cfg.min_abstract_chars),
        )
        ctx.prepared_abstracts = prepared
        span.set_attr("prepared_abstracts", int(len(prepared)))
        return ctx

    def _extract_abstracts(
        self,
        *,
        markdown: str,
        min_abstract_chars: int,
    ) -> list[PreparedAbstract]:
        text = (markdown or "").strip()
        if not text:
            return []

        out: list[PreparedAbstract] = []
        in_code = False
        current_heading = ""
        position = 0

        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            stripped = line.strip()
            if not stripped:
                continue
            if _FENCE_RE.match(stripped):
                in_code = not in_code
                continue
            if in_code:
                continue

            heading_match = _HEADING_RE.match(stripped)
            if heading_match:
                current_heading = clean_whitespace(heading_match.group(1) or "")
                continue
            if _INLINE_CODE_ONLY_RE.match(stripped):
                continue

            if _TABLE_ROW_RE.match(stripped):
                row = self._parse_table_row(stripped)
                if row and len(row) >= max(1, int(min_abstract_chars)):
                    out.append(
                        PreparedAbstract(
                            text=row,
                            heading=current_heading,
                            position=position,
                        )
                    )
                    position += 1
                continue

            normalized = _LIST_PREFIX_RE.sub("", stripped)
            normalized = clean_whitespace(normalized)
            if not normalized:
                continue

            for sentence in self._split_line_sentences(normalized):
                if len(sentence) < max(1, int(min_abstract_chars)):
                    continue
                out.append(
                    PreparedAbstract(
                        text=sentence,
                        heading=current_heading,
                        position=position,
                    )
                )
                position += 1

        return out

    def _split_line_sentences(self, text: str) -> list[str]:
        normalized = clean_whitespace(text or "")
        if not normalized:
            return []
        out: list[str] = []
        buf: list[str] = []
        for idx, ch in enumerate(normalized):
            buf.append(ch)
            if ch in _CJK_SENTENCE_END or ch in _GENERAL_SENTENCE_END:
                sentence = clean_whitespace("".join(buf))
                if sentence:
                    out.append(sentence)
                buf = []
                continue
            if ch != ".":
                continue
            next_char = normalized[idx + 1] if idx + 1 < len(normalized) else ""
            if next_char and not next_char.isspace():
                continue
            sentence = clean_whitespace("".join(buf))
            if sentence:
                out.append(sentence)
            buf = []

        tail = clean_whitespace("".join(buf))
        if tail:
            out.append(tail)
        return out or [normalized]

    def _parse_table_row(self, line: str) -> str:
        body = line.strip()
        body = body.removeprefix("|")
        body = body.removesuffix("|")
        cells = [clean_whitespace(cell) for cell in body.split("|")]
        if not cells:
            return ""
        if all(_TABLE_SEP_CELL_RE.fullmatch(cell or "") for cell in cells if cell):
            return ""
        values = [cell for cell in cells if cell]
        return " | ".join(values)


__all__ = ["FetchAbstractBuildStep"]
