from __future__ import annotations

import re
from typing_extensions import override

from serpsage.models.steps.fetch import FetchStepContext, PreparedPassage
from serpsage.steps.base import StepBase
from serpsage.tokenize import tokenize
from serpsage.utils import clean_whitespace

_FENCE_RE = re.compile(r"^\s*(```|~~~)")
_HEADING_RE = re.compile(r"^\s*#{1,6}\s+(.+?)\s*$")
_LIST_PREFIX_RE = re.compile(r"^\s*(?:[-*+]\s+|\d+[.)]\s+)")
_TABLE_ROW_RE = re.compile(r"^\s*\|.*\|\s*$")
_TABLE_SEP_CELL_RE = re.compile(r"^:?-{2,}:?$")
_INLINE_CODE_ONLY_RE = re.compile(r"^\s*`[^`]+`\s*$")
_DOT_SENTENCE_END = {".", "\uff0e"}
_CJK_SENTENCE_END = {"\u3002", "\uff01", "\uff1f", "\uff1b", "\uff61"}
_GENERAL_SENTENCE_END = {"!", "?", ";"}
_TRAILING_END_CHARS = (
    _DOT_SENTENCE_END | _CJK_SENTENCE_END | _GENERAL_SENTENCE_END | {"\u2026"}
)
_OPEN_BRACKETS = {
    "(",
    "[",
    "{",
    "\uff08",
    "\uff3b",
    "\uff5b",
    "\u3010",
    "\u300c",
    "\u300e",
    "\u300a",
    "\u3008",
    "\u3014",
    "\u3016",
}
_CLOSE_BRACKETS = {
    ")",
    "]",
    "}",
    "\uff09",
    "\uff3d",
    "\uff5d",
    "\u3011",
    "\u300d",
    "\u300f",
    "\u300b",
    "\u3009",
    "\u3015",
    "\u3017",
}
_TRAILING_QUOTES = {'"', "'", "\u201d", "\u2019", "\u00bb", "\u203a"}
_TRAILING_CLOSERS = _CLOSE_BRACKETS | _TRAILING_QUOTES
_DOT_PREV_TOKEN_RE = re.compile(r"([A-Za-z][A-Za-z0-9'-]{0,20})\.$")
_DOT_INITIALISM_RE = re.compile(r"(?:\b[A-Za-z]\.){2,}$")
_ALWAYS_NON_TERMINAL_DOT_ABBR = {
    "mr",
    "mrs",
    "ms",
    "dr",
    "prof",
    "sr",
    "jr",
    "st",
    "mt",
    "vs",
    "no",
    "fig",
    "eq",
    "sec",
    "art",
    "vol",
    "dept",
    "gen",
    "sen",
    "gov",
    "pres",
    "capt",
    "cmdr",
    "col",
    "lt",
    "sgt",
    "adm",
    "rev",
}
_LOWER_CONTINUATION_DOT_ABBR = {
    "etc",
    "inc",
    "ltd",
    "corp",
    "co",
    "jan",
    "feb",
    "mar",
    "apr",
    "jun",
    "jul",
    "aug",
    "sep",
    "sept",
    "oct",
    "nov",
    "dec",
}


class FetchAbstractBuildStep(StepBase[FetchStepContext]):
    @override
    async def run_inner(self, ctx: FetchStepContext) -> FetchStepContext:
        if ctx.error.failed:
            return ctx
        req = ctx.analysis.abstracts.request
        if req is None:
            return ctx
        if ctx.page.doc is None:
            await self.emit_tracking_event(
                event_name="fetch.abstract_build.error",
                request_id=ctx.request_id,
                stage="abstract_build",
                status="error",
                error_code="fetch_abstract_build_failed",
                attrs={
                    "url": ctx.url,
                    "url_index": int(ctx.url_index),
                    "fatal": False,
                    "crawl_mode": str(ctx.page.crawl_mode),
                    "message": "missing extracted content",
                },
            )
            return ctx
        markdown = str(
            ctx.page.doc.content.abstract_text or ctx.page.doc.content.markdown or ""
        )
        cfg = ctx.settings.fetch.abstract
        prepared = self._extract_abstracts(
            markdown=markdown,
            min_abstract_tokens=int(cfg.min_abstract_tokens),
        )
        ctx.analysis.abstracts.prepared = prepared
        return ctx

    def _extract_abstracts(
        self,
        *,
        markdown: str,
        min_abstract_tokens: int,
    ) -> list[PreparedPassage]:
        text = (markdown or "").strip()
        if not text:
            return []
        out: list[PreparedPassage] = []
        seen: set[str] = set()
        in_code = False
        current_heading = ""
        order = 0
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
                if row:
                    key = self._dedupe_key(row)
                    if key not in seen:
                        seen.add(key)
                        out.append(
                            PreparedPassage(
                                text=row,
                                heading=current_heading,
                                order=order,
                            )
                        )
                        order += 1
                continue
            normalized = _LIST_PREFIX_RE.sub("", stripped)
            normalized = clean_whitespace(normalized)
            if not normalized:
                continue
            for sentence in self._split_line_sentences(normalized):
                if self._token_count(sentence) <= int(min_abstract_tokens):
                    continue
                key = self._dedupe_key(sentence)
                if key in seen:
                    continue
                seen.add(key)
                out.append(
                    PreparedPassage(
                        text=sentence,
                        heading=current_heading,
                        order=order,
                    )
                )
                order += 1
        return out

    def _dedupe_key(self, text: str) -> str:
        return clean_whitespace(text).casefold()

    def _token_count(self, text: str) -> int:
        return len(tokenize(text))

    def _split_line_sentences(self, text: str) -> list[str]:
        normalized = clean_whitespace(text or "")
        if not normalized:
            return []
        out: list[str] = []
        start = 0
        bracket_depth = 0
        idx = 0
        while idx < len(normalized):
            ch = normalized[idx]
            if ch in _OPEN_BRACKETS:
                bracket_depth += 1
                idx += 1
                continue
            if ch in _CLOSE_BRACKETS and bracket_depth > 0:
                bracket_depth -= 1
            if not self._is_sentence_boundary(
                text=normalized,
                idx=idx,
                ch=ch,
                bracket_depth=bracket_depth,
            ):
                idx += 1
                continue
            end = self._expand_sentence_boundary(text=normalized, idx=idx)
            sentence = clean_whitespace(normalized[start : end + 1])
            if sentence:
                out.append(sentence)
            for tail_ch in normalized[idx + 1 : end + 1]:
                if tail_ch in _OPEN_BRACKETS:
                    bracket_depth += 1
                elif tail_ch in _CLOSE_BRACKETS and bracket_depth > 0:
                    bracket_depth -= 1
            start = self._skip_spaces(text=normalized, idx=end + 1)
            idx = start
        tail = clean_whitespace(normalized[start:])
        if tail:
            out.append(tail)
        return out or [normalized]

    def _is_sentence_boundary(
        self,
        *,
        text: str,
        idx: int,
        ch: str,
        bracket_depth: int,
    ) -> bool:
        if ch in _GENERAL_SENTENCE_END or ch in _CJK_SENTENCE_END:
            return bracket_depth <= 0 or self._is_bracket_boundary_allowed(
                text=text, idx=idx + 1
            )
        if ch not in _DOT_SENTENCE_END:
            return False
        return self._is_dot_boundary(
            text=text,
            idx=idx,
            bracket_depth=bracket_depth,
        )

    def _is_dot_boundary(self, *, text: str, idx: int, bracket_depth: int) -> bool:
        prev_char = text[idx - 1] if idx > 0 else ""
        next_char = text[idx + 1] if idx + 1 < len(text) else ""
        if not prev_char:
            return False
        if not next_char:
            return True
        if bracket_depth > 0 and not self._is_bracket_boundary_allowed(
            text=text, idx=idx + 1
        ):
            return False
        if prev_char.isdigit() and next_char.isdigit():
            return False
        if self._is_latin_letter(prev_char) and self._is_latin_letter(next_char):
            return False
        if next_char in _DOT_SENTENCE_END:
            return False
        if next_char in _TRAILING_CLOSERS:
            return True
        if next_char.isspace():
            _, next_non_space = self._next_non_space(text=text, idx=idx + 1)
            if not next_non_space:
                return True
            token_before = self._token_before_dot(text=text, idx=idx)
            if token_before in _ALWAYS_NON_TERMINAL_DOT_ABBR:
                return False
            if token_before in _LOWER_CONTINUATION_DOT_ABBR and self._is_latin_lower(
                next_non_space
            ):
                return False
            if self._looks_like_initialism(text=text, idx=idx) and (
                self._is_latin_letter(next_non_space) or next_non_space.isdigit()
            ):
                return False
            if next_non_space in _OPEN_BRACKETS or next_non_space in _TRAILING_QUOTES:
                return True
            if self._is_cjk_or_kana(next_non_space):
                return True
            return bool(self._is_latin_upper(next_non_space))
        if self._is_cjk_or_kana(next_char):
            return True
        return self._is_latin_upper(next_char)

    def _expand_sentence_boundary(self, *, text: str, idx: int) -> int:
        end = idx
        while end + 1 < len(text):
            next_ch = text[end + 1]
            if next_ch in _TRAILING_END_CHARS or next_ch in _TRAILING_CLOSERS:
                end += 1
                continue
            break
        return end

    def _skip_spaces(self, *, text: str, idx: int) -> int:
        cursor = idx
        while cursor < len(text) and text[cursor].isspace():
            cursor += 1
        return cursor

    def _next_non_space(self, *, text: str, idx: int) -> tuple[int, str]:
        cursor = self._skip_spaces(text=text, idx=idx)
        if cursor >= len(text):
            return -1, ""
        return cursor, text[cursor]

    def _is_bracket_boundary_allowed(self, *, text: str, idx: int) -> bool:
        cursor = idx
        while cursor < len(text):
            ch = text[cursor]
            if ch.isspace():
                return False
            if ch in _TRAILING_QUOTES or ch in _TRAILING_END_CHARS:
                cursor += 1
                continue
            return ch in _CLOSE_BRACKETS
        return True

    def _token_before_dot(self, *, text: str, idx: int) -> str:
        window = text[max(0, idx - 32) : idx + 1]
        match = _DOT_PREV_TOKEN_RE.search(window)
        if match is None:
            return ""
        return str(match.group(1)).casefold()

    def _looks_like_initialism(self, *, text: str, idx: int) -> bool:
        window = text[max(0, idx - 32) : idx + 1]
        return bool(_DOT_INITIALISM_RE.search(window))

    def _is_latin_letter(self, ch: str) -> bool:
        code = ord(ch)
        return 65 <= code <= 90 or 97 <= code <= 122

    def _is_latin_upper(self, ch: str) -> bool:
        code = ord(ch)
        return 65 <= code <= 90

    def _is_latin_lower(self, ch: str) -> bool:
        code = ord(ch)
        return 97 <= code <= 122

    def _is_cjk_or_kana(self, ch: str) -> bool:
        code = ord(ch)
        return (
            0x3400 <= code <= 0x9FFF
            or 0x3040 <= code <= 0x30FF
            or 0x31F0 <= code <= 0x31FF
            or 0xFF66 <= code <= 0xFF9D
            or ch in {"\u3005", "\u30fc"}
        )

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
