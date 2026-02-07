from __future__ import annotations

import re
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Iterable

WORD_RE = re.compile(r"[A-Za-z0-9]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
HTML_TAG_RE = re.compile(r"<[^>]+>")
PUNCTUATION_RE = re.compile(
    r"[\s_—–…\"'`~!@#$%^&*()\[\]{}<>|/?:;,.，。、《》“”‘’「」『』【】（）-]+"
)


class TextUtils:
    """Text normalization helpers."""

    @staticmethod
    def clean_whitespace(text: str) -> str:
        """Normalize whitespace."""

        return re.sub(r"\s+", " ", (text or "")).strip()

    @staticmethod
    def strip_html(text: str) -> str:
        """Strip HTML tags from text."""

        return HTML_TAG_RE.sub("", text or "").strip()

    @staticmethod
    def unique_preserve_order(items: Iterable[str]) -> list[str]:
        """Remove duplicates while preserving order."""

        seen: set[str] = set()
        output: list[str] = []
        for item in items:
            if item in seen:
                continue
            seen.add(item)
            output.append(item)
        return output

    @staticmethod
    def ngrams(text: str, size: int) -> list[str]:
        """Generate n-grams for a string."""

        if len(text) < size:
            return []
        return [text[i : i + size] for i in range(len(text) - size + 1)]

    @staticmethod
    def char_ngrams(text: str, size: int) -> set[str]:
        """Generate character n-grams for Jaccard similarity."""

        compact = text.replace(" ", "")
        if len(compact) < size:
            return {compact} if compact else set()
        return {compact[i : i + size] for i in range(len(compact) - size + 1)}

    @staticmethod
    def jaccard(a: set[str], b: set[str]) -> float:
        """Compute Jaccard similarity."""

        if not a or not b:
            return 0.0
        intersection = len(a & b)
        union = len(a | b)
        return intersection / union if union else 0.0

    @staticmethod
    def tokenize(text: str) -> list[str]:
        """Tokenize text for BM25 scoring."""

        tokens: list[str] = [match.group(0).lower() for match in WORD_RE.finditer(text)]
        for run in CJK_RE.findall(text):
            if len(run) <= 3:
                tokens.append(run)
            else:
                tokens.extend(TextUtils.ngrams(run, 2))
                tokens.extend(TextUtils.ngrams(run, 3))
        return tokens
