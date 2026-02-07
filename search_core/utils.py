from __future__ import annotations

import re
import unicodedata
from typing import TYPE_CHECKING

try:
    import jieba

    JIEBA_AVAILABLE = True
except Exception:  # noqa: BLE001
    jieba = None
    JIEBA_AVAILABLE = False

if TYPE_CHECKING:
    from collections.abc import Iterable

WORD_RE = re.compile(r"[A-Za-z0-9]+")
CJK_RE = re.compile(r"[\u4e00-\u9fff]{2,}")
HTML_TAG_RE = re.compile(r"<[^>]+>")
PUNCTUATION_RE = re.compile(
    r"[\s_—–…\"'`~!@#$%^&*()\[\]{}<>|/?:;,.，。、《》“”‘’「」『』【】（）-]+"
)
CJK_STOPWORDS = {
    "的",
    "了",
    "呢",
    "啊",
    "吧",
    "吗",
    "呀",
    "与",
    "及",
    "并",
    "和",
    "或",
    "被",
    "给",
    "于",
    "在",
    "对",
    "从",
    "为",
    "是",
    "就",
    "都",
    "也",
    "很",
    "又",
}


class TextUtils:
    """Text normalization helpers."""

    @staticmethod
    def normalize_text(text: str) -> str:
        """Normalize text (NFKC, lower, collapse whitespace)."""

        if not text:
            return ""
        normalized = unicodedata.normalize("NFKC", text)
        normalized = normalized.lower()
        return re.sub(r"\s+", " ", normalized).strip()

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

        normalized = TextUtils.normalize_text(text)
        if not normalized:
            return []

        tokens: list[str] = []

        # English/numeric tokens.
        tokens.extend(match.group(0).lower() for match in WORD_RE.finditer(normalized))

        if JIEBA_AVAILABLE and jieba is not None:
            for token in jieba.lcut(normalized, cut_all=False):
                token = token.strip()
                if not token:
                    continue
                if len(token) == 1:
                    # Ignore single-character Chinese tokens (too noisy).
                    if CJK_RE.fullmatch(token):
                        continue
                    if token in CJK_STOPWORDS:
                        continue
                if CJK_RE.fullmatch(token):
                    tokens.append(token)
                elif WORD_RE.fullmatch(token):
                    tokens.append(token.lower())
        else:
            # Fallback: use CJK runs directly when jieba isn't available.
            for run in CJK_RE.findall(normalized):
                if len(run) == 1:
                    continue
                tokens.append(run)

        # Add CJK n-grams as recall boost (avoid duplicates).
        for run in CJK_RE.findall(normalized):
            if len(run) <= 3:
                tokens.append(run)
            else:
                tokens.extend(TextUtils.ngrams(run, 2))
                tokens.extend(TextUtils.ngrams(run, 3))

        return TextUtils.unique_preserve_order(tokens)


__all__ = [
    "WORD_RE",
    "CJK_RE",
    "HTML_TAG_RE",
    "PUNCTUATION_RE",
    "TextUtils",
]
