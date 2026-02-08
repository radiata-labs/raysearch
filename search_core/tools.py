from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from search_core.text import PUNCTUATION_RE, TextUtils

if TYPE_CHECKING:
    from collections.abc import Iterable

    from search_core.config import SearchContextConfig

logger = logging.getLogger(__name__)


def compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    compiled: list[re.Pattern[str]] = []
    for pattern in patterns:
        if not pattern:
            continue
        try:
            compiled.append(re.compile(pattern, re.IGNORECASE))
        except re.error:
            logger.warning("Invalid regex pattern: %s", pattern)
    return compiled


def domain_bonus(domain: str | None, bonus_map: dict[str, int]) -> int:
    normalized = (domain or "").lower()
    if not normalized:
        return 0
    if normalized in bonus_map:
        return int(bonus_map[normalized])
    for key, value in bonus_map.items():
        if normalized.endswith(key):
            return int(value)
    return 0


def canonical_site(domain: str, domain_groups: dict[str, tuple[str, ...]]) -> str:
    normalized = (domain or "").lower()
    if not normalized:
        return "other"

    if domain_groups:
        for group, needles in domain_groups.items():
            if any(needle in normalized for needle in needles):
                return group
        return normalized

    return normalized


def strip_title_tails(title: str, patterns: list[re.Pattern[str]]) -> str:
    cleaned = title or ""
    for pattern in patterns:
        cleaned = pattern.sub("", cleaned).strip()
    return cleaned


def fuzzy_normalize(text: str) -> str:
    lowered = (text or "").lower()
    lowered = PUNCTUATION_RE.sub(" ", lowered)
    return re.sub(r"\s+", " ", lowered).strip()


def hybrid_similarity(a: str, b: str) -> float:
    """A robust similarity for fuzzy dedupe (sequence + char-ngram Jaccard)."""

    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    seq = SequenceMatcher(None, a, b).ratio()
    jac = TextUtils.jaccard(TextUtils.char_ngrams(a, 2), TextUtils.char_ngrams(b, 2))
    return float(max(seq * 0.95, jac))


def is_duplicate_text(
    text: str,
    kept: list[str],
    *,
    threshold: float,
) -> bool:
    """Chunk/text dedupe using normalized char-ngram Jaccard similarity."""

    if not kept:
        return False
    if threshold <= 0:
        return False

    a = TextUtils.normalize_text(text)
    if not a:
        return True
    a_grams = TextUtils.char_ngrams(a, 2)

    for b in kept:
        b_norm = TextUtils.normalize_text(b)
        if not b_norm:
            continue
        jac = TextUtils.jaccard(a_grams, TextUtils.char_ngrams(b_norm, 2))
        if jac >= threshold:
            return True
    return False


def extract_intent_tokens(query: str, config: SearchContextConfig) -> list[str]:
    lowered = query.lower()
    return [term for term in config.intent_terms if term.lower() in lowered]


def has_noise_word(text: str, context_config: SearchContextConfig) -> bool:
    lowered = TextUtils.normalize_text(text)
    return any(word and word.lower() in lowered for word in context_config.noise_words)


__all__ = [
    "compile_patterns",
    "domain_bonus",
    "canonical_site",
    "strip_title_tails",
    "fuzzy_normalize",
    "hybrid_similarity",
    "is_duplicate_text",
    "extract_intent_tokens",
    "has_noise_word",
]
