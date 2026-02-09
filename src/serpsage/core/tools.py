"""Utility helpers used by search/ranking/web enrichment.

This module intentionally contains small, dependency-free helpers that are shared
across the codebase (regex compilation, normalization, fuzzy similarity, etc.).
"""

from __future__ import annotations

import logging
import re
from difflib import SequenceMatcher
from typing import TYPE_CHECKING

from serpsage.core.text import PUNCTUATION_RE, TextUtils

if TYPE_CHECKING:
    from collections.abc import Iterable

    from serpsage.core.config import SearchContextConfig

logger = logging.getLogger(__name__)


def compile_patterns(patterns: Iterable[str]) -> list[re.Pattern[str]]:
    """Compile regex patterns (case-insensitive), skipping invalid ones.

    Args:
        patterns: Regex pattern strings.

    Returns:
        A list of compiled regex patterns. Invalid patterns are skipped with a warning.
    """
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
    """Compute a bonus score for a domain based on exact/suffix matches.

    Args:
        domain: Domain name (e.g. ``"example.com"``). May be None/empty.
        bonus_map: Mapping of domain or suffix -> integer bonus. Example:
            ``{"wikipedia.org": 5, ".edu": 2}`` (suffix match uses ``endswith``).

    Returns:
        An integer bonus, or 0 when no match is found.
    """
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
    """Map a domain into a canonical bucket name.

    This is used to group multiple related sites together for dedupe/bonus logic.

    Args:
        domain: Domain name.
        domain_groups: Mapping of canonical group -> tuple of substrings. If any
            substring exists in the domain, the group name is returned.

    Returns:
        A canonical site/group name. If no groups are configured, returns the
        normalized domain. For empty input, returns ``"other"``.
    """
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
    """Strip common tail fragments from a title (site name, separators, etc.).

    Args:
        title: Original title string.
        patterns: Compiled regex patterns to remove.

    Returns:
        Cleaned title string.
    """
    cleaned = title or ""
    for pattern in patterns:
        cleaned = pattern.sub("", cleaned).strip()
    return cleaned


def fuzzy_normalize(text: str) -> str:
    """Normalize text for fuzzy matching (lower + punctuation -> spaces).

    Args:
        text: Input text.

    Returns:
        Normalized text with collapsed whitespace.
    """
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
    """Extract "intent" tokens that appear in the query.

    Args:
        query: User query string.
        config: Profile configuration that defines ``intent_terms``.

    Returns:
        A list of intent terms that occur as substrings in the lowercased query.
    """
    lowered = query.lower()
    return [term for term in config.intent_terms if term.lower() in lowered]


def has_noise_word(text: str, context_config: SearchContextConfig) -> bool:
    """Check whether a text contains any configured noise word.

    Args:
        text: Input text.
        context_config: Profile configuration with ``noise_words``.

    Returns:
        True if any noise word is found after normalization; otherwise False.
    """
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
