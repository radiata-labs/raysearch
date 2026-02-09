from __future__ import annotations

import hashlib
from difflib import SequenceMatcher

from serpsage.text.normalize import normalize_text


def char_ngrams(text: str, n: int) -> set[str]:
    compact = (text or "").replace(" ", "")
    if not compact:
        return set()
    if len(compact) < n:
        return {compact}
    return {compact[i : i + n] for i in range(len(compact) - n + 1)}


def jaccard(a: set[str], b: set[str]) -> float:
    if not a or not b:
        return 0.0
    inter = len(a & b)
    union = len(a | b)
    return inter / union if union else 0.0


def hybrid_similarity(a: str, b: str) -> float:
    if not a or not b:
        return 0.0
    if a == b:
        return 1.0
    seq = SequenceMatcher(None, a, b).ratio()
    jac = jaccard(char_ngrams(a, 2), char_ngrams(b, 2))
    return float(max(seq * 0.95, jac))


def simhash64(text: str) -> int:
    """A tiny 64-bit simhash for near-duplicate detection."""
    t = normalize_text(text)
    if not t:
        return 0
    feats = [f for f in t.split(" ") if f]
    if not feats:
        return 0

    vec = [0] * 64
    for f in feats:
        h = int.from_bytes(
            hashlib.blake2b(f.encode("utf-8"), digest_size=8).digest(), "big"
        )
        for i in range(64):
            bit = 1 if (h >> i) & 1 else -1
            vec[i] += bit

    out = 0
    for i in range(64):
        if vec[i] >= 0:
            out |= 1 << i
    return out


def is_duplicate_text(text: str, kept: list[str], *, threshold: float) -> bool:
    if not kept or threshold <= 0:
        return False
    a = normalize_text(text)
    if not a:
        return True
    a_grams = char_ngrams(a, 2)
    for b in kept:
        b_norm = normalize_text(b)
        if not b_norm:
            continue
        if jaccard(a_grams, char_ngrams(b_norm, 2)) >= threshold:
            return True
    return False


__all__ = [
    "hybrid_similarity",
    "is_duplicate_text",
    "jaccard",
    "simhash64",
]
