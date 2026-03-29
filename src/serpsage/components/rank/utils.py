from __future__ import annotations

import math
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable


def safe_float(x: object) -> float:
    if not isinstance(x, (int, float, str)):
        return 0.0
    try:
        xf = float(x)
    except Exception:  # noqa: BLE001
        return 0.0
    if math.isnan(xf) or math.isinf(xf):
        return 0.0
    return xf


def rank_scales(scores: list[float]) -> list[float]:
    """Convert raw scores to percentile ranks in [0, 1].

    Handles three cases:
    1. Empty list: return empty
    2. Single item with score > 0: return [0.75] (above neutral, not saturated)
    3. Single item with score = 0: return [0.0] (no match signal)
    4. All scores identical (non-zero): use positional tiebreaker
    5. All scores zero: return [0.0] for all (no match signal)
    6. Normal case: compute percentile ranks based on score ordering
    """
    cleaned = [safe_float(s) for s in (scores or [])]
    n = len(cleaned)
    if n == 0:
        return []
    if n == 1:
        # Single item: return meaningful signal based on actual score
        return [0.75 if cleaned[0] > 0 else 0.0]

    max_val = max(cleaned)
    min_val = min(cleaned)

    # All scores are zero: no match signal for any document
    if max_val <= 0.0:
        return [0.0 for _ in cleaned]

    # All scores identical but positive: use positional tiebreaker
    # Earlier documents get higher percentile (first-match advantage)
    if max_val - min_val < 1e-9:
        # Distribute uniformly across positions, first gets highest
        return [1.0 - (i / (n - 1)) for i in range(n)]

    # Normal case: compute percentile based on score ranking
    ordered = sorted(enumerate(cleaned), key=lambda t: t[1], reverse=True)
    out = [0.0 for _ in cleaned]
    i = 0
    while i < n:
        j = i
        while j < n and ordered[j][1] == ordered[i][1]:
            j += 1
        avg_pos = (i + (j - 1)) / 2.0
        percentile = 1.0 - (avg_pos / (n - 1))
        for k in range(i, j):
            out[ordered[k][0]] = float(percentile)
        i = j
    return out


def blend_weighted(
    *,
    scores: dict[str, list[float]],
    weights: dict[str, float],
    transforms: dict[str, Callable[[list[float]], list[float]]] | None = None,
) -> list[float]:
    if not scores:
        return []
    n = len(next(iter(scores.values())))
    out = [0.0] * n
    for name, w in weights.items():
        if w <= 0:
            continue
        s = scores.get(name)
        if s is None:
            continue
        if transforms is not None:
            fn = transforms.get(name)
            if fn is not None:
                s = fn(s)
        for i in range(n):
            out[i] += float(s[i]) * float(w)
    return out


__all__ = ["rank_scales", "safe_float", "blend_weighted"]
