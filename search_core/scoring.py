from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING, Literal

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    BM25Okapi = None
    BM25_AVAILABLE = False

from .utils import TextUtils

if TYPE_CHECKING:
    from .config import ScoreNormalizationConfig


def _safe_float(x: float) -> float:
    try:
        xf = float(x)
    except Exception:  # noqa: BLE001
        return 0.0
    if math.isnan(xf) or math.isinf(xf):
        return 0.0
    return xf


def bm25_scores(docs: list[str], *, query: str) -> list[float]:
    """Compute BM25 scores for docs against query.

    Returns a list aligned with `docs`. If BM25 isn't available, returns zeros.
    """

    if not docs:
        return []

    if not BM25_AVAILABLE or BM25Okapi is None:
        return [0.0 for _ in docs]

    query_tokens = TextUtils.tokenize(query)
    if not query_tokens:
        return [0.0 for _ in docs]

    corpus = [TextUtils.tokenize(doc) for doc in docs]
    if not corpus:
        return [0.0 for _ in docs]

    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(query_tokens)
    return [float(s) for s in scores]


def rank_scale(scores: list[float]) -> list[float]:
    """Rank-based scaling to [0,1] (top=1, bottom=0), stable with ties.

    This is intentionally robust for near-flat distributions (returns all zeros).
    """

    cleaned = [_safe_float(s) for s in (scores or [])]
    n = len(cleaned)
    if n == 0:
        return []
    if n == 1:
        return [1.0 if cleaned[0] > 0 else 0.0]

    if max(cleaned) - min(cleaned) < 1e-9:
        return [0.0 for _ in cleaned]

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


def normalize_scores(scores: list[float], cfg: ScoreNormalizationConfig) -> list[float]:
    """Monotonic mapping of a candidate set to [0,1] for output.

    This is a per-candidate-set normalization (not cross-query comparable).
    """

    cleaned = [_safe_float(s) for s in (scores or [])]
    n = len(cleaned)
    if n == 0:
        return []
    if n == 1:
        return [1.0]

    spread = max(cleaned) - min(cleaned)
    if spread < float(cfg.flat_spread_eps):
        # Avoid false confidence in almost-flat distributions.
        return [0.0 for _ in cleaned]

    method = (cfg.method or "robust_sigmoid").lower()
    if method == "rank" or n < int(cfg.min_items_for_sigmoid):
        return rank_scale(cleaned)

    # Robust sigmoid: z = (x - median) / (1.4826*MAD + eps), then sigmoid(z / T).
    m = statistics.median(cleaned)
    deviations = [abs(x - m) for x in cleaned]
    mad = statistics.median(deviations)
    scale = 1.4826 * mad + 1e-12

    z_clip = max(0.0, float(cfg.z_clip))
    temp = float(cfg.temperature) if float(cfg.temperature) > 0 else 1.0

    out: list[float] = []
    for x in cleaned:
        z = (x - m) / scale
        if z_clip:
            z = max(-z_clip, min(z_clip, z))
        t = z / temp
        p = 1.0 / (1.0 + math.exp(-t))
        out.append(float(p))

    # Guarantee bounds (floating noise).
    return [min(1.0, max(0.0, x)) for x in out]


def blend_scores(
    *,
    strategy: Literal["heuristic", "bm25", "hybrid"],
    heuristic: list[float],
    bm25: list[float] | None,
    weights: tuple[float, float],
) -> list[float]:
    """Blend heuristic + bm25 scores according to strategy.

    BM25 is rank-scaled to [0,1] then aligned to heuristic scale via max(heuristic).
    If bm25 is unavailable, the function falls back to heuristic-only.
    """

    n = len(heuristic)
    if n == 0:
        return []

    strat = (strategy or "heuristic").lower()
    if strat not in {"heuristic", "bm25", "hybrid"}:
        strat = "heuristic"

    if strat in {"bm25", "hybrid"} and (bm25 is None or len(bm25) != n):
        strat = "heuristic"

    if strat == "heuristic":
        return [float(x) for x in heuristic]

    max_heur = max(heuristic) if heuristic else 1.0
    scaled = rank_scale(bm25 or [])

    if strat == "bm25":
        return [float(scaled[i]) * float(max_heur) for i in range(n)]

    bm25_w, heur_w = weights
    return [
        float(scaled[i]) * float(bm25_w) * float(max_heur)
        + float(heuristic[i]) * float(heur_w)
        for i in range(n)
    ]


__all__ = [
    "blend_scores",
    "normalize_scores",
    "rank_scale",
    "BM25_AVAILABLE",
    "bm25_scores",
]
