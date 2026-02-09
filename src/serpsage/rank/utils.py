from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from serpsage.settings.models import NormalizationSettings


def safe_float(x: float) -> float:
    try:
        xf = float(x)
    except Exception:  # noqa: BLE001
        return 0.0
    if math.isnan(xf) or math.isinf(xf):
        return 0.0
    return xf


def rank_scales(scores: list[float]) -> list[float]:
    cleaned = [safe_float(s) for s in (scores or [])]
    n = len(cleaned)
    if n == 0:
        return []
    if n == 1:
        # Single item: keep a neutral-ish rank signal instead of saturating at 1.0.
        return [0.5 if cleaned[0] > 0 else 0.0]
    if max(cleaned) - min(cleaned) < 1e-9:
        return [0.5 for _ in cleaned]

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


def normalize_scores(scores: list[float], cfg: NormalizationSettings) -> list[float]:  # noqa: PLR0911
    cleaned = [safe_float(s) for s in (scores or [])]
    n = len(cleaned)
    if n == 0:
        return []
    if n == 1:
        method = (cfg.single_item_method or "sigmoid_log1p").lower()
        x = max(0.0, float(cleaned[0]))
        scale = (
            float(cfg.single_item_scale) if float(cfg.single_item_scale) > 0 else 1.0
        )
        if method == "fixed_0.5":
            return [0.5]
        if method == "exp":
            # Smooth (0,1): 1 - exp(-x/scale). Avoids saturating too quickly.
            return [float(1.0 - math.exp(-x / scale))]
        # Default: "rounder" head/tail without saturating to 1.0.
        # Map x=0 -> 0.0 by re-scaling sigmoid from [0.5, 1) to [0, 1).
        t = math.log1p(x) / scale
        sig = 1.0 / (1.0 + math.exp(-t))
        return [float(max(0.0, min(1.0, (2.0 * sig) - 1.0)))]

    spread = max(cleaned) - min(cleaned)
    if spread < float(cfg.flat_spread_eps):
        # Flat/degenerate: do not collapse to 0.0 (which misleads downstream filters).
        return [0.5 for _ in cleaned]

    method = (cfg.method or "robust_sigmoid").lower()
    if method == "rank" or n < int(cfg.min_items_for_sigmoid):
        return rank_scales(cleaned)

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

    return [min(1.0, max(0.0, x)) for x in out]


__all__ = ["normalize_scores", "rank_scales", "safe_float"]
