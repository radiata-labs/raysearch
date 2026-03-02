from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from serpsage.settings.models import HeuristicRankSettings


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


def normalize_scores(scores: list[float], cfg: HeuristicRankSettings) -> list[float]:
    cleaned = [safe_float(s) for s in (scores or [])]
    n = len(cleaned)
    if n == 0:
        return []
    out = [0.0 for _ in cleaned]
    pos_idx = [i for i, x in enumerate(cleaned) if x > 0.0]
    if not pos_idx:
        return out
    pos_vals = [float(cleaned[i]) for i in pos_idx]
    if len(pos_vals) == 1:
        # Single item: use 0.5 to be consistent with rank_scales (avoid saturating at 1.0)
        out[pos_idx[0]] = 0.5
        return out
    log_vals = [math.log1p(x) for x in pos_vals]
    temp = float(cfg.temperature) if float(cfg.temperature) > 0 else 1.0
    min_items = max(1, int(cfg.min_items_for_sigmoid))
    flat_eps = max(0.0, float(cfg.flat_spread_eps))
    z_clip = max(0.0, float(cfg.z_clip))
    spread = max(log_vals) - min(log_vals)
    if spread <= flat_eps:
        pos_norm = [0.5 for _ in log_vals]
    elif len(log_vals) < min_items:
        pos_norm = rank_scales(log_vals)
    else:
        med = statistics.median(log_vals)
        deviations = [abs(x - med) for x in log_vals]
        mad = statistics.median(deviations)
        scale = 1.4826 * mad + 1e-12
        sig: list[float] = []
        for x in log_vals:
            z = (x - med) / scale
            if z_clip:
                z = max(-z_clip, min(z_clip, z))
            p = 1.0 / (1.0 + math.exp(-(z / temp)))
            sig.append(float(p))
        rank_part = rank_scales(log_vals)
        pos_norm = [
            float((0.65 * sig[i]) + (0.35 * rank_part[i])) for i in range(len(sig))
        ]
    for idx, value in zip(pos_idx, pos_norm, strict=False):
        out[idx] = float(min(1.0, max(0.0, value)))
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


__all__ = ["normalize_scores", "rank_scales", "safe_float", "blend_weighted"]
