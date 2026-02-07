from __future__ import annotations

import math
import statistics
from typing import TYPE_CHECKING

try:
    from rank_bm25 import BM25Okapi

    BM25_AVAILABLE = True
except ImportError:  # pragma: no cover - optional dependency
    BM25Okapi = None
    BM25_AVAILABLE = False

from .utils import TextUtils

if TYPE_CHECKING:
    from .config import ScoringConfig


def _safe_float(x: float) -> float:
    try:
        xf = float(x)
    except Exception:  # noqa: BLE001
        return 0.0
    if math.isnan(xf) or math.isinf(xf):
        return 0.0
    return xf


class ScoringEngine:
    """Scores plain text strings; nothing else."""

    config: ScoringConfig

    def __init__(self, config: ScoringConfig) -> None:
        self.config = config

    def score(
        self,
        texts: list[str],
        *,
        query: str,
        query_tokens: list[str] | None = None,
        intent_tokens: list[str] | None = None,
    ) -> list[tuple[float, str]]:
        if not texts:
            return []

        hcfg = self.config.heuristic
        q_tokens = (
            query_tokens if query_tokens is not None else TextUtils.tokenize(query)
        )
        q_tokens = [t for t in (q_tokens or []) if len(t) >= int(hcfg.min_token_len)]
        i_tokens = intent_tokens or []

        normalized_query = TextUtils.normalize_text(query)

        heuristic_raw: list[float] = []
        for text in texts:
            normalized_text = TextUtils.normalize_text(text)
            if not normalized_text:
                heuristic_raw.append(0.0)
                continue

            unique_hits: set[str] = set()
            count_hits = 0
            for t in q_tokens:
                tl = t.lower()
                if tl and tl in normalized_text:
                    unique_hits.add(tl)
                    count_hits += min(
                        normalized_text.count(tl), int(hcfg.max_count_per_token)
                    )

            intent_hits = 0
            for t in i_tokens:
                tl = (t or "").lower()
                if tl and tl in normalized_text:
                    intent_hits += 1

            score = 0.0
            score += float(hcfg.unique_hit_weight) * float(len(unique_hits))
            score += float(hcfg.count_weight) * float(count_hits)
            score += float(hcfg.intent_hit_weight) * float(intent_hits)

            if normalized_query and normalized_query in normalized_text:
                score += float(hcfg.phrase_bonus)

            heuristic_raw.append(float(score))

        weights = self.normalize_provider_weights()
        heur_w = float(weights.get("heuristic", 0.0))
        bm25_w = float(weights.get("bm25", 0.0))

        blended = [float(s) * heur_w for s in heuristic_raw]
        if bm25_w > 0:
            bm25_raw = self.bm25_scores(texts, query=query)
            scaled = self.rank_scale(bm25_raw)
            max_heur = max(heuristic_raw) if heuristic_raw else 0.0
            anchor = float(max_heur) if float(max_heur) > 0 else 1.0
            for i in range(len(blended)):
                blended[i] += float(scaled[i]) * bm25_w * anchor

        norm = self.normalize_scores(blended)
        # If the distribution is flat but still has a positive signal, return a
        # neutral non-zero baseline so downstream penalties can still apply.
        if norm and max(norm) <= 0.0 and max(blended) > 0.0:
            norm = [0.5 for _ in norm]
        return [(float(norm[i]), texts[i]) for i in range(len(texts))]

    def bm25_scores(self, docs: list[str], *, query: str) -> list[float]:
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

    def rank_scale(self, scores: list[float]) -> list[float]:
        """Rank-based scaling to [0,1] (top=1, bottom=0), stable with ties.

        Intentionally robust for near-flat distributions (returns all zeros).
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

    def normalize_scores(self, scores: list[float]) -> list[float]:
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
        if spread < float(self.config.normalization.flat_spread_eps):
            # Avoid false confidence in almost-flat distributions.
            return [0.0 for _ in cleaned]

        method = (self.config.normalization.method or "robust_sigmoid").lower()
        if method == "rank" or n < int(self.config.normalization.min_items_for_sigmoid):
            return self.rank_scale(cleaned)

        # Robust sigmoid: z = (x - median) / (1.4826*MAD + eps), then sigmoid(z / T).
        m = statistics.median(cleaned)
        deviations = [abs(x - m) for x in cleaned]
        mad = statistics.median(deviations)
        scale = 1.4826 * mad + 1e-12

        z_clip = max(0.0, float(self.config.normalization.z_clip))
        temp = (
            float(self.config.normalization.temperature)
            if float(self.config.normalization.temperature) > 0
            else 1.0
        )

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

    def normalize_provider_weights(self) -> dict[str, float]:
        raw = {
            k: float(v)
            for k, v in (self.config.providers or {}).items()
            if float(v) > 0
        }
        if raw.get("bm25") and not BM25_AVAILABLE:
            raw.pop("bm25", None)
        if not raw:
            return {"heuristic": 1.0}
        total = sum(raw.values())
        if total <= 0:
            return {"heuristic": 1.0}
        return {k: float(v) / total for k, v in raw.items()}


__all__ = [
    "BM25_AVAILABLE",
    "ScoringEngine",
]
