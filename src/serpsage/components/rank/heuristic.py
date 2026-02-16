from __future__ import annotations

import math
import re
from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.components.rank.base import RankerBase
from serpsage.components.rank.utils import normalize_scores
from serpsage.utils import normalize_text

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime

_SHORT_ENG_TOKEN_RE = re.compile(r"^[a-z0-9]{1,3}$")
_WORD_CHAR_CLASS = r"[a-z0-9]"


def _dedupe_tokens(tokens: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for tok in tokens or []:
        normalized = normalize_text(tok)
        if not normalized or normalized in seen:
            continue
        seen.add(normalized)
        out.append(normalized)
    return out


def _find_occurrences(text: str, token: str) -> list[int]:
    if not text or not token:
        return []

    if _SHORT_ENG_TOKEN_RE.fullmatch(token):
        pat = re.compile(
            rf"(?<!{_WORD_CHAR_CLASS}){re.escape(token)}(?!{_WORD_CHAR_CLASS})"
        )
        return [m.start() for m in pat.finditer(text)]

    positions: list[int] = []
    cursor = 0
    step = max(1, len(token))
    while True:
        pos = text.find(token, cursor)
        if pos < 0:
            break
        positions.append(pos)
        cursor = pos + step
    return positions


class HeuristicRanker(RankerBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    async def score_texts(
        self,
        *,
        texts: list[str],
        query: str,
        query_tokens: list[str],
    ) -> list[float]:
        cfg = self.settings.rank.heuristic
        query_terms = _dedupe_tokens(query_tokens)
        normalized_query = normalize_text(query)
        max_count = max(1, int(cfg.max_count_per_token))
        max_count_log = math.log1p(float(max_count))

        raw_scores: list[float] = []
        for text in texts:
            normalized_text = normalize_text(text)
            if not normalized_text:
                raw_scores.append(0.0)
                continue

            all_hit_positions: list[int] = []
            query_first_positions: list[int] = []
            query_hit_count = 0
            query_tf_quality_sum = 0.0

            for token in query_terms:
                positions = _find_occurrences(normalized_text, token)
                if not positions:
                    continue
                query_hit_count += 1
                query_first_positions.append(positions[0])
                all_hit_positions.extend(positions)
                capped_count = min(len(positions), max_count)
                query_tf_quality_sum += math.log1p(float(capped_count))

            total_query = len(query_terms)
            query_coverage = (
                float(query_hit_count) / float(total_query) if total_query > 0 else 0.0
            )

            if total_query > 0 and max_count_log > 0:
                query_tf_quality = query_tf_quality_sum / (
                    float(total_query) * float(max_count_log)
                )
            else:
                query_tf_quality = 0.0
            query_tf_quality = max(0.0, min(1.0, float(query_tf_quality)))

            if len(query_first_positions) >= 2:
                span = float(max(query_first_positions) - min(query_first_positions))
                span_norm = span / max(1.0, float(len(normalized_text) - 1))
                query_proximity = 1.0 - min(1.0, span_norm)
            elif len(query_first_positions) == 1:
                query_proximity = 0.5
            else:
                query_proximity = 0.0

            phrase_hit = 0.0
            if normalized_query and len(normalized_query) >= 2:
                phrase_pos = normalized_text.find(normalized_query)
                if phrase_pos >= 0:
                    phrase_hit = 1.0
                    all_hit_positions.append(phrase_pos)

            query_quality = (
                0.60 * query_coverage + 0.25 * query_tf_quality + 0.15 * query_proximity
            )

            base = (
                float(cfg.unique_hit_weight) * query_quality
                + float(cfg.count_weight) * phrase_hit
            )

            if query_coverage <= 0.0:
                base = 0.0
            elif query_coverage < 0.5 and phrase_hit <= 0.0:
                base *= 0.6

            if base <= 0.0 or not all_hit_positions:
                raw_scores.append(max(0.0, float(base)))
                continue

            earliest_pos = float(min(all_hit_positions))
            position_ratio = earliest_pos / max(1.0, float(len(normalized_text) - 1))
            position_ratio = max(0.0, min(1.0, position_ratio))
            early_gain = 1.0 + max(0.0, float(cfg.early_bonus) - 1.0) * (
                (1.0 - position_ratio) ** 2
            )
            raw_scores.append(max(0.0, float(base * early_gain)))

        return normalize_scores(raw_scores, cfg)


__all__ = ["HeuristicRanker"]
