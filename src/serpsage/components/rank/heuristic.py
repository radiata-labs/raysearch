from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.components.rank.utils import normalize_scores
from serpsage.contracts.services import RankerBase
from serpsage.text.normalize import normalize_text

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


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
        intent_tokens: list[str],
    ) -> list[float]:
        cfg = self.settings.rank.heuristic

        out: list[float] = []
        for text in texts:
            normalized_text = normalize_text(text)
            if not normalized_text:
                out.append(0.0)
                continue

            unique_hits: set[str] = set()
            count_hits = 0
            for t in query_tokens:
                tl = t.lower()
                if tl and tl in normalized_text:
                    unique_hits.add(tl)
                    count_hits += min(
                        normalized_text.count(tl), int(cfg.max_count_per_token)
                    )

            intent_hits = 0
            for t in intent_tokens:
                tl = (t or "").lower()
                if tl and tl in normalized_text:
                    intent_hits += 1

            score = 0.0
            score += float(cfg.unique_hit_weight) * float(len(unique_hits))
            score += float(cfg.count_weight) * float(count_hits)
            score += float(cfg.intent_hit_weight) * float(intent_hits)

            out.append(float(score))

        return normalize_scores(out, self.settings.rank.normalization)


__all__ = ["HeuristicRanker"]
