from __future__ import annotations

from typing import TYPE_CHECKING
from typing_extensions import override

from serpsage.contracts.services import RankerBase
from serpsage.text.normalize import normalize_text
from serpsage.text.tokenize import tokenize

if TYPE_CHECKING:
    from serpsage.core.runtime import Runtime


class HeuristicRanker(RankerBase):
    def __init__(self, *, rt: Runtime) -> None:
        super().__init__(rt=rt)

    @override
    def score_texts(
        self,
        *,
        texts: list[str],
        query: str,
        query_tokens: list[str] | None = None,
        intent_tokens: list[str] | None = None,
    ) -> list[float]:
        cfg = self.settings.rank.heuristic
        q_tokens = query_tokens if query_tokens is not None else tokenize(query)
        q_tokens = [t for t in (q_tokens or []) if len(t) >= int(cfg.min_token_len)]
        i_tokens = intent_tokens or []

        normalized_query = normalize_text(query)
        out: list[float] = []
        for text in texts:
            normalized_text = normalize_text(text)
            if not normalized_text:
                out.append(0.0)
                continue

            unique_hits: set[str] = set()
            count_hits = 0
            for t in q_tokens:
                tl = t.lower()
                if tl and tl in normalized_text:
                    unique_hits.add(tl)
                    count_hits += min(
                        normalized_text.count(tl), int(cfg.max_count_per_token)
                    )

            intent_hits = 0
            for t in i_tokens:
                tl = (t or "").lower()
                if tl and tl in normalized_text:
                    intent_hits += 1

            score = 0.0
            score += float(cfg.unique_hit_weight) * float(len(unique_hits))
            score += float(cfg.count_weight) * float(count_hits)
            score += float(cfg.intent_hit_weight) * float(intent_hits)

            if normalized_query and normalized_query in normalized_text:
                score += float(cfg.phrase_bonus)

            out.append(float(score))

        return out

    @override
    def normalize(self, *, scores: list[float]) -> list[float]:
        # The combiner owns normalization.
        return list(scores or [])


__all__ = ["HeuristicRanker"]
