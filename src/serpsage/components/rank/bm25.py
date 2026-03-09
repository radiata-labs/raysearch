from __future__ import annotations

from typing_extensions import override

from anyio import to_thread

try:
    from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

    BM25_AVAILABLE = True
except Exception:  # noqa: BLE001
    BM25Okapi = None
    BM25_AVAILABLE = False

from serpsage.components.base import ComponentMeta
from serpsage.components.rank.base import RankBm25Settings, RankerBase, RankMode
from serpsage.load import register_component
from serpsage.tokenize import tokenize

_BM25_META = ComponentMeta(
    family="rank",
    name="bm25",
    version="1.0.0",
    summary="BM25 text ranker.",
    provides=("rank.bm25_engine",),
    config_model=RankBm25Settings,
    config_optional=True,
)


@register_component(meta=_BM25_META)
class Bm25Ranker(RankerBase[RankBm25Settings]):
    meta = _BM25_META

    @override
    async def score_texts(
        self,
        texts: list[str],
        *,
        query: str,
        query_tokens: list[str],
        mode: RankMode = "retrieve",
    ) -> list[float]:
        _ = self._resolve_mode(mode, supported=("retrieve",))
        if not texts:
            return []
        if not BM25_AVAILABLE or BM25Okapi is None:
            return [0.0 for _ in texts]
        if not query_tokens:
            return [0.0 for _ in texts]
        corpus = [tokenize(doc) for doc in texts]
        bm25 = await to_thread.run_sync(BM25Okapi, corpus)
        scores = await to_thread.run_sync(bm25.get_scores, query_tokens)
        return [float(s) for s in scores]


__all__ = ["BM25_AVAILABLE", "Bm25Ranker"]
