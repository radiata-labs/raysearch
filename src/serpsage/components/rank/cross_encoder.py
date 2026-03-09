from __future__ import annotations

import math
from typing import Any, cast
from typing_extensions import override

from anyio import to_thread

try:
    from sentence_transformers import CrossEncoder as _ImportedCrossEncoder

    _CROSS_ENCODER_CTOR: Any | None = _ImportedCrossEncoder
    CROSS_ENCODER_AVAILABLE = True
except Exception:  # noqa: BLE001
    _CROSS_ENCODER_CTOR = None
    CROSS_ENCODER_AVAILABLE = False

from serpsage.components.base import ComponentMeta
from serpsage.components.rank.base import (
    RankCrossEncoderSettings,
    RankerBase,
    RankMode,
)
from serpsage.components.registry import register_component
from serpsage.utils import clean_whitespace


def _sigmoid(value: float) -> float:
    if math.isnan(value) or math.isinf(value):
        return 0.0
    if value >= 0.0:
        exp_value = math.exp(-value)
        return float(1.0 / (1.0 + exp_value))
    exp_value = math.exp(value)
    return float(exp_value / (1.0 + exp_value))


def _coerce_prediction_value(value: object) -> float:
    if isinstance(value, (list, tuple)):
        if not value:
            return 0.0
        return _coerce_prediction_value(value[0])
    try:
        return _sigmoid(float(cast("Any", value)))
    except Exception:
        return 0.0


def _normalize_predictions(raw: object) -> list[float]:
    if hasattr(raw, "tolist"):
        raw = cast("Any", raw).tolist()
    if isinstance(raw, (list, tuple)):
        return [_coerce_prediction_value(item) for item in raw]
    return [_coerce_prediction_value(raw)]


_CROSS_ENCODER_META = ComponentMeta(
    family="rank",
    name="cross_encoder",
    version="1.0.0",
    summary="Cross-encoder ranker.",
    provides=("rank.cross_encoder_engine",),
    config_model=RankCrossEncoderSettings,
)


@register_component(meta=_CROSS_ENCODER_META)
class CrossEncoderRanker(RankerBase[RankCrossEncoderSettings]):
    meta = _CROSS_ENCODER_META

    def __init__(
        self,
        *,
        rt: object,
        config: RankCrossEncoderSettings,
    ) -> None:
        super().__init__(rt=rt, config=config)
        self._model: object | None = None

    @override
    async def on_init(self) -> None:
        await self._ensure_model()

    async def _ensure_model(self) -> object:
        if self._model is not None:
            return self._model
        if not CROSS_ENCODER_AVAILABLE or _CROSS_ENCODER_CTOR is None:
            raise RuntimeError(
                "cross-encoder ranker is unavailable: install sentence-transformers"
            )
        cross_encoder_ctor = cast("Any", _CROSS_ENCODER_CTOR)
        self._model = await to_thread.run_sync(
            lambda: cross_encoder_ctor(
                str(self.config.model_name or "").strip(),
                max_length=max(1, int(self.config.max_length)),
            )
        )
        return self._model

    @override
    async def score_texts(
        self,
        texts: list[str],
        *,
        query: str,
        query_tokens: list[str],
        mode: RankMode = "retrieve",
    ) -> list[float]:
        _ = self._resolve_mode(mode, supported=("retrieve", "rerank"))
        if not texts:
            return []
        effective_query = clean_whitespace(query) or " ".join(query_tokens)
        if not effective_query:
            return [0.0 for _ in texts]
        model = await self._ensure_model()
        batch_size = max(1, int(self.config.batch_size))
        pairs = [(effective_query, text) for text in texts]
        predictions = await to_thread.run_sync(
            lambda: cast("Any", model).predict(
                pairs,
                batch_size=batch_size,
                show_progress_bar=False,
            )
        )
        scores = _normalize_predictions(predictions)
        return [
            float(scores[index]) if index < len(scores) else 0.0
            for index in range(len(texts))
        ]


__all__ = ["CROSS_ENCODER_AVAILABLE", "CrossEncoderRanker"]
