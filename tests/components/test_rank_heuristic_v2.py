from __future__ import annotations

import asyncio
from types import SimpleNamespace

from serpsage.components.rank.heuristic import HeuristicRanker
from serpsage.components.rank.utils import normalize_scores
from serpsage.settings.models import AppSettings


def _base_settings() -> AppSettings:
    return AppSettings.model_validate(
        {
            "overview": {
                "enabled": True,
                "use_model": "gpt-4.1-mini",
                "models": [
                    {
                        "name": "gpt-4.1-mini",
                        "backend": "openai",
                        "model": "gpt-4.1-mini",
                        "api_key": "sk-test",
                    }
                ],
            }
        }
    )


def _ranker(settings: AppSettings) -> HeuristicRanker:
    rt = SimpleNamespace(settings=settings, telemetry=None, clock=None)
    return HeuristicRanker(rt=rt)


def _score(
    ranker: HeuristicRanker,
    *,
    texts: list[str],
    query: str,
    query_tokens: list[str],
    intent_tokens: list[str],
) -> list[float]:
    return asyncio.run(
        ranker.score_texts(
            texts=texts,
            query=query,
            query_tokens=query_tokens,
            intent_tokens=intent_tokens,
        )
    )


def test_early_bonus_prefers_earlier_hit() -> None:
    settings = _base_settings()
    settings.rank.heuristic.early_bonus = 1.5
    ranker = _ranker(settings)

    texts = [
        "iphone release date is officially announced",
        " ".join(["intro"] * 40) + " iphone release date is officially announced",
    ]
    scores = _score(
        ranker,
        texts=texts,
        query="iphone release date",
        query_tokens=["iphone", "release", "date"],
        intent_tokens=["officially"],
    )
    assert scores[0] > scores[1]


def test_normalize_scores_zero_scores_stay_zero() -> None:
    settings = _base_settings()
    normalized = normalize_scores([0.0, 0.0, 0.0, 5.0], settings.rank.heuristic)
    assert normalized == [0.0, 0.0, 0.0, 1.0]


def test_unmatched_text_gets_zero_score() -> None:
    settings = _base_settings()
    ranker = _ranker(settings)
    scores = _score(
        ranker,
        texts=["completely unrelated content"],
        query="iphone release date",
        query_tokens=["iphone", "release", "date"],
        intent_tokens=["official"],
    )
    assert scores == [0.0]


def test_token_stuffing_does_not_beat_high_coverage_text() -> None:
    settings = _base_settings()
    ranker = _ranker(settings)
    texts = [
        " ".join(["iphone"] * 40),
        "iphone release date and price details are now available",
    ]
    scores = _score(
        ranker,
        texts=texts,
        query="iphone release date price",
        query_tokens=["iphone", "release", "date", "price"],
        intent_tokens=[],
    )
    assert scores[1] > scores[0]


def test_intent_tokens_add_bonus() -> None:
    settings = _base_settings()
    ranker = _ranker(settings)
    texts = [
        "iphone release date is rumored",
        "official iphone release date is rumored",
    ]
    scores = _score(
        ranker,
        texts=texts,
        query="iphone release",
        query_tokens=["iphone", "release"],
        intent_tokens=["official"],
    )
    assert scores[1] > scores[0]


def test_normalize_scores_stays_in_mid_range_for_close_values() -> None:
    settings = _base_settings()
    values = [1.0, 1.1, 1.2, 1.3, 1.4, 1.5]
    normalized = normalize_scores(values, settings.rank.heuristic)
    assert min(normalized) > 0.0
    assert max(normalized) < 1.0
    assert len(set(normalized)) > 1


def test_single_positive_score_maps_to_one() -> None:
    settings = _base_settings()
    normalized = normalize_scores([0.0, 0.0, 3.0, 0.0], settings.rank.heuristic)
    assert normalized == [0.0, 0.0, 1.0, 0.0]


def test_rank_normalization_block_is_ignored_and_heuristic_fields_work() -> None:
    settings = AppSettings.model_validate(
        {
            "overview": {
                "enabled": True,
                "use_model": "gpt-4.1-mini",
                "models": [{"name": "gpt-4.1-mini", "backend": "openai"}],
            },
            "rank": {
                "normalization": {
                    "method": "rank",
                    "single_item_method": "exp",
                },
                "heuristic": {
                    "temperature": 0.75,
                    "min_items_for_sigmoid": 3,
                    "flat_spread_eps": 1e-8,
                    "z_clip": 6.0,
                },
            },
        }
    )
    assert settings.rank.heuristic.temperature == 0.75
    assert settings.rank.heuristic.min_items_for_sigmoid == 3
    assert settings.rank.heuristic.flat_spread_eps == 1e-8
    assert settings.rank.heuristic.z_clip == 6.0
    assert not hasattr(settings.rank, "normalization")
