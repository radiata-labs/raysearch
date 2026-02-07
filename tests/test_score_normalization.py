from __future__ import annotations

from search_core.config import ScoreNormalizationConfig
from search_core.scoring import normalize_scores


def test_normalize_scores_bounds_robust_sigmoid():
    cfg = ScoreNormalizationConfig(method="robust_sigmoid")
    out = normalize_scores([-10.0, 0.0, 5.0, 5.0, 100.0], cfg)
    assert all(0.0 <= x <= 1.0 for x in out)


def test_normalize_scores_monotonic():
    cfg = ScoreNormalizationConfig(method="robust_sigmoid")
    scores = [-3.0, -1.0, 0.0, 2.0, 9.0]
    out = normalize_scores(scores, cfg)
    assert all(out[i] <= out[i + 1] for i in range(len(out) - 1))


def test_normalize_scores_small_n_fallback_to_rank():
    cfg = ScoreNormalizationConfig(method="robust_sigmoid", min_items_for_sigmoid=5)
    out = normalize_scores([10.0, 5.0, 5.0], cfg)
    assert out[0] == 1.0
    assert out[1] == out[2]


def test_normalize_scores_flat_distribution_returns_zeros():
    cfg = ScoreNormalizationConfig(method="robust_sigmoid")
    out = normalize_scores([1.0, 1.0, 1.0, 1.0, 1.0], cfg)
    assert out == [0.0, 0.0, 0.0, 0.0, 0.0]
