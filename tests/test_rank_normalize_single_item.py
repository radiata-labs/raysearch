from __future__ import annotations

from serpsage.components.rank.utils import normalize_scores
from serpsage.settings.models import NormalizationSettings


def test_normalize_scores_single_item_is_smooth_and_not_saturated():
    cfg = NormalizationSettings(
        single_item_method="sigmoid_log1p", single_item_scale=1.0
    )

    s0 = normalize_scores([0.0], cfg)[0]
    s1 = normalize_scores([1.0], cfg)[0]
    s10 = normalize_scores([10.0], cfg)[0]

    assert 0.0 <= s0 <= 0.05
    assert 0.0 < s1 < 1.0
    assert s1 < s10 < 1.0


def test_normalize_scores_flat_spread_is_neutral():
    cfg = NormalizationSettings()
    out = normalize_scores([1.0, 1.0, 1.0], cfg)
    assert out == [0.5, 0.5, 0.5]
