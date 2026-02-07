from __future__ import annotations

from search_core.scoring import rank_scale


def test_bm25_rank_scale_is_stable():
    # Flat scores => no signal.
    assert rank_scale([1.0, 1.0, 1.0]) == [0.0, 0.0, 0.0]

    scaled = rank_scale([0.1, 0.2, 0.15])
    assert max(scaled) == scaled[1]
    assert min(scaled) == scaled[0]
