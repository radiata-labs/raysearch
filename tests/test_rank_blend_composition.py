from __future__ import annotations

from serpsage.components.rank.blend import BlendRanker
from serpsage.contracts.lifecycle import ClockBase
from serpsage.core.runtime import Runtime
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class FakeClock(ClockBase):
    def now_ms(self) -> int:
        return 0


def test_blend_ranker_scores_length_matches_inputs():
    settings = AppSettings.model_validate(
        {"rank": {"blend": {"providers": {"bm25": 0.7, "heuristic": 0.3}}}}
    )
    rt = Runtime(settings=settings, telemetry=NoopTelemetry(), clock=FakeClock())
    r = BlendRanker(rt=rt)

    docs = ["a", "b", "c"]
    scores = r.score_texts(texts=docs, query="python")
    assert isinstance(scores, list)
    assert len(scores) == len(docs)
