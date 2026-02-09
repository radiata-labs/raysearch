from __future__ import annotations

from serpsage.app.runtime import CoreRuntime
from serpsage.rank.blend import BlendRanker
from serpsage.settings.models import AppSettings
from serpsage.telemetry.trace import NoopTelemetry


class FakeClock:
    def now_ms(self) -> int:
        return 0


def test_blend_ranker_scores_length_matches_inputs():
    settings = AppSettings.model_validate(
        {"rank": {"providers": {"bm25": 0.7, "heuristic": 0.3}}}
    )
    rt = CoreRuntime(settings=settings, telemetry=NoopTelemetry(), clock=FakeClock())
    r = BlendRanker(rt=rt)

    docs = ["a", "b", "c"]
    scores = r.score_texts(texts=docs, query="python")
    assert isinstance(scores, list)
    assert len(scores) == len(docs)

