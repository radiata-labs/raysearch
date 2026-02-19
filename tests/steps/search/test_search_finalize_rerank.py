from __future__ import annotations

import anyio

from serpsage.app.bootstrap import build_runtime
from serpsage.app.request import SearchRequest
from serpsage.app.response import FetchResultItem, FetchSubpagesResult
from serpsage.components.rank.base import RankerBase
from serpsage.models.pipeline import (
    SearchFetchedCandidate,
    SearchFetchState,
    SearchStepContext,
)
from serpsage.settings.models import AppSettings
from serpsage.steps.search.finalize import SearchFinalizeStep


class _KeywordRanker(RankerBase):
    async def score_texts(
        self,
        *,
        texts: list[str],
        query: str,
        query_tokens: list[str],
    ) -> list[float]:
        del query, query_tokens
        out = []
        for text in texts:
            lower = text.lower()
            if "high" in lower:
                out.append(0.95)
            elif "mid" in lower:
                out.append(0.60)
            elif "low" in lower:
                out.append(0.20)
            else:
                out.append(0.10)
        return out


def _candidate(
    *,
    url: str,
    main_text: str = "",
    main_abstract_scores: list[float] | None = None,
    main_overview_scores: list[float] | None = None,
    subpage_texts: list[str] | None = None,
    subpage_abstract_scores: list[list[float]] | None = None,
    subpage_overview_scores: list[list[float]] | None = None,
) -> SearchFetchedCandidate:
    subpage_abstract_scores = list(subpage_abstract_scores or [])
    subpages = [
        FetchSubpagesResult(
            url=f"{url}/sub{i + 1}",
            title=f"sub{i + 1}",
            content="",
            abstracts=[
                f"sub{i + 1}-abs{j + 1}" for j, _ in enumerate(list(scores))
            ],
            abstract_scores=[float(x) for x in list(scores)],
        )
        for i, scores in enumerate(subpage_abstract_scores)
    ]
    main_abstract_scores = [float(x) for x in list(main_abstract_scores or [])]
    return SearchFetchedCandidate(
        result=FetchResultItem(
            url=url,
            title=url,
            content="",
            abstracts=[f"main-abs{i + 1}" for i, _ in enumerate(main_abstract_scores)],
            abstract_scores=main_abstract_scores,
            subpages=subpages,
        ),
        main_md_for_abstract=main_text,
        subpages_md_for_abstract=[str(x) for x in list(subpage_texts or [])],
        main_overview_scores=[float(x) for x in list(main_overview_scores or [])],
        subpages_overview_scores=[
            [float(x) for x in list(scores)]
            for scores in list(subpage_overview_scores or [])
        ],
    )


def _run(
    *,
    request: SearchRequest,
    candidates: list[SearchFetchedCandidate],
) -> list[str]:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    ranker = _KeywordRanker(rt=rt)
    step = SearchFinalizeStep(rt=rt, ranker=ranker)
    ctx = SearchStepContext(
        settings=settings,
        request=request,
        fetch=SearchFetchState(candidates=candidates),
    )
    out = anyio.run(step.run, ctx)
    return [item.url for item in out.output.results]


def test_finalize_only_abstracts_uses_top3_avg_and_subpage_max() -> None:
    urls = _run(
        request=SearchRequest(
            query="q",
            max_results=2,
            fetchs={"abstracts": True},
        ),
        candidates=[
            _candidate(
                url="https://a.com",
                main_abstract_scores=[0.9, 0.6, 0.3],
                subpage_abstract_scores=[[0.5, 0.5, 0.5]],
            ),
            _candidate(
                url="https://b.com",
                main_abstract_scores=[0.4, 0.4, 0.4],
                subpage_abstract_scores=[[0.95, 0.9, 0.85]],
            ),
        ],
    )

    assert urls == ["https://b.com", "https://a.com"]


def test_finalize_only_overview_uses_top3_avg_and_subpage_max() -> None:
    urls = _run(
        request=SearchRequest(
            query="q",
            max_results=2,
            fetchs={"overview": True},
        ),
        candidates=[
            _candidate(
                url="https://a.com",
                main_overview_scores=[0.3, 0.3, 0.3],
                subpage_overview_scores=[[0.9, 0.9, 0.6]],
                subpage_abstract_scores=[[]],
            ),
            _candidate(
                url="https://b.com",
                main_overview_scores=[0.7, 0.6, 0.5],
            ),
        ],
    )

    assert urls == ["https://a.com", "https://b.com"]


def test_finalize_only_content_keeps_content_semantic_sorting() -> None:
    urls = _run(
        request=SearchRequest(
            query="q",
            max_results=2,
            fetchs={"content": True},
        ),
        candidates=[
            _candidate(
                url="https://a.com",
                main_text="main low",
                subpage_texts=["sub high"],
                subpage_abstract_scores=[[]],
            ),
            _candidate(
                url="https://b.com",
                main_text="main mid",
            ),
        ],
    )

    assert urls == ["https://a.com", "https://b.com"]


def test_finalize_multi_feature_uses_page_average_then_subpage_max() -> None:
    urls = _run(
        request=SearchRequest(
            query="q",
            max_results=2,
            fetchs={"content": True, "abstracts": True, "overview": True},
        ),
        candidates=[
            _candidate(
                url="https://a.com",
                main_text="main high",
                main_abstract_scores=[0.9, 0.9, 0.9],
                main_overview_scores=[0.3, 0.3, 0.3],
                subpage_texts=["sub mid"],
                subpage_abstract_scores=[[0.7, 0.7, 0.7]],
                subpage_overview_scores=[[0.7, 0.7, 0.7]],
            ),
            _candidate(
                url="https://b.com",
                main_text="main mid",
                main_abstract_scores=[0.8, 0.8, 0.8],
                main_overview_scores=[0.8, 0.8, 0.8],
            ),
        ],
    )

    assert urls == ["https://b.com", "https://a.com"]


def test_finalize_required_empty_component_forces_zero_score() -> None:
    urls = _run(
        request=SearchRequest(
            query="q",
            max_results=2,
            fetchs={"content": True, "abstracts": True},
        ),
        candidates=[
            _candidate(
                url="https://zero.com",
                main_text="main high",
                main_abstract_scores=[],
            ),
            _candidate(
                url="https://ok.com",
                main_text="main mid",
                main_abstract_scores=[0.5, 0.5, 0.5],
            ),
        ],
    )

    assert urls == ["https://ok.com", "https://zero.com"]


def test_finalize_applies_text_filters_only_on_main_page() -> None:
    urls = _run(
        request=SearchRequest(
            query="q",
            max_results=2,
            include_text=["  KEEP   phrase  "],
            exclude_text=["BLOCKED"],
            fetchs={"content": True},
        ),
        candidates=[
            _candidate(
                url="https://ok.com",
                main_text="keep phrase",
                subpage_texts=["blocked high"],
                subpage_abstract_scores=[[]],
            ),
            _candidate(
                url="https://drop.com",
                main_text="keep phrase blocked",
            ),
        ],
    )

    assert urls == ["https://ok.com"]


def test_finalize_when_no_sort_feature_enabled_keeps_prefetch_order() -> None:
    urls = _run(
        request=SearchRequest(
            query="q",
            max_results=2,
            fetchs={"others": {"max_links": 1}},
        ),
        candidates=[
            _candidate(
                url="https://first.com",
                main_text="main low",
            ),
            _candidate(
                url="https://second.com",
                main_text="main high",
            ),
        ],
    )

    assert urls == ["https://first.com", "https://second.com"]
