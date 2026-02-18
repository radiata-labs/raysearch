from __future__ import annotations

import anyio

from serpsage.app.bootstrap import build_runtime
from serpsage.app.request import SearchRequest
from serpsage.app.response import FetchResultItem, FetchSubpagesResult
from serpsage.components.rank.base import RankerBase
from serpsage.models.pipeline import SearchFetchedCandidate, SearchStepContext
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


def test_finalize_uses_best_score_among_page_and_subpages() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    ranker = _KeywordRanker(rt=rt)
    step = SearchFinalizeStep(rt=rt, ranker=ranker)
    ctx = SearchStepContext(
        settings=settings,
        request=SearchRequest(query="q", max_results=2, fetchs={"content": True}),
        fetched_candidates=[
            SearchFetchedCandidate(
                result=FetchResultItem(
                    url="https://a.com",
                    title="a",
                    content="",
                    abstracts=[],
                    abstract_scores=[],
                    subpages=[
                        FetchSubpagesResult(
                            url="https://a.com/sub",
                            title="sub",
                            content="",
                            abstracts=[],
                            abstract_scores=[],
                        )
                    ],
                ),
                main_md_for_abstract="main low",
                subpages_md_for_abstract=["sub high"],
            ),
            SearchFetchedCandidate(
                result=FetchResultItem(
                    url="https://b.com",
                    title="b",
                    content="",
                    abstracts=[],
                    abstract_scores=[],
                ),
                main_md_for_abstract="main mid",
                subpages_md_for_abstract=[],
            ),
        ],
    )

    out = anyio.run(step.run, ctx)

    assert [item.url for item in out.results] == ["https://a.com", "https://b.com"]


def test_finalize_applies_text_filters_only_on_main_page() -> None:
    settings = AppSettings()
    rt = build_runtime(settings=settings)
    ranker = _KeywordRanker(rt=rt)
    step = SearchFinalizeStep(rt=rt, ranker=ranker)
    ctx = SearchStepContext(
        settings=settings,
        request=SearchRequest(
            query="q",
            max_results=2,
            include_text=["  KEEP   phrase  "],
            exclude_text=["BLOCKED"],
            fetchs={"content": True},
        ),
        fetched_candidates=[
            SearchFetchedCandidate(
                result=FetchResultItem(
                    url="https://ok.com",
                    title="ok",
                    content="",
                    abstracts=[],
                    abstract_scores=[],
                    subpages=[
                        FetchSubpagesResult(
                            url="https://ok.com/sub",
                            title="sub",
                            content="",
                            abstracts=[],
                            abstract_scores=[],
                        )
                    ],
                ),
                main_md_for_abstract="keep phrase",
                subpages_md_for_abstract=["blocked high"],
            ),
            SearchFetchedCandidate(
                result=FetchResultItem(
                    url="https://drop.com",
                    title="drop",
                    content="",
                    abstracts=[],
                    abstract_scores=[],
                ),
                main_md_for_abstract="keep phrase blocked",
                subpages_md_for_abstract=[],
            ),
        ],
    )

    out = anyio.run(step.run, ctx)

    assert [item.url for item in out.results] == ["https://ok.com"]
