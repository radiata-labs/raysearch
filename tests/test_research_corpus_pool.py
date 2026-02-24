from __future__ import annotations

from serpsage.app.request import ResearchRequest
from serpsage.models.pipeline import ResearchRoundState, ResearchStepContext
from serpsage.settings.models import AppSettings
from serpsage.steps.research.search import (
    append_source_version,
    rebuild_corpus_ranking,
    select_context_source_ids,
)


def _build_ctx(*, themes: str, round_index: int, queries: list[str]) -> ResearchStepContext:
    ctx = ResearchStepContext(
        settings=AppSettings(),
        request=ResearchRequest(themes=themes),
        request_id="test-research-corpus",
    )
    ctx.current_round = ResearchRoundState(
        round_index=int(round_index),
        queries=list(queries),
    )
    return ctx


def test_append_only_with_same_round_fingerprint_dedup() -> None:
    ctx = _build_ctx(themes="python async", round_index=1, queries=["python async"])
    first = append_source_version(
        ctx=ctx,
        url="https://example.com/page?utm_source=x",
        title="Async Intro",
        abstracts=["Python async basics"],
        content="A content body",
        round_index=1,
        is_subpage=False,
        ingest_query="python async",
        ingest_intent="coverage",
    )
    second = append_source_version(
        ctx=ctx,
        url="https://example.com/page?utm_medium=foo",
        title="Async Intro",
        abstracts=["Python async basics"],
        content="A content body",
        round_index=1,
        is_subpage=False,
        ingest_query="python async",
        ingest_intent="coverage",
    )
    third = append_source_version(
        ctx=ctx,
        url="https://example.com/page",
        title="Async Intro v2",
        abstracts=["Python async details"],
        content="A different content body",
        round_index=1,
        is_subpage=False,
        ingest_query="python async details",
        ingest_intent="deepen",
    )

    assert first.is_new_canonical is True
    assert first.is_new_version is True
    assert second.is_new_canonical is False
    assert second.is_new_version is False
    assert third.is_new_canonical is False
    assert third.is_new_version is True
    assert len(ctx.corpus.sources) == 2

    first_source = ctx.corpus.sources[0]
    second_source = ctx.corpus.sources[1]
    assert first_source.seen_count == 2
    assert first_source.url_version == 1
    assert second_source.url_version == 2
    assert first_source.canonical_url == second_source.canonical_url
    assert ctx.corpus.source_url_to_id[first_source.canonical_url] == second_source.source_id
    assert ctx.corpus.source_url_to_ids[first_source.canonical_url] == [
        first_source.source_id,
        second_source.source_id,
    ]


def test_rank_and_window_keep_new_plus_history() -> None:
    ctx = _build_ctx(themes="python async", round_index=1, queries=["python async"])
    append_source_version(
        ctx=ctx,
        url="https://docs.python.org/3/library/asyncio.html",
        title="Asyncio Docs",
        abstracts=["Official asyncio docs"],
        content="Python asyncio event loop create_task await gather.",
        round_index=1,
        is_subpage=False,
        ingest_query="python async",
        ingest_intent="coverage",
    )
    append_source_version(
        ctx=ctx,
        url="https://example.org/blog/async-patterns",
        title="Async Patterns",
        abstracts=["Practical async patterns"],
        content="Async queue semaphore cancellation patterns.",
        round_index=1,
        is_subpage=False,
        ingest_query="python async",
        ingest_intent="deepen",
    )

    ctx.current_round = ResearchRoundState(
        round_index=2,
        queries=["python async latest"],
    )
    new_source = append_source_version(
        ctx=ctx,
        url="https://realpython.com/async-io-python/",
        title="Async IO Latest Guide",
        abstracts=["Latest async IO guide"],
        content="Latest async Python guide for 2026 with task groups and structured concurrency.",
        round_index=2,
        is_subpage=False,
        ingest_query="python async latest",
        ingest_intent="refresh",
    )
    gain = rebuild_corpus_ranking(ctx=ctx, round_index=2)

    ranked = list(ctx.corpus.ranked_source_ids)
    assert ranked
    assert len(ranked) == len(set(ranked))
    assert all(source_id in ctx.corpus.source_scores for source_id in ranked)
    assert gain >= 0.0

    selected = select_context_source_ids(
        ctx=ctx,
        round_index=2,
        topk=2,
        new_result_target_ratio=1.0,
        min_history_sources=1,
    )
    assert len(selected) == 2
    assert int(new_source.source_id) in selected

    source_by_id = {item.source_id: item for item in ctx.corpus.sources}
    history_count = sum(
        1 for source_id in selected if int(source_by_id[source_id].round_index) < 2
    )
    assert history_count >= 1
