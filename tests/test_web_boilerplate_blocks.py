from __future__ import annotations

from search_core.config import SearchConfig, SearchContextConfig
from search_core.pipeline import SearchPipeline


def test_web_boilerplate_blocks_are_filtered_before_chunking():
    profile_cfg = SearchContextConfig.model_validate(
        {
            "ranking": {
                "strategy": "hybrid",
                "min_relevance_score": 1,
                "min_intent_score": 1,
            }
        }
    )
    cfg = SearchConfig(default_profile="general", profiles={"general": profile_cfg})

    html = """<html><body>
<nav>アーカイブ 月を選択 カテゴリ 次へ 前へ</nav>
<div>2026年2月 2026年1月 2025年12月</div>
<article>
<p>本文: 初音ミク の新作MV が公開された。</p>
<p>詳細: 透明感のある楽曲とマッチしたステキなMV。</p>
</article>
<footer>privacy terms sitemap</footer>
</body></html>"""

    def fetcher(_url: str) -> str:
        return html

    pipeline = SearchPipeline(cfg, page_fetcher=fetcher)
    response = {
        "results": [
            {
                "url": "https://blog.example/x",
                "title": "初音ミク 新作MV",
                "snippet": "初音ミク MV 公開",
                "engine": "x",
            }
        ]
    }

    ctx = pipeline.build_context(
        response,
        "初音ミク MV",
        "low",
        max_results=1,
        chunk_target_chars=200,
        chunk_overlap_sentences=0,
        min_chunk_chars=1,
    )

    assert ctx.results[0].page.chunks
    joined = "\n".join(c.text for c in ctx.results[0].page.chunks)
    assert "アーカイブ" not in joined
    assert "privacy" not in joined.lower()
    assert "初音ミク" in joined
