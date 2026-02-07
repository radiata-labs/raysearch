from __future__ import annotations

from typing import Any

from search_core import SearchConfig, SearchPipeline


def main(query: str, max_results: int = 1) -> dict[str, Any]:
    """Convenience entry point for simple usage."""

    config_file = SearchConfig.load(
        "D:/WjjFiles/code/google-ai-overview-api/search_config.yaml"
    )
    pipeline = SearchPipeline(config_file)
    result = pipeline.search_markdown(
        query,
        "high",  # depth: simple|low|medium|high
        max_results=max_results,
        fuzzy_threshold=0.3,
        chunk_target_chars=200,
        chunk_overlap_sentences=1,
    )
    return {"search_result": result}


if __name__ == "__main__":
    print(main("初音ミク 新曲 2025", 5)["search_result"])
