from __future__ import annotations

import json
from typing import Any

from search_core import SearchConfig, SearchPipeline


def main(query: str, max_results: int = 1) -> dict[str, Any]:
    """Convenience entry point for simple usage."""

    cfg = SearchConfig.load()
    pipeline = SearchPipeline(cfg)
    result = json.dumps(
        pipeline.search_json(
            query,
            "low",  # depth: simple|low|medium|high
            max_results=max_results,
            # fuzzy_threshold=0.3,
            chunk_target_chars=1200,
            chunk_overlap_sentences=1,
        ),
        indent=2,
        ensure_ascii=False,
    )
    return {"search_result": result}


if __name__ == "__main__":
    print(main("初音ミク 新曲 2025", 5)["search_result"])
