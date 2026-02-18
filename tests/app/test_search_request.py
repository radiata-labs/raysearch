from __future__ import annotations

import pytest

from pydantic import ValidationError

from serpsage.app.request import SearchRequest


def test_search_request_requires_fetchs() -> None:
    with pytest.raises(ValidationError):
        SearchRequest(query="latest ai")


def test_search_request_rejects_additional_queries_when_depth_auto() -> None:
    with pytest.raises(ValidationError, match="depth=deep"):
        SearchRequest(
            query="latest ai",
            depth="auto",
            additional_queries=["llm"],
            fetchs={"content": True},
        )


def test_search_request_rejects_domain_overlap() -> None:
    with pytest.raises(ValidationError, match="must not overlap"):
        SearchRequest(
            query="latest ai",
            depth="deep",
            include_domains=["arxiv.org"],
            exclude_domains=["arxiv.org"],
            fetchs={"content": True},
        )


def test_search_request_rejects_too_long_word_phrase() -> None:
    with pytest.raises(ValidationError, match="at most 5 words"):
        SearchRequest(
            query="latest ai",
            include_text=["one two three four five six"],
            fetchs={"content": True},
        )


def test_search_request_rejects_too_long_cjk_phrase() -> None:
    with pytest.raises(ValidationError, match="at most 6 Chinese/Japanese"):
        SearchRequest(
            query="latest ai",
            include_text=["这是超过六个汉字的短语"],
            fetchs={"content": True},
        )
