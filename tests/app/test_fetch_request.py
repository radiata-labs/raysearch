from __future__ import annotations

import pytest

from pydantic import ValidationError

from serpsage.app.request import (
    FetchAbstractsRequest,
    FetchContentRequest,
    FetchOverviewRequest,
    FetchRequest,
)


def test_fetch_request_defaults_raise_when_nothing_to_do() -> None:
    with pytest.raises(ValidationError, match="nothing to do"):
        FetchRequest(urls=["https://example.com"])


def test_fetch_request_accepts_bool_switches() -> None:
    req = FetchRequest(
        urls=["https://example.com"],
        content=True,
        abstracts=True,
        overview=True,
    )
    assert req.content is True
    assert req.abstracts is True
    assert req.overview is True


def test_fetch_request_content_true_is_minimal_valid_action() -> None:
    req = FetchRequest(
        urls=["https://example.com"],
        content=True,
        abstracts=False,
        overview=False,
    )
    assert req.content is True


def test_fetch_queries_normalize_empty_to_none() -> None:
    assert FetchAbstractsRequest(query="  ").query is None
    assert FetchOverviewRequest(query="  ").query is None


def test_fetch_content_detail_default_is_concise() -> None:
    assert FetchContentRequest().detail == "concise"
