from __future__ import annotations

import pytest
from pydantic import ValidationError

from serpsage.app.request import (
    FetchChunksRequest,
    FetchContentRequest,
    FetchOverviewRequest,
)


def test_fetch_content_request_rejects_tag_overlap() -> None:
    with pytest.raises(ValidationError):
        FetchContentRequest(include_tags=["header"], exclude_tags=["header"])


def test_fetch_chunks_request_rejects_blank_query() -> None:
    with pytest.raises(ValidationError):
        FetchChunksRequest(query="   ")


def test_fetch_overview_request_rejects_blank_query() -> None:
    with pytest.raises(ValidationError):
        FetchOverviewRequest(query="   ")


def test_fetch_request_positive_constraints() -> None:
    with pytest.raises(ValidationError):
        FetchContentRequest(max_chars=0)
    with pytest.raises(ValidationError):
        FetchChunksRequest(query="alpha", max_chars=-1)
    with pytest.raises(ValidationError):
        FetchChunksRequest(query="alpha", top_k_chunks=0)
    with pytest.raises(ValidationError):
        FetchOverviewRequest(query="alpha", max_chars=0)


def test_fetch_query_fields_are_trimmed() -> None:
    chunks = FetchChunksRequest(query="  alpha beta  ")
    overview = FetchOverviewRequest(query="  gamma  ")
    assert chunks.query == "alpha beta"
    assert overview.query == "gamma"
