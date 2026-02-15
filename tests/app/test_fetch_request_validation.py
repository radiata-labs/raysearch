from __future__ import annotations

import pytest
from pydantic import ValidationError

from serpsage.app.request import (
    FetchAbstractsRequest,
    FetchContentRequest,
    FetchOthersRequest,
    FetchOverviewRequest,
    FetchRequest,
    SearchOverviewRequest,
)


def test_fetch_content_request_rejects_tag_overlap() -> None:
    with pytest.raises(ValidationError):
        FetchContentRequest(include_tags=["header"], exclude_tags=["header"])


def test_fetch_abstracts_request_rejects_blank_query() -> None:
    with pytest.raises(ValidationError):
        FetchAbstractsRequest(query="   ")


def test_fetch_overview_request_rejects_blank_query() -> None:
    with pytest.raises(ValidationError):
        FetchOverviewRequest(query="   ")


def test_fetch_request_positive_constraints() -> None:
    with pytest.raises(ValidationError):
        FetchContentRequest(max_chars=0)
    with pytest.raises(ValidationError):
        FetchAbstractsRequest(query="alpha", max_chars=-1)
    with pytest.raises(ValidationError):
        FetchAbstractsRequest(query="alpha", top_k_abstracts=0)
    with pytest.raises(ValidationError):
        FetchOthersRequest(max_links=0)
    with pytest.raises(ValidationError):
        FetchOthersRequest(max_image_links=0)
    with pytest.raises(ValidationError):
        FetchRequest(
            urls=["https://example.com"],
            content=True,
            crawl_timeout=0,
        )


def test_fetch_query_fields_are_trimmed() -> None:
    abstracts = FetchAbstractsRequest(query="  alpha beta  ")
    overview = FetchOverviewRequest(query="  gamma  ")
    assert abstracts.query == "alpha beta"
    assert overview.query == "gamma"


def test_json_schema_validation() -> None:
    with pytest.raises(ValidationError):
        FetchOverviewRequest(query="alpha", json_schema={"type": "wat"})
    with pytest.raises(ValidationError):
        SearchOverviewRequest(json_schema=["not", "object"])


def test_fetch_request_urls_validation() -> None:
    with pytest.raises(ValidationError):
        FetchRequest(urls=[], content=True)
    with pytest.raises(ValidationError):
        FetchRequest(urls=[""], content=True)
    with pytest.raises(ValidationError):
        FetchRequest(urls=["ftp://example.com"], content=True)

    req = FetchRequest(
        urls=[" https://example.com ", "http://example.org/a"],
        content=True,
    )
    assert req.urls == ["https://example.com", "http://example.org/a"]
