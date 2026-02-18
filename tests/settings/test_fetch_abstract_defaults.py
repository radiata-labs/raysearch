from __future__ import annotations

from serpsage.settings.models import FetchAbstractSettings


def test_fetch_abstract_min_tokens_default_is_4() -> None:
    assert FetchAbstractSettings().min_abstract_tokens == 4


def test_fetch_abstract_max_chars_default_is_2000() -> None:
    assert FetchAbstractSettings().max_abstract_chars == 2000
