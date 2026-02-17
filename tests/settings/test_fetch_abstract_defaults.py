from __future__ import annotations

from serpsage.settings.models import FetchAbstractSettings


def test_fetch_abstract_min_chars_default_is_8() -> None:
    assert FetchAbstractSettings().min_abstract_chars == 8
