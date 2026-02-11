from __future__ import annotations

import pytest
from pydantic import ValidationError

from serpsage.settings.models import AppSettings


def test_removed_fetch_common_keys_fail_fast() -> None:
    with pytest.raises(ValidationError, match="validate_extractable"):
        AppSettings.model_validate(
            {
                "enrich": {
                    "fetch": {
                        "common": {
                            "validate_extractable": True,
                        }
                    }
                }
            }
        )

