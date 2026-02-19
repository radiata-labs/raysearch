from __future__ import annotations

import pytest
from pydantic import ValidationError

from serpsage.settings.models import AppSettings, RunnerSettings


def test_runner_queue_size_default() -> None:
    runner = RunnerSettings()
    assert runner.queue_size == 256
    assert AppSettings().runner.queue_size == 256


def test_runner_queue_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        RunnerSettings(queue_size=0)
