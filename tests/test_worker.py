from __future__ import annotations

import pytest

from scholar_agents.worker.consumer import (
    MAX_JOB_ATTEMPTS,
    PermanentJobError,
    is_permanent_error,
    should_retry,
)


@pytest.mark.parametrize(
    ("message", "expected"),
    [
        ("quota exceeded", True),
        ("insufficient balance", True),
        ("invalid api key", True),
        ("model not found", True),
        ("connection reset by peer", False),
    ],
)
def test_classifies_provider_errors_for_retry(message: str, expected: bool) -> None:
    assert is_permanent_error(RuntimeError(message)) is expected


def test_permanent_job_error_is_always_non_retryable() -> None:
    assert not should_retry(PermanentJobError("quota exceeded"), read_count=1)


def test_transient_job_retries_until_three_total_attempts() -> None:
    assert MAX_JOB_ATTEMPTS == 3
    assert should_retry(RuntimeError("temporary"), read_count=1)
    assert should_retry(RuntimeError("temporary"), read_count=2)
    assert not should_retry(RuntimeError("temporary"), read_count=3)
