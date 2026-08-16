from __future__ import annotations

import time
from datetime import UTC, datetime
from uuid import uuid4

from scholar_agents.errors import normalize_provider_error
from scholar_agents.job_context import (
    JobContext,
    child_payload,
    reset_current_job,
    set_current_job,
)


class FakeResponse:
    def __init__(self, retry_after: str | None = None) -> None:
        self.headers = {"retry-after": retry_after} if retry_after else {}


class FakeSDKError(Exception):
    def __init__(
        self,
        *,
        status_code: int | None,
        body: dict[str, object] | None = None,
        retry_after: str | None = None,
    ) -> None:
        super().__init__("provider failed")
        self.status_code = status_code
        self.body = body
        self.response = FakeResponse(retry_after)


def test_generic_429_is_retryable() -> None:
    error = normalize_provider_error("openai", FakeSDKError(status_code=429))
    assert error.retryable


def test_quota_429_is_permanent_by_structured_code() -> None:
    error = normalize_provider_error(
        "openai",
        FakeSDKError(
            status_code=429,
            body={"error": {"code": "insufficient_quota"}},
        ),
    )
    assert not error.retryable
    assert error.error_code == "insufficient_quota"


def test_authentication_status_is_permanent() -> None:
    error = normalize_provider_error("anthropic", FakeSDKError(status_code=401))
    assert not error.retryable


def test_retry_after_is_preserved() -> None:
    error = normalize_provider_error("anthropic", FakeSDKError(status_code=429, retry_after="37"))
    assert error.retry_after_seconds == 37


def test_child_payload_inherits_correlation_and_parent_job() -> None:
    job_id, correlation_id = uuid4(), uuid4()
    context = JobContext(
        queue="source_fetch",
        msg_id=7,
        read_count=1,
        job_id=job_id,
        correlation_id=correlation_id,
        parent_job_id=None,
        enqueued_at=datetime.now(UTC),
        trigger_type="api",
        deadline_monotonic=time.monotonic() + 60,
    )
    token = set_current_job(context)
    try:
        payload = child_payload({"rawItemIds": ["item-1"]})
    finally:
        reset_current_job(token)

    assert payload["_meta"]["correlationId"] == str(correlation_id)
    assert payload["_meta"]["parentJobId"] == str(job_id)
    assert payload["_meta"]["jobId"] != str(job_id)
