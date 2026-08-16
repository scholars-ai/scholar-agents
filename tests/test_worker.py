from __future__ import annotations

import pytest

from scholar_agents.errors import ProviderError
from scholar_agents.worker.consumer import (
    HANDLERS,
    MAX_JOB_ATTEMPTS,
    PermanentJobError,
    is_permanent_error,
    should_retry,
)


def test_m1_and_m2_handlers_are_registered() -> None:
    assert {
        "source_fetch",
        "topic_scout",
        "topic_evaluate",
        "article_write",
        "memory_reflect",
        "article_evaluate",
    }.issubset(HANDLERS)


def test_manual_source_fetch_builds_targeted_scout_payload() -> None:
    from scholar_agents.worker.consumer import _manual_scout_payload

    payload = _manual_scout_payload(
        {"sourceId": "source-1", "url": "https://example.com/article"},
        ["item-1"],
    )

    assert payload == {"rawItemIds": ["item-1"]}


def test_non_manual_source_fetch_does_not_build_targeted_scout_payload() -> None:
    from scholar_agents.worker.consumer import _manual_scout_payload

    assert _manual_scout_payload({"sourceId": "source-1"}, ["item-1"]) is None


def test_scheduled_scout_has_bounded_default_item_batch() -> None:
    from scholar_agents.worker.consumer import _scout_item_limit

    assert _scout_item_limit({}, []) == 20
    assert _scout_item_limit({"maxItems": 7}, []) == 7


def test_targeted_scout_keeps_all_requested_items() -> None:
    from uuid import uuid4

    from scholar_agents.worker.consumer import _scout_item_limit

    item_ids = [uuid4(), uuid4(), uuid4()]

    assert _scout_item_limit({}, item_ids) == len(item_ids)


@pytest.mark.parametrize(
    ("error", "expected"),
    [
        (
            ProviderError(
                "quota exceeded",
                provider="openai",
                status_code=429,
                error_code="insufficient_quota",
                retryable=False,
            ),
            True,
        ),
        (
            ProviderError(
                "temporarily rate limited",
                provider="anthropic",
                status_code=429,
                error_code="rate_limit_error",
                retryable=True,
            ),
            False,
        ),
        (RuntimeError("connection reset by peer"), False),
    ],
)
def test_classifies_typed_provider_errors_for_retry(error: BaseException, expected: bool) -> None:
    assert is_permanent_error(error) is expected


def test_permanent_job_error_is_always_non_retryable() -> None:
    assert not should_retry(PermanentJobError("quota exceeded"), read_count=1)


def test_transient_job_retries_until_three_total_attempts() -> None:
    assert MAX_JOB_ATTEMPTS == 3
    assert should_retry(RuntimeError("temporary"), read_count=1)
    assert should_retry(RuntimeError("temporary"), read_count=2)
    assert not should_retry(RuntimeError("temporary"), read_count=3)


def test_worker_keeps_failed_message_claim_until_visibility_timeout(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """handler 失败后不能因事务回滚立即再次读取同一条消息。"""
    from scholar_agents.worker.consumer import Worker

    class FakeCursor:
        def __init__(self) -> None:
            self.executed: list[tuple[str, tuple[object, ...]]] = []
            self.rows = iter([{"msg_id": 7, "read_ct": 1, "message": {"x": 1}}])

        def __enter__(self) -> FakeCursor:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, query: str, params: tuple[object, ...] = ()) -> None:
            self.executed.append((query, params))

        def fetchone(self) -> dict[str, object] | None:
            return next(self.rows, None)

    class FakeConnection:
        def __init__(self) -> None:
            self.cursor_instance = FakeCursor()
            self.commits = 0
            self.rollbacks = 0

        def cursor(self, **_kwargs: object) -> FakeCursor:
            return self.cursor_instance

        def commit(self) -> None:
            self.commits += 1

        def rollback(self) -> None:
            self.rollbacks += 1

    queue = "source_fetch"
    monkeypatch.setitem(
        HANDLERS, queue, lambda _conn, _payload: (_ for _ in ()).throw(RuntimeError("temporary"))
    )
    conn = FakeConnection()
    assert Worker(conn, visibility_timeout=30).poll_once()
    # claim、初始 lease、失败退避各自提交，消息不会因 rollback 立即重现。
    assert conn.commits == 3
    assert conn.rollbacks == 1
    set_vt_calls = [call for call in conn.cursor_instance.executed if "pgmq.set_vt" in call[0]]
    assert [call[1][2] for call in set_vt_calls] == [270, 15]


def test_worker_connection_uses_dict_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from scholar_agents.worker import consumer

    calls: list[dict[str, object]] = []

    class FakeConnectionContext:
        def __enter__(self) -> object:
            return object()

        def __exit__(self, *_args: object) -> None:
            return None

    def fake_connect(dsn: str, **kwargs: object) -> FakeConnectionContext:
        calls.append({"dsn": dsn, **kwargs})
        return FakeConnectionContext()

    monkeypatch.setattr(consumer.Connection, "connect", staticmethod(fake_connect))
    assert consumer._connect_worker_database("postgres://test") is not None
    assert calls == [{"dsn": "postgres://test", "row_factory": consumer.dict_row}]
