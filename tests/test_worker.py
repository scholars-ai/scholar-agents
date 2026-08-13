from __future__ import annotations

import pytest

from scholar_agents.worker.consumer import (
    HANDLERS,
    MAX_JOB_ATTEMPTS,
    PermanentJobError,
    is_permanent_error,
    should_retry,
)


def test_m1_handlers_are_registered() -> None:
    assert {"source_fetch", "topic_scout", "topic_evaluate"}.issubset(HANDLERS)


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
    ("message", "expected"),
    [
        ("quota exceeded", True),
        ("insufficient balance", True),
        ("invalid api key", True),
        ("model not found", True),
        ("env ANTHROPIC_API_KEY is required for provider 'anthropic'", True),
        ("source Manual Feed has no url", True),
        ("source source-1 not found", True),
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
    monkeypatch.setitem(HANDLERS, queue, lambda _conn, _payload: (_ for _ in ()).throw(
        RuntimeError("temporary")
    ))
    conn = FakeConnection()
    assert Worker(conn, visibility_timeout=30).poll_once()
    assert conn.commits == 1
    assert conn.rollbacks == 1


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
