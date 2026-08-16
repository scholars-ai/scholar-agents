"""跨队列 job 元数据、deadline 和当前上下文。"""

from __future__ import annotations

import signal
import threading
import time
from collections.abc import Iterator
from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any
from uuid import NAMESPACE_URL, UUID, uuid4, uuid5

from opentelemetry import propagate

from scholar_agents.errors import JobDeadlineExceeded


def _uuid(value: object, fallback: UUID) -> UUID:
    try:
        return UUID(str(value))
    except (TypeError, ValueError, AttributeError):
        return fallback


def _timestamp(value: object) -> datetime:
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=UTC)
    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return parsed if parsed.tzinfo else parsed.replace(tzinfo=UTC)
        except ValueError:
            pass
    return datetime.now(UTC)


@dataclass(frozen=True, slots=True)
class JobContext:
    queue: str
    msg_id: int
    read_count: int
    job_id: UUID
    correlation_id: UUID
    parent_job_id: UUID | None
    enqueued_at: datetime
    trigger_type: str
    deadline_monotonic: float

    @classmethod
    def from_message(
        cls,
        queue: str,
        msg_id: int,
        read_count: int,
        payload: dict[str, Any],
        timeout_seconds: float,
    ) -> JobContext:
        raw = payload.get("_meta")
        meta = raw if isinstance(raw, dict) else {}
        stable_legacy_id = uuid5(NAMESPACE_URL, f"pgmq:{queue}:{msg_id}")
        job_id = _uuid(meta.get("jobId"), stable_legacy_id)
        return cls(
            queue=queue,
            msg_id=msg_id,
            read_count=read_count,
            job_id=job_id,
            correlation_id=_uuid(meta.get("correlationId"), job_id),
            parent_job_id=(
                _uuid(meta.get("parentJobId"), job_id) if meta.get("parentJobId") else None
            ),
            enqueued_at=_timestamp(meta.get("enqueuedAt")),
            trigger_type=str(meta.get("triggerType") or "worker"),
            deadline_monotonic=time.monotonic() + timeout_seconds,
        )

    def remaining_seconds(self) -> float:
        return max(0.0, self.deadline_monotonic - time.monotonic())

    def raise_if_expired(self) -> None:
        if self.remaining_seconds() <= 0:
            raise JobDeadlineExceeded(f"job {self.job_id} exceeded its deadline")


_CURRENT: ContextVar[JobContext | None] = ContextVar("scholar_job_context", default=None)


def current_job() -> JobContext | None:
    return _CURRENT.get()


def set_current_job(context: JobContext) -> Token[JobContext | None]:
    return _CURRENT.set(context)


def reset_current_job(token: Token[JobContext | None]) -> None:
    _CURRENT.reset(token)


def child_payload(payload: dict[str, Any], *, trigger_type: str = "worker") -> dict[str, Any]:
    """复制 payload 并注入当前 trace 与父子 job 关系。"""
    current = current_job()
    correlation_id = current.correlation_id if current else uuid4()
    parent_job_id = current.job_id if current else None
    carrier: dict[str, str] = {}
    propagate.inject(carrier)
    meta: dict[str, Any] = {
        "jobId": str(uuid4()),
        "correlationId": str(correlation_id),
        "parentJobId": str(parent_job_id) if parent_job_id else None,
        "traceparent": carrier.get("traceparent"),
        "tracestate": carrier.get("tracestate"),
        "baggage": carrier.get("baggage"),
        "enqueuedAt": datetime.now(UTC).isoformat(),
        "triggerType": trigger_type,
    }
    return {**payload, "_meta": meta}


@contextmanager
def hard_deadline(seconds: float) -> Iterator[None]:
    """在 worker 主线程使用 SIGALRM 实施整任务 wall-clock deadline。"""
    if seconds <= 0:
        raise JobDeadlineExceeded("job deadline already expired")
    if threading.current_thread() is not threading.main_thread():
        yield
        return

    def expired(_signum: int, _frame: object) -> None:
        raise JobDeadlineExceeded(f"job exceeded {seconds:.1f}s wall-clock deadline")

    previous_handler = signal.getsignal(signal.SIGALRM)
    signal.signal(signal.SIGALRM, expired)
    previous_timer = signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, *previous_timer)
        signal.signal(signal.SIGALRM, previous_handler)
