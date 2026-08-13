"""pgmq 消费 worker（ADR-003）。

纯 worker 纪律（SPEC-001 §2）：消费 job → 跑 Agent → 写结果表；
绝不改业务状态机（那是 core 的唯一职责）。

M0 骨架：只有注册表与消费循环；具体 handler 随 M1（sourcing/topic）逐个填充。
"""

from __future__ import annotations

import json
import os
import signal
import types
from collections.abc import Callable
from pathlib import Path
from typing import Any

import structlog
from psycopg import Connection
from psycopg.rows import dict_row

from scholar_agents.agents.judge import run_judge
from scholar_agents.agents.scout import run_scout
from scholar_agents.db_access import AgentRepository
from scholar_agents.observability import ObservedProvider, TraceRecorder
from scholar_agents.providers.router import ModelRouter
from scholar_agents.sourcing.handler import handle_source_fetch

log = structlog.get_logger()

# 同一个 job 最多执行三次；供应商配额/认证类错误不应进入 visibility timeout 重试。
MAX_JOB_ATTEMPTS = 3
_PERMANENT_ERROR_MARKERS = (
    "quota",
    "rate limit",
    "insufficient balance",
    "invalid api key",
    "authentication",
    "unauthorized",
    "is required for provider",
    "model not found",
)


class PermanentJobError(RuntimeError):
    """当前 job 无法通过重试恢复，应直接进入失败/死信处理。"""


def is_permanent_error(exc: BaseException) -> bool:
    """按供应商错误文本识别不可重试的额度、认证和模型配置错误。"""
    message = str(exc).lower()
    return any(marker in message for marker in _PERMANENT_ERROR_MARKERS)


def should_retry(exc: BaseException, read_count: int) -> bool:
    """临时错误最多重试到 MAX_JOB_ATTEMPTS；永久错误永不重试。"""
    return (
        read_count < MAX_JOB_ATTEMPTS
        and not isinstance(exc, PermanentJobError)
        and not is_permanent_error(exc)
    )


# 队列名以 scholar-shared/schemas/queues.json 为准
Handler = Callable[[Connection[Any], dict[str, Any]], None]
HANDLERS: dict[str, Handler] = {}


def handler(queue: str) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        HANDLERS[queue] = fn
        return fn

    return register


@handler("source_fetch")
def handle_source_fetch_job(conn: Connection[Any], payload: dict[str, Any]) -> None:
    handle_source_fetch(conn, payload)


@handler("topic_scout")
def handle_topic_scout(conn: Connection[Any], payload: dict[str, Any]) -> None:
    router = ModelRouter.from_yaml(_routing_config_path())
    provider, model = router.resolve("topic_scout")
    trace = TraceRecorder(name="topic-scout")
    trace.trace(metadata={"jobType": "topic.scout"})
    provider = ObservedProvider(
        provider,
        trace,
        observation_name="topic-scout-structured",
        prompt_version="topic-scout@v1",
    )
    repository = AgentRepository(conn)
    max_topics = payload.get("maxTopics")
    max_items = payload.get("maxItems")
    run_scout(
        repository.list_new_raw_items(limit=int(max_items) if max_items is not None else 100),
        provider,
        model,
        repository,
        max_topics=int(max_topics) if max_topics is not None else None,
        langfuse_trace_id=trace.trace_id,
    )


@handler("topic_evaluate")
def handle_topic_evaluate(conn: Connection[Any], payload: dict[str, Any]) -> None:
    topic_id = payload.get("topicId") or payload.get("topic_id")
    if not topic_id:
        raise ValueError("topic_evaluate payload requires topicId")
    router = ModelRouter.from_yaml(_routing_config_path())
    provider, model = router.resolve("topic_judge")
    trace = TraceRecorder(name="topic-judge")
    trace.trace(metadata={"jobType": "topic.evaluate", "topicId": str(topic_id)})
    observed_provider = ObservedProvider(
        provider,
        trace,
        observation_name="topic-judge-structured",
        prompt_version="topic-judge@v1",
    )
    run_judge(
        topic_id,
        observed_provider,
        model,
        AgentRepository(conn),
        _rubric_path(),
        recorder=trace,
    )


def _routing_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "model_routing.yaml"


def _rubric_path() -> Path:
    return Path(__file__).resolve().parents[4] / "scholar-shared" / "rubrics" / "topic.v1.yaml"


class Worker:
    def __init__(self, conn: Connection[Any], visibility_timeout: int = 300) -> None:
        self._conn = conn
        self._vt = visibility_timeout
        self._running = True

    def stop(self, _sig: int | None = None, _frm: types.FrameType | None = None) -> None:
        self._running = False

    def poll_once(self) -> bool:
        """轮询所有已注册队列各一次；处理了任意消息返回 True。"""
        got = False
        for queue, fn in HANDLERS.items():
            with self._conn.cursor(row_factory=dict_row) as cur:
                cur.execute(
                    "select msg_id, read_ct, message from pgmq.read(%s, %s, 1)",
                    (queue, self._vt),
                )
                row = cur.fetchone()
            if row is None:
                continue
            # pgmq.read changes the message visibility inside the current
            # transaction. Commit that claim before running the handler so a
            # handler failure cannot roll it back and hot-loop the same job.
            self._conn.commit()
            got = True
            msg_id, read_count, payload = row["msg_id"], row["read_ct"], row["message"]
            if isinstance(payload, str):  # psycopg 的 jsonb 通常已解码，防御字符串形态
                payload = json.loads(payload)
            log.info("job_start", queue=queue, msg_id=msg_id)
            try:
                fn(self._conn, payload)
            except Exception as exc:  # noqa: BLE001 — worker 边界必须兜住 job 异常
                retry = should_retry(exc, read_count)
                log.exception(
                    "job_failed",
                    queue=queue,
                    msg_id=msg_id,
                    read_count=read_count,
                    retry=retry,
                )
                self._conn.rollback()
                if not retry:
                    with self._conn.cursor() as cur:
                        cur.execute("select pgmq.delete(%s, %s)", (queue, msg_id))
                    self._conn.commit()
                    log.error("job_discarded", queue=queue, msg_id=msg_id)
            else:
                with self._conn.cursor() as cur:
                    cur.execute("select pgmq.delete(%s, %s)", (queue, msg_id))
                self._conn.commit()
                log.info("job_done", queue=queue, msg_id=msg_id)
        return got


def main() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is required")
    import time

    with Connection.connect(dsn) as conn:
        worker = Worker(conn)
        signal.signal(signal.SIGINT, worker.stop)
        signal.signal(signal.SIGTERM, worker.stop)
        log.info("worker_started", queues=sorted(HANDLERS))
        while worker._running:  # noqa: SLF001
            if not worker.poll_once():
                time.sleep(1.0)
        log.info("worker_stopped")


if __name__ == "__main__":
    main()
