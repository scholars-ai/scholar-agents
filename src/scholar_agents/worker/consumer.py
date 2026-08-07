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
from typing import Any

import structlog
from psycopg import Connection
from psycopg.rows import dict_row

log = structlog.get_logger()

# 队列名以 scholar-shared/schemas/queues.json 为准
Handler = Callable[[dict[str, Any]], None]
HANDLERS: dict[str, Handler] = {}


def handler(queue: str) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        HANDLERS[queue] = fn
        return fn

    return register


@handler("topic_evaluate")
def handle_topic_evaluate(payload: dict[str, Any]) -> None:
    # M1: TopicJudge —— 读取 rubric YAML + 生效权重，complete_structured 出分，写 topic_evaluations
    raise NotImplementedError("M1: topic_evaluate")


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
                    "select msg_id, message from pgmq.read(%s, %s, 1)", (queue, self._vt)
                )
                row = cur.fetchone()
            if row is None:
                continue
            got = True
            msg_id, payload = row["msg_id"], row["message"]
            if isinstance(payload, str):  # psycopg 的 jsonb 通常已解码，防御字符串形态
                payload = json.loads(payload)
            log.info("job_start", queue=queue, msg_id=msg_id)
            try:
                fn(payload)
            except Exception:
                # 不 delete：visibility timeout 到期后 pgmq 自动重投；
                # 重试上限与死信队列在 M1 落地
                log.exception("job_failed", queue=queue, msg_id=msg_id)
                self._conn.rollback()
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
