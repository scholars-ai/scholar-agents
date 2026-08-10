"""数据库连接（psycopg 3）。

纪律（SPEC-001 §2）：agents 只写结果表，**绝不改业务状态机**——状态流转是 core 的
唯一职责。本模块因此只暴露读 + 写结果表的能力。
"""

from __future__ import annotations

import os
from collections.abc import Iterator
from contextlib import contextmanager
from typing import Any

from psycopg import Connection, connect
from psycopg.rows import dict_row


def dsn() -> str:
    url = os.environ.get("DATABASE_URL")
    if not url:
        raise RuntimeError("DATABASE_URL is required")
    return url


@contextmanager
def connection() -> Iterator[Connection[dict[str, Any]]]:
    """一个自动提交/回滚的连接。row_factory 统一 dict_row。"""
    conn = connect(dsn(), row_factory=dict_row)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def to_pgvector(vec: list[float]) -> str:
    """pgvector 的文本输入格式：'[0.1,0.2,...]'。"""
    return "[" + ",".join(f"{x:.7f}" for x in vec) + "]"
