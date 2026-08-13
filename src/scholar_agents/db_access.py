"""Agents 侧数据库访问。

本模块只负责给 Agent 提供上下文和保存结果：

- 读取待聚类素材、选题上下文、当前生效权重；
- 写入 TopicScout/Judge 的结果和 agent_runs 留痕；
- 将已被 Scout 处理的 raw_items 标记为 clustered。

Topic 的业务状态流转仍由 scholar-core 负责，本模块不会推进 topics.status。
事务由调用方持有，便于一个 Agent job 将结果和留痕原子提交。
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Literal
from uuid import UUID

from psycopg import Connection

from scholar_agents.db import to_pgvector

AgentRunStatus = Literal["running", "succeeded", "failed"]


def _uuid(value: object) -> UUID:
    if isinstance(value, UUID):
        return value
    return UUID(str(value))


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    return float(str(value))


def _embedding(value: object) -> list[float] | None:
    if value is None:
        return None
    if isinstance(value, list):
        return [float(item) for item in value]
    values = getattr(value, "to_list", None)
    if callable(values):
        return [float(item) for item in values()]
    if isinstance(value, str):
        return [float(item) for item in value.strip("[]").split(",") if item]
    raise TypeError(f"unsupported embedding value: {type(value).__name__}")


@dataclass(frozen=True, slots=True)
class RawItemRecord:
    id: UUID
    source_id: UUID
    title: str
    url: str | None
    author: str | None
    content: str
    published_at: datetime | None
    source_name: str
    source_weight: float
    embedding: list[float] | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> RawItemRecord:
        return cls(
            id=_uuid(row["id"]),
            source_id=_uuid(row["source_id"]),
            title=str(row["title"]),
            url=row.get("url"),
            author=row.get("author"),
            content=str(row["content"]),
            published_at=row.get("published_at"),
            source_name=str(row.get("source_name") or ""),
            source_weight=float(row.get("source_weight") or 0),
            embedding=_embedding(row.get("embedding")),
        )


@dataclass(frozen=True, slots=True)
class TopicRecord:
    id: UUID
    title: str
    angle: str
    summary: str
    raw_item_ids: list[UUID]
    target_platforms: list[str]
    status: str
    latest_score: float | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> TopicRecord:
        return cls(
            id=_uuid(row["id"]),
            title=str(row["title"]),
            angle=str(row["angle"]),
            summary=str(row["summary"]),
            raw_item_ids=[_uuid(value) for value in row.get("raw_item_ids") or []],
            target_platforms=[str(value) for value in row.get("target_platforms") or []],
            status=str(row["status"]),
            latest_score=_optional_float(row.get("latest_score")),
        )


@dataclass(frozen=True, slots=True)
class WeightSetRecord:
    rubric_id: str
    version: int
    weights: dict[str, float]

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> WeightSetRecord:
        weights = row["weights"]
        if isinstance(weights, str):
            weights = json.loads(weights)
        if not isinstance(weights, dict):
            raise TypeError("weight_sets.weights must be a JSON object")
        return cls(
            rubric_id=str(row["rubric_id"]),
            version=int(row["version"]),
            weights={str(key): float(value) for key, value in weights.items()},
        )


@dataclass(frozen=True, slots=True)
class TopicEvaluationInsert:
    topic_id: UUID
    rubric_version: str
    dimension_scores: dict[str, float]
    total_score: float
    rationale: str
    judge_model: str
    agent_run_id: UUID | None
    weight_version: int | None
    vetoed_dimension: str | None


@dataclass(frozen=True, slots=True)
class AgentRunInsert:
    job_type: str
    entity_type: str | None
    entity_id: UUID | None
    model: str | None
    prompt_version: str | None
    tokens_in: int | None
    tokens_out: int | None
    cost_usd: float | None
    langfuse_trace_id: str | None
    status: AgentRunStatus


class AgentRepository:
    """SQL 访问薄层；不负责 commit/rollback。"""

    def __init__(self, conn: Connection[dict[str, Any]]) -> None:
        self._conn = conn

    def list_new_raw_items(self, limit: int = 100) -> list[RawItemRecord]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                select r.id, r.source_id, r.title, r.url, r.author, r.content,
                       r.published_at, r.embedding, s.name as source_name, s.weight as source_weight
                from raw_items r
                join sources s on s.id = r.source_id
                where r.status = 'new'
                order by r.created_at, r.id
                limit %s
                """,
                (limit,),
            )
            rows = cur.fetchall()
        return [RawItemRecord.from_row(row) for row in rows]

    def get_topic(self, topic_id: UUID) -> TopicRecord | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                select id, title, angle, summary, raw_item_ids, target_platforms,
                       status, latest_score
                from topics
                where id = %s
                """,
                (topic_id,),
            )
            row = cur.fetchone()
        return None if row is None else TopicRecord.from_row(row)

    def list_topic_raw_items(self, topic: TopicRecord) -> list[RawItemRecord]:
        if not topic.raw_item_ids:
            return []
        with self._conn.cursor() as cur:
            cur.execute(
                """
                select r.id, r.source_id, r.title, r.url, r.author, r.content,
                       r.published_at, r.embedding, s.name as source_name, s.weight as source_weight
                from raw_items r
                join sources s on s.id = r.source_id
                where r.id = any(%s)
                order by r.created_at, r.id
                """,
                (topic.raw_item_ids,),
            )
            rows = cur.fetchall()
        return [RawItemRecord.from_row(row) for row in rows]

    def get_active_weight_set(self, rubric_id: str) -> WeightSetRecord | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                select rubric_id, version, weights
                from weight_sets
                where rubric_id = %s
                order by version desc
                limit 1
                """,
                (rubric_id,),
            )
            row = cur.fetchone()
        return None if row is None else WeightSetRecord.from_row(row)

    def find_similar_topic(
        self,
        embedding: list[float],
        threshold: float = 0.92,
    ) -> tuple[UUID, float] | None:
        vector = to_pgvector(embedding)
        with self._conn.cursor() as cur:
            cur.execute(
                """
                select id, 1 - (topics.embedding <=> %s::vector) as similarity
                from topics
                where embedding is not null
                  and status <> 'rejected'
                order by embedding <=> %s::vector
                limit 1
                """,
                (vector, vector),
            )
            row = cur.fetchone()
        if row is None or row["similarity"] is None:
            return None
        similarity = float(row["similarity"])
        return (_uuid(row["id"]), similarity) if similarity >= threshold else None

    def create_topic(
        self,
        *,
        title: str,
        angle: str,
        summary: str,
        raw_item_ids: list[UUID],
        target_platforms: list[str],
        embedding: list[float],
    ) -> UUID:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                insert into topics
                    (title, angle, summary, raw_item_ids, target_platforms, status, embedding)
                values (%s, %s, %s, %s, %s::platform[], 'candidate', %s::vector)
                returning id
                """,
                (
                    title,
                    angle,
                    summary,
                    raw_item_ids,
                    target_platforms,
                    to_pgvector(embedding),
                ),
            )
            row = cur.fetchone()
        if row is None:
            raise RuntimeError("create topic returned no id")
        return _uuid(row["id"])

    def mark_raw_items_clustered(self, raw_item_ids: list[UUID]) -> None:
        if not raw_item_ids:
            return
        with self._conn.cursor() as cur:
            cur.execute(
                """
                update raw_items
                set status = 'clustered', updated_at = now()
                where id = any(%s) and status = 'new'
                """,
                (raw_item_ids,),
            )

    def create_agent_run(self, run: AgentRunInsert) -> UUID:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                insert into agent_runs
                    (job_type, entity_type, entity_id, langfuse_trace_id, model,
                     prompt_version, tokens_in, tokens_out, cost_usd, status)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::agent_run_status)
                returning id
                """,
                (
                    run.job_type,
                    run.entity_type,
                    run.entity_id,
                    run.langfuse_trace_id,
                    run.model,
                    run.prompt_version,
                    run.tokens_in,
                    run.tokens_out,
                    run.cost_usd,
                    run.status,
                ),
            )
            row = cur.fetchone()
        if row is None:
            raise RuntimeError("create agent run returned no id")
        return _uuid(row["id"])

    def update_agent_run(self, run_id: UUID, run: AgentRunInsert) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                update agent_runs
                set langfuse_trace_id = %s, model = %s, prompt_version = %s,
                    tokens_in = %s, tokens_out = %s, cost_usd = %s,
                    status = %s::agent_run_status, updated_at = now()
                where id = %s
                """,
                (
                    run.langfuse_trace_id,
                    run.model,
                    run.prompt_version,
                    run.tokens_in,
                    run.tokens_out,
                    run.cost_usd,
                    run.status,
                    run_id,
                ),
            )

    def create_topic_evaluation(self, evaluation: TopicEvaluationInsert) -> UUID:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                insert into topic_evaluations
                    (topic_id, rubric_version, dimension_scores, total_score, rationale,
                     judge_model, agent_run_id, weight_version, vetoed_dimension)
                values (%s, %s, %s::jsonb, %s, %s, %s, %s, %s, %s)
                returning id
                """,
                (
                    evaluation.topic_id,
                    evaluation.rubric_version,
                    json.dumps(evaluation.dimension_scores, ensure_ascii=False),
                    evaluation.total_score,
                    evaluation.rationale,
                    evaluation.judge_model,
                    evaluation.agent_run_id,
                    evaluation.weight_version,
                    evaluation.vetoed_dimension,
                ),
            )
            row = cur.fetchone()
        if row is None:
            raise RuntimeError("create topic evaluation returned no id")
        return _uuid(row["id"])
