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
from scholar_agents.job_context import current_job

AgentRunStatus = Literal["running", "succeeded", "failed"]


def _current_correlation_id() -> UUID | None:
    job = current_job()
    return job.correlation_id if job else None


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


def _string_list(value: object) -> list[str]:
    """兼容 psycopg 对已知数组和未注册自定义 enum[] 的两种返回形态。"""
    if value is None:
        return []
    if isinstance(value, str):
        raw = value.strip()
        if raw.startswith("{") and raw.endswith("}"):
            raw = raw[1:-1]
        if not raw:
            return []
        return [item.strip().strip('"') for item in raw.split(",") if item.strip()]
    if isinstance(value, (list, tuple)):
        return [str(item) for item in value]
    raise TypeError(f"unsupported string array value: {type(value).__name__}")


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
    correlation_id: UUID | None = None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> TopicRecord:
        return cls(
            id=_uuid(row["id"]),
            title=str(row["title"]),
            angle=str(row["angle"]),
            summary=str(row["summary"]),
            raw_item_ids=[_uuid(value) for value in row.get("raw_item_ids") or []],
            target_platforms=_string_list(row.get("target_platforms")),
            status=str(row["status"]),
            latest_score=_optional_float(row.get("latest_score")),
            correlation_id=(
                _uuid(row["correlation_id"]) if row.get("correlation_id") is not None else None
            ),
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
    dimension_reasons: dict[str, str] | None = None


@dataclass(frozen=True, slots=True)
class ArticleEvaluationInsert:
    article_id: UUID
    rubric_version: str
    dimension_scores: dict[str, float]
    dimension_reasons: dict[str, str]
    total_score: float
    rationale: str
    judge_model: str
    agent_run_id: UUID | None
    weight_version: int
    vetoed_dimension: str | None
    pass_threshold: float
    passed: bool


@dataclass(frozen=True, slots=True)
class InsightRecord:
    content: str
    evidence: dict[str, Any]
    confidence: float


@dataclass(frozen=True, slots=True)
class ArticleReferenceRecord:
    title: str
    content_md: str
    latest_score: float


@dataclass(frozen=True, slots=True)
class ArticleRecord:
    id: UUID
    topic_id: UUID
    platform: str
    version: int
    title: str
    content_md: str
    writer_agent: str
    status: str
    latest_score: float | None
    previous_article_id: UUID | None

    @classmethod
    def from_row(cls, row: dict[str, Any]) -> ArticleRecord:
        return cls(
            id=_uuid(row["id"]),
            topic_id=_uuid(row["topic_id"]),
            platform=str(row["platform"]),
            version=int(row["version"]),
            title=str(row["title"]),
            content_md=str(row["content_md"]),
            writer_agent=str(row["writer_agent"]),
            status=str(row["status"]),
            latest_score=_optional_float(row.get("latest_score")),
            previous_article_id=(
                _uuid(row["previous_article_id"])
                if row.get("previous_article_id") is not None
                else None
            ),
        )


@dataclass(frozen=True, slots=True)
class ArticleInsert:
    topic_id: UUID
    platform: str
    version: int
    title: str
    content_md: str
    writer_agent: str
    previous_article_id: UUID | None = None


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
    correlation_id: UUID | None = None


class AgentRepository:
    """SQL 访问薄层；不负责 commit/rollback。"""

    def __init__(self, conn: Connection[dict[str, Any]]) -> None:
        self._conn = conn

    def list_new_raw_items(
        self, limit: int = 100, raw_item_ids: list[UUID] | None = None
    ) -> list[RawItemRecord]:
        id_filter = ""
        params: list[Any] = []
        if raw_item_ids:
            id_filter = "and r.id = any(%s)"
            params.append(raw_item_ids)
        params.append(limit)
        with self._conn.cursor() as cur:
            cur.execute(
                f"""
                select r.id, r.source_id, r.title, r.url, r.author, r.content,
                       r.published_at, r.embedding, s.name as source_name, s.weight as source_weight
                from raw_items r
                join sources s on s.id = r.source_id
                where r.status = 'new' {id_filter}
                order by r.created_at, r.id
                limit %s
                """,
                tuple(params),
            )
            rows = cur.fetchall()
        return [RawItemRecord.from_row(row) for row in rows]

    def get_topic(self, topic_id: UUID) -> TopicRecord | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                select id, title, angle, summary, raw_item_ids, target_platforms,
                       status, latest_score, correlation_id
                from topics
                where id = %s
                """,
                (topic_id,),
            )
            row = cur.fetchone()
        return None if row is None else TopicRecord.from_row(row)

    def get_article(self, article_id: UUID) -> ArticleRecord | None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                select id, topic_id, platform, version, title, content_md,
                       writer_agent, status, latest_score, previous_article_id
                from articles
                where id = %s
                """,
                (article_id,),
            )
            row = cur.fetchone()
        return None if row is None else ArticleRecord.from_row(row)

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

    def list_writing_insights(self, platform: str, limit: int = 5) -> list[InsightRecord]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                select content, evidence, confidence
                from insights
                where status = 'active'
                  and kind in ('writing_lesson', 'platform_lesson')
                  and (platform is null or platform = %s::platform)
                order by confidence desc, updated_at desc
                limit %s
                """,
                (platform, limit),
            )
            rows = cur.fetchall()
        records: list[InsightRecord] = []
        for row in rows:
            evidence = row.get("evidence") or {}
            if isinstance(evidence, str):
                evidence = json.loads(evidence)
            records.append(
                InsightRecord(
                    content=str(row["content"]),
                    evidence=evidence if isinstance(evidence, dict) else {},
                    confidence=float(row["confidence"]),
                )
            )
        return records

    def list_high_score_articles(
        self, platform: str, limit: int = 3
    ) -> list[ArticleReferenceRecord]:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                select title, content_md, latest_score
                from articles
                where platform = %s::platform and latest_score is not null
                  and status in ('scored', 'pending_review', 'approved', 'published')
                order by latest_score desc, updated_at desc
                limit %s
                """,
                (platform, limit),
            )
            rows = cur.fetchall()
        return [
            ArticleReferenceRecord(
                title=str(row["title"]),
                content_md=str(row["content_md"]),
                latest_score=float(row["latest_score"]),
            )
            for row in rows
        ]

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
                    (title, angle, summary, raw_item_ids, target_platforms, status,
                     embedding, correlation_id)
                values (%s, %s, %s, %s, %s::platform[], 'candidate', %s::vector, %s)
                returning id
                """,
                (
                    title,
                    angle,
                    summary,
                    raw_item_ids,
                    target_platforms,
                    to_pgvector(embedding),
                    _current_correlation_id(),
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
                     prompt_version, tokens_in, tokens_out, cost_usd, status, correlation_id)
                values (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s::agent_run_status, %s)
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
                    run.correlation_id or _current_correlation_id(),
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
                    status = %s::agent_run_status, correlation_id = coalesce(%s, correlation_id),
                    updated_at = now()
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
                    run.correlation_id or _current_correlation_id(),
                    run_id,
                ),
            )

    def create_topic_evaluation(self, evaluation: TopicEvaluationInsert) -> UUID:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                insert into topic_evaluations
                    (topic_id, rubric_version, dimension_scores, dimension_reasons,
                     total_score, rationale,
                     judge_model, agent_run_id, weight_version, vetoed_dimension)
                values (%s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s)
                returning id
                """,
                (
                    evaluation.topic_id,
                    evaluation.rubric_version,
                    json.dumps(evaluation.dimension_scores, ensure_ascii=False),
                    json.dumps(evaluation.dimension_reasons or {}, ensure_ascii=False),
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

    def create_article(self, article: ArticleInsert) -> UUID:
        """写入 Agent 结果；唯一键保证同一 topic/platform/version 幂等。"""
        with self._conn.cursor() as cur:
            cur.execute(
                """
                insert into articles
                    (topic_id, platform, version, format, title, content_md,
                     assets, writer_agent, status, previous_article_id)
                values (%s, %s::platform, %s, 'markdown', %s, %s, '[]'::jsonb, %s, 'draft', %s)
                on conflict (topic_id, platform, version) do nothing
                returning id
                """,
                (
                    article.topic_id,
                    article.platform,
                    article.version,
                    article.title,
                    article.content_md,
                    article.writer_agent,
                    article.previous_article_id,
                ),
            )
            row = cur.fetchone()
            if row is None:
                cur.execute(
                    """
                    select id from articles
                    where topic_id = %s and platform = %s::platform and version = %s
                    """,
                    (article.topic_id, article.platform, article.version),
                )
                row = cur.fetchone()
        if row is None:
            raise RuntimeError("create article returned no id")
        return _uuid(row["id"])

    def create_article_evaluation(self, evaluation: ArticleEvaluationInsert) -> UUID:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                insert into article_evaluations
                    (article_id, rubric_version, dimension_scores, dimension_reasons,
                     total_score, rationale, judge_model, agent_run_id, weight_version,
                     vetoed_dimension, pass_threshold, passed)
                values (%s, %s, %s::jsonb, %s::jsonb, %s, %s, %s, %s, %s, %s, %s, %s)
                returning id
                """,
                (
                    evaluation.article_id,
                    evaluation.rubric_version,
                    json.dumps(evaluation.dimension_scores, ensure_ascii=False),
                    json.dumps(evaluation.dimension_reasons, ensure_ascii=False),
                    evaluation.total_score,
                    evaluation.rationale,
                    evaluation.judge_model,
                    evaluation.agent_run_id,
                    evaluation.weight_version,
                    evaluation.vetoed_dimension,
                    evaluation.pass_threshold,
                    evaluation.passed,
                ),
            )
            row = cur.fetchone()
        if row is None:
            raise RuntimeError("create article evaluation returned no id")
        return _uuid(row["id"])
