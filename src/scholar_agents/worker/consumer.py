"""pgmq Worker：消费、deadline、重试、死信、幂等与 Trace Context。"""

from __future__ import annotations

import json
import math
import os
import signal
import time
import types
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import UUID

import structlog
from psycopg import Connection
from psycopg.rows import dict_row
from pydantic import ValidationError
from scholar_contracts.models import ArticleEvaluateJob, ArticleWriteJob, MemoryReflectJob

from scholar_agents import telemetry
from scholar_agents.agents.article_judge import run_article_judge
from scholar_agents.agents.judge import run_judge
from scholar_agents.agents.reflector import run_reflector
from scholar_agents.agents.scout import run_scout
from scholar_agents.agents.writer import WriterModels, WriterProviders, run_writer
from scholar_agents.db_access import AgentRepository
from scholar_agents.errors import JobError, PermanentJobError, ProviderError
from scholar_agents.job_context import (
    JobContext,
    child_payload,
    hard_deadline,
    reset_current_job,
    set_current_job,
)
from scholar_agents.observability import ObservedProvider, TraceRecorder
from scholar_agents.providers.router import ModelRouter
from scholar_agents.sourcing.handler import SourcingStats, handle_source_fetch
from scholar_agents.writing.profiles import load_platform_profile

log = structlog.get_logger()

MAX_JOB_ATTEMPTS = 3
DEFAULT_SCOUT_MAX_ITEMS = 20
DEFAULT_SCOUT_MAX_CONCURRENCY = 6
DEFAULT_JOB_TIMEOUT_SECONDS = 240.0
DEFAULT_VISIBILITY_GRACE_SECONDS = 30


def is_permanent_error(exc: BaseException) -> bool:
    """只信任显式错误语义，不再解析供应商自然语言错误消息。"""
    return isinstance(exc, JobError) and not exc.retryable


def should_retry(exc: BaseException, read_count: int) -> bool:
    retryable = exc.retryable if isinstance(exc, JobError) else True
    return read_count < MAX_JOB_ATTEMPTS and retryable


def retry_delay_seconds(exc: BaseException, read_count: int) -> int:
    retry_after = exc.retry_after_seconds if isinstance(exc, JobError) else None
    if retry_after is not None:
        return max(1, min(900, math.ceil(float(retry_after))))
    return int(min(300, 15 * (2 ** max(0, read_count - 1))))


def workflow_node_key(queue: str) -> str:
    return queue


def _workflow_event(
    conn: Connection[Any],
    context: JobContext,
    *,
    event_type: str,
    status: str,
    message: str,
    payload: dict[str, Any] | None = None,
) -> None:
    """老消息没有 workflow_runs 时静默跳过，保持 Worker 向后兼容。"""
    with conn.cursor() as cur:
        cur.execute(
            """
            insert into workflow_node_runs (run_id, node_key, status, config_snapshot)
            select %s, %s, 'queued', %s::jsonb
            where exists (select 1 from workflow_runs where id = %s)
            on conflict (run_id, node_key) do nothing
            """,
            (
                context.correlation_id,
                workflow_node_key(context.queue),
                json.dumps({"queue": context.queue}, ensure_ascii=False),
                context.correlation_id,
            ),
        )
        cur.execute(
            """
            insert into workflow_events
                (run_id, node_key, event_type, status, message, payload)
            select %s, %s, %s, %s, %s, %s::jsonb
            where exists (select 1 from workflow_runs where id = %s)
            """,
            (
                context.correlation_id,
                workflow_node_key(context.queue),
                event_type,
                status,
                message,
                json.dumps(payload or {}, ensure_ascii=False, default=str),
                context.correlation_id,
            ),
        )
        # Core's workflow runtime is the sole authority for node/run terminal
        # state. A worker event describes one item attempt only; a barrier must
        # observe all expected items before it can close a node or run.


def _record_workflow_failure_decision(
    conn: Connection[Any], context: JobContext, payload: dict[str, Any], error: BaseException
) -> None:
    """Persist a terminal technical failure as an item decision when possible."""
    if context.correlation_id is None:
        return
    identity = next(
        ((key, payload[key]) for key in ("sourceId", "topicId", "articleId") if payload.get(key)),
        None,
    )
    if identity is None:
        return
    key, value = identity
    item_type = {"sourceId": "raw_item", "topicId": "topic", "articleId": "article"}[key]
    try:
        item_id = UUID(str(value))
    except (TypeError, ValueError):
        return
    with conn.cursor() as cur:
        cur.execute(
            "select id from workflow_node_runs where run_id = %s and node_key = %s",
            (context.correlation_id, context.queue),
        )
        row = cur.fetchone()
        if row is None:
            return
        node_run_id = row[0] if not isinstance(row, dict) else row["id"]
        cur.execute(
            """
            insert into workflow_item_decisions
              (run_id, node_run_id, item_id, item_type, decision, reason_code,
               reason, input_refs, evidence_refs)
            values (%s, %s, %s, %s, 'failed', 'technical_failure', %s,
                    %s::jsonb, %s::jsonb)
            on conflict (run_id, node_run_id, item_id) do nothing
            """,
            (
                context.correlation_id,
                node_run_id,
                item_id,
                item_type,
                str(error)[:2000],
                json.dumps({key: str(item_id)}),
                json.dumps({"queue": context.queue}),
            ),
        )


def _record_workflow_decision(
    conn: Connection[Any], context: JobContext, payload: dict[str, Any], result: object
) -> None:
    if context.correlation_id is None or context.queue not in {
        "topic_evaluate",
        "article_evaluate",
    }:
        return
    item_key = "topicId" if context.queue == "topic_evaluate" else "articleId"
    item_type = "topic" if context.queue == "topic_evaluate" else "article"
    try:
        item_id = UUID(
            str(payload.get(item_key) or payload.get(item_key[0].lower() + item_key[1:]))
        )
    except (TypeError, ValueError):
        return
    passed = True
    total_score: float | None = None
    reason_code = "passed"
    reason = "评估通过"
    agent_run_id = None
    threshold: float | None = None
    if context.queue == "article_evaluate":
        score = getattr(result, "score", None)
        passed = bool(getattr(score, "passed", False))
        total_score = getattr(score, "total_score", None)
        threshold = getattr(score, "pass_threshold", None)
        agent_run_id = getattr(result, "agent_run_id", None)
        if not passed:
            reason_code = "article_quality_rejected"
            reason = getattr(score, "vetoed_dimension", None) or "未达到文章评估通过条件"
    else:
        passed = bool(getattr(result, "passed", False))
        total_score = getattr(result, "total_score", None)
        threshold = getattr(result, "pass_threshold", None)
        agent_run_id = getattr(result, "agent_run_id", None)
        if not passed:
            veto = getattr(result, "vetoed_dimension", None)
            reason_code = "topic_vetoed" if veto else "topic_score_below_threshold"
            reason = f"触发否决维度: {veto}" if veto else "总分未达到选题评估通过阈值"
    evaluation_id = getattr(result, "evaluation_id", None)
    dimension_scores: dict[str, Any] = {}
    rubric_version: str | None = None
    weight_version: int | None = None
    trace_id: str | None = None
    with conn.cursor() as cur:
        cur.execute(
            "select id from workflow_node_runs where run_id = %s and node_key = %s",
            (context.correlation_id, context.queue),
        )
        row = cur.fetchone()
        if row is None:
            return
        node_run_id = row[0] if not isinstance(row, dict) else row["id"]
        if evaluation_id is not None:
            table = "article_evaluations" if item_type == "article" else "topic_evaluations"
            threshold_column = (
                "pass_threshold"
                if table == "article_evaluations"
                else "null::numeric as pass_threshold"
            )
            with conn.cursor(row_factory=dict_row) as details:
                details.execute(
                    f"select rubric_version, dimension_scores, weight_version, "
                    f"{threshold_column}, "
                    "agent_run_id from " + table + " where id = %s",
                    (evaluation_id,),
                )
                evaluation = details.fetchone()
            if evaluation:
                raw_dimensions = evaluation.get("dimension_scores")
                if isinstance(raw_dimensions, str):
                    raw_dimensions = json.loads(raw_dimensions)
                if isinstance(raw_dimensions, dict):
                    dimension_scores = raw_dimensions
                rubric_version = evaluation.get("rubric_version")
                weight_version = evaluation.get("weight_version")
                if evaluation.get("pass_threshold") is not None:
                    threshold = float(evaluation["pass_threshold"])
                agent_run_id = evaluation.get("agent_run_id") or agent_run_id
            if agent_run_id:
                with conn.cursor(row_factory=dict_row) as agent_cursor:
                    agent_cursor.execute(
                        "select langfuse_trace_id from agent_runs where id = %s",
                        (agent_run_id,),
                    )
                    agent = agent_cursor.fetchone()
                    if agent:
                        trace_id = agent.get("langfuse_trace_id")
        cur.execute(
            """
            insert into workflow_item_decisions
              (run_id, node_run_id, item_id, item_type, decision, reason_code,
               reason, dimension_scores, total_score, threshold, weight_version,
               rubric_version, input_refs, evidence_refs, agent_run_id, trace_id)
            values (%s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s,
                    %s::jsonb, %s::jsonb, %s, %s)
            on conflict (run_id, node_run_id, item_id) do nothing
            """,
            (
                context.correlation_id,
                node_run_id,
                item_id,
                item_type,
                "accepted" if passed else "rejected",
                reason_code,
                reason,
                json.dumps(dimension_scores, ensure_ascii=False),
                total_score,
                threshold,
                weight_version,
                rubric_version,
                json.dumps({item_key: str(item_id)}),
                json.dumps({"queue": context.queue}),
                agent_run_id,
                trace_id,
            ),
        )


def _workflow_result_payload(queue: str, result: object, payload: dict[str, Any]) -> dict[str, Any]:
    output: dict[str, Any] = {
        key: payload[key]
        for key in ("sourceId", "topicId", "articleId", "platform")
        if key in payload
    }
    if isinstance(result, SourcingStats):
        output.update(result.as_dict())
        output["itemIds"] = [str(item_id) for item_id in result.item_ids]
    else:
        for key in ("created_topics", "clusters_processed", "failed_clusters"):
            value = getattr(result, key, None)
            if value is not None:
                output[key] = value
    return output


def _workflow_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    value = payload.get("workflowConfigOverrides")
    return value if isinstance(value, dict) else {}


def _override_value(payload: dict[str, Any], *keys: str) -> Any:
    overrides = _workflow_overrides(payload)
    for key in keys:
        if key in overrides:
            return overrides[key]
    return None


def _override_model(payload: dict[str, Any], current: str, *keys: str) -> str:
    value = _override_value(payload, *keys)
    return value.strip() if isinstance(value, str) and value.strip() else current


def _override_number(payload: dict[str, Any], *keys: str) -> float | None:
    value = _override_value(payload, *keys)
    if value is None:
        return None
    if isinstance(value, bool):
        raise PermanentJobError(f"workflow override {keys[0]!r} must be numeric")
    try:
        return float(value)
    except (TypeError, ValueError):
        raise PermanentJobError(f"workflow override {keys[0]!r} must be numeric") from None


def _override_int(payload: dict[str, Any], *keys: str) -> int | None:
    value = _override_number(payload, *keys)
    if value is None:
        return None
    if value != int(value):
        raise PermanentJobError(f"workflow override {keys[0]!r} must be an integer")
    return int(value)


def _version_override(payload: dict[str, Any], *keys: str) -> str | None:
    value = _override_value(payload, *keys)
    if value is None:
        return None
    if not isinstance(value, str) or not value.strip():
        raise PermanentJobError(f"workflow override {keys[0]!r} must be a non-empty string")
    return value.strip()


Handler = Callable[[Connection[Any], dict[str, Any]], Any]
HANDLERS: dict[str, Handler] = {}


def handler(queue: str) -> Callable[[Handler], Handler]:
    def register(fn: Handler) -> Handler:
        HANDLERS[queue] = fn
        return fn

    return register


@handler("source_fetch")
def handle_source_fetch_job(conn: Connection[Any], payload: dict[str, Any]) -> SourcingStats:
    with telemetry.span("source_fetch.process"):
        stats = handle_source_fetch(conn, payload)
        scout_payload = _manual_scout_payload(payload, [str(item_id) for item_id in stats.item_ids])
        if scout_payload is None:
            return stats
        downstream = child_payload(scout_payload)
        with (
            telemetry.span("messaging.publish", **{"messaging.destination.name": "topic_scout"}),
            conn.cursor() as cur,
        ):
            cur.execute(
                "select pgmq.send(%s, %s::jsonb)",
                ("topic_scout", json.dumps(downstream)),
            )
        return stats


def _manual_scout_payload(payload: dict[str, Any], item_ids: list[str]) -> dict[str, Any] | None:
    # SPEC-010 runs scout once after all source_fetch fan-out jobs finish.
    if payload.get("workflowRunId"):
        return None
    if (not payload.get("url") and not payload.get("cascade")) or not item_ids:
        return None
    scout_payload: dict[str, Any] = {"rawItemIds": item_ids}
    if payload.get("cascade"):
        scout_payload["cascade"] = True
    return scout_payload


def _scout_item_limit(payload: dict[str, Any], raw_item_ids: list[UUID]) -> int:
    if raw_item_ids:
        return len(raw_item_ids)
    value = payload.get("maxItems", DEFAULT_SCOUT_MAX_ITEMS)
    return int(value)


def _scout_max_concurrency(payload: dict[str, Any]) -> int:
    value = _override_int(payload, "maxConcurrency", "scoutMaxConcurrency")
    if value is None:
        raw = os.environ.get("SCOUT_MAX_CONCURRENCY", str(DEFAULT_SCOUT_MAX_CONCURRENCY))
        try:
            value = int(raw)
        except ValueError:
            value = DEFAULT_SCOUT_MAX_CONCURRENCY
    if value < 1:
        raise PermanentJobError("topic_scout max concurrency must be at least 1")
    return value


def _maybe_enqueue_workflow_scout(conn: Connection[Any], context: JobContext) -> None:
    """Release the topic_scout barrier only after every source in the run succeeded."""
    with conn.cursor(row_factory=dict_row) as cur:
        cur.execute(
            "select metadata from workflow_runs where id = %s for update",
            (context.correlation_id,),
        )
        run = cur.fetchone()
        if run is None:
            return
        expected = int((run["metadata"] or {}).get("sourceCount", 0))
        cur.execute(
            """
            select count(*) as count from workflow_events
            where run_id = %s and node_key = 'source_fetch' and event_type = 'succeeded'
            """,
            (context.correlation_id,),
        )
        count_row = cur.fetchone()
        completed = int(count_row["count"]) if count_row is not None else 0
        if completed < expected:
            cur.execute(
                """
                update workflow_node_runs set status = 'running', completed_at = null
                where run_id = %s and node_key = 'source_fetch'
                """,
                (context.correlation_id,),
            )
            return
        cur.execute(
            """
            select 1 from workflow_events
            where run_id = %s and node_key = 'topic_scout' and event_type = 'queued'
            limit 1
            """,
            (context.correlation_id,),
        )
        if cur.fetchone() is not None:
            return
        cur.execute(
            "select id::text as id from raw_items "
            "where correlation_id = %s order by created_at asc",
            (context.correlation_id,),
        )
        raw_item_ids = [row["id"] for row in cur.fetchall()]
        scout_payload = {
            "rawItemIds": raw_item_ids,
            "cascade": True,
            "workflowRunId": str(context.correlation_id),
        }
        cur.execute("select pgmq.send(%s, %s::jsonb)", ("topic_scout", json.dumps(scout_payload)))
        cur.execute(
            """
            insert into workflow_events
              (run_id, node_key, event_type, status, message, payload)
            values (%s, 'topic_scout', 'queued', 'queued',
                    '采集阶段完成，选题阶段已入队', %s::jsonb)
            """,
            (context.correlation_id, json.dumps({"rawItemCount": len(raw_item_ids)})),
        )


@handler("topic_scout")
def handle_topic_scout(conn: Connection[Any], payload: dict[str, Any]) -> None:
    with telemetry.span("topic_scout.process"):
        router = ModelRouter.from_yaml(_routing_config_path())
        provider, model = router.resolve("topic_scout")
        model_override = _override_value(payload, "model", "topicScoutModel")
        if isinstance(model_override, str) and model_override.strip():
            model = model_override.strip()
        trace = TraceRecorder(name="topic-scout")
        trace.trace(
            metadata={
                "jobType": "topic.scout",
                "agentVersion": _version_override(payload, "agentVersion"),
            }
        )
        provider = ObservedProvider(
            provider,
            trace,
            observation_name="topic-scout-structured",
            prompt_version="topic-scout@v1",
        )
        repository = AgentRepository(conn)
        max_topics = payload.get("maxTopics")
        raw_item_ids = [UUID(str(value)) for value in payload.get("rawItemIds", [])]
        with telemetry.span("raw_items.load"):
            items = repository.list_new_raw_items(
                limit=_scout_item_limit(payload, raw_item_ids),
                raw_item_ids=raw_item_ids or None,
            )
        run_scout(
            items,
            provider,
            model,
            repository,
            max_topics=int(max_topics) if max_topics is not None else None,
            langfuse_trace_id=trace.trace_id,
            targeted=bool(raw_item_ids),
            agent_version_override=_version_override(payload, "agentVersion"),
            max_concurrency=_scout_max_concurrency(payload),
        )


@handler("topic_evaluate")
def handle_topic_evaluate(conn: Connection[Any], payload: dict[str, Any]) -> object:
    topic_id = payload.get("topicId") or payload.get("topic_id")
    if not topic_id:
        raise PermanentJobError("topic_evaluate payload requires topicId")
    with telemetry.span("topic_evaluate.process", **{"topic.id": str(topic_id)}):
        router = ModelRouter.from_yaml(_routing_config_path())
        provider, model = router.resolve("topic_judge")
        model_override = _override_value(payload, "model", "topicJudgeModel")
        if isinstance(model_override, str) and model_override.strip():
            model = model_override.strip()
        prompt_version = _version_override(payload, "promptVersion")
        rubric_version = _version_override(payload, "rubricVersion", "topicRubricVersion")
        weight_version = _override_int(payload, "weightVersion", "topicWeightVersion")
        trace = TraceRecorder(name="topic-judge")
        trace.trace(
            metadata={
                "jobType": "topic.evaluate",
                "topicId": str(topic_id),
                "agentVersion": _version_override(payload, "agentVersion"),
                "promptVersion": prompt_version,
                "rubricVersion": rubric_version,
                "weightVersion": weight_version,
            }
        )
        observed_provider = ObservedProvider(
            provider,
            trace,
            observation_name="topic-judge-structured",
            prompt_version="topic-judge@v2",
        )
        return run_judge(
            topic_id,
            observed_provider,
            model,
            AgentRepository(conn),
            _rubric_path(),
            recorder=trace,
            pass_threshold_override=_override_number(
                payload, "passThreshold", "topicPassThreshold"
            ),
            prompt_version_override=prompt_version,
            rubric_version_override=rubric_version,
            weight_version_override=weight_version,
            agent_version_override=_version_override(payload, "agentVersion"),
        )


@handler("article_write")
def handle_article_write(conn: Connection[Any], payload: dict[str, Any]) -> None:
    try:
        job = ArticleWriteJob.model_validate(payload)
    except ValidationError as exc:
        raise PermanentJobError(f"invalid article_write payload: {exc}") from exc
    with telemetry.span(
        "article_write.process",
        **{"topic.id": str(job.topicId), "article.platform": job.platform.value},
    ):
        router = ModelRouter.from_yaml(_routing_config_path())
        outline_provider, outline_model = router.resolve("writer_outline")
        draft_provider, draft_model = router.resolve("writer_draft")
        critic_provider, critic_model = router.resolve("writer_self_critic")
        outline_model = _override_model(payload, outline_model, "outlineModel", "model")
        draft_model = _override_model(payload, draft_model, "draftModel", "model")
        critic_model = _override_model(payload, critic_model, "criticModel", "model")
        prompt_version = _version_override(payload, "promptVersion")
        trace = TraceRecorder(name="article-writer")
        trace.trace(
            metadata={
                "jobType": "article.write",
                "topicId": str(job.topicId),
                "platform": job.platform.value,
                "agentVersion": _version_override(payload, "agentVersion"),
                "promptVersion": prompt_version,
            }
        )
        providers = WriterProviders(
            outline=ObservedProvider(
                outline_provider,
                trace,
                observation_name="writer-outline-structured",
                prompt_version="writer-outline@v1",
            ),
            draft=ObservedProvider(
                draft_provider,
                trace,
                observation_name="writer-draft-structured",
                prompt_version="writer-draft@v1",
            ),
            critic=ObservedProvider(
                critic_provider,
                trace,
                observation_name="writer-self-critic-structured",
                prompt_version="writer-self-critic@v1",
            ),
        )
        run_writer(
            job.topicId,
            job.platform,
            load_platform_profile(_profiles_path(), job.platform),
            providers,
            WriterModels(outline=outline_model, draft=draft_model, critic=critic_model),
            AgentRepository(conn),
            recorder=trace,
            rewrite=job.rewrite,
            replay=bool(job.replay),
            agent_version_override=_version_override(payload, "agentVersion"),
        )


@handler("article_evaluate")
def handle_article_evaluate(conn: Connection[Any], payload: dict[str, Any]) -> object:
    try:
        job = ArticleEvaluateJob.model_validate(payload)
    except ValidationError as exc:
        raise PermanentJobError(f"invalid article_evaluate payload: {exc}") from exc
    with telemetry.span("article_evaluate.process", **{"article.id": str(job.articleId)}):
        router = ModelRouter.from_yaml(_routing_config_path())
        provider, model = router.resolve("article_judge")
        model_override = _override_value(payload, "model", "articleJudgeModel")
        if isinstance(model_override, str) and model_override.strip():
            model = model_override.strip()
        prompt_version = _version_override(payload, "promptVersion")
        rubric_version = _version_override(payload, "rubricVersion", "articleRubricVersion")
        weight_version = _override_int(payload, "weightVersion", "articleWeightVersion")
        trace = TraceRecorder(name="article-judge")
        trace.trace(
            metadata={
                "jobType": "article.evaluate",
                "articleId": str(job.articleId),
                "agentVersion": _version_override(payload, "agentVersion"),
                "promptVersion": prompt_version,
                "rubricVersion": rubric_version,
                "weightVersion": weight_version,
            }
        )
        observed_provider = ObservedProvider(
            provider,
            trace,
            observation_name="article-judge-structured",
            prompt_version="article-judge@v1",
        )
        repository = AgentRepository(conn)
        article = repository.get_article(job.articleId)
        if article is None:
            raise PermanentJobError(f"article {job.articleId} not found")
        return run_article_judge(
            job.articleId,
            observed_provider,
            model,
            repository,
            _article_rubric_path(article.platform),
            recorder=trace,
            pass_threshold_override=_override_number(
                payload, "passThreshold", "articlePassThreshold"
            ),
            prompt_version_override=prompt_version,
            rubric_version_override=rubric_version,
            weight_version_override=weight_version,
            agent_version_override=_version_override(payload, "agentVersion"),
        )


@handler("memory_reflect")
def handle_memory_reflect(conn: Connection[Any], payload: dict[str, Any]) -> None:
    try:
        job = MemoryReflectJob.model_validate(payload)
    except ValidationError as exc:
        raise PermanentJobError(f"invalid memory_reflect payload: {exc}") from exc
    with telemetry.span(
        "memory_reflect.process",
        **{
            "memory.period_start": job.periodStart.isoformat(),
            "memory.period_end": job.periodEnd.isoformat(),
        },
    ):
        router = ModelRouter.from_yaml(_routing_config_path())
        provider, model = router.resolve("reflector")
        trace = TraceRecorder(name="memory-reflector")
        trace.trace(
            metadata={
                "jobType": "memory.reflect",
                "periodStart": job.periodStart.isoformat(),
                "periodEnd": job.periodEnd.isoformat(),
            }
        )
        observed = ObservedProvider(
            provider,
            trace,
            observation_name="memory-reflector-structured",
            prompt_version="memory-reflector@v1",
        )
        run_reflector(
            job.periodStart,
            job.periodEnd,
            observed,
            model,
            AgentRepository(conn),
            recorder=trace,
        )


def _routing_config_path() -> Path:
    return Path(__file__).resolve().parents[3] / "config" / "model_routing.yaml"


def _rubric_path() -> Path:
    return Path(__file__).resolve().parents[4] / "scholar-shared" / "rubrics" / "topic.v2.yaml"


def _article_rubric_path(platform: str) -> Path:
    if platform not in {"xiaohongshu", "zhihu", "wechat"}:
        raise PermanentJobError(f"unsupported article platform: {platform}")
    return (
        Path(__file__).resolve().parents[4]
        / "scholar-shared"
        / "rubrics"
        / f"article-{platform}.v1.yaml"
    )


def _profiles_path() -> Path:
    return Path(__file__).resolve().parents[4] / "scholar-shared" / "profiles"


def _connect_worker_database(dsn: str) -> Connection[Any]:
    return Connection.connect(dsn, row_factory=dict_row)


class Worker:
    def __init__(
        self,
        conn: Connection[Any],
        visibility_timeout: int = 300,
        *,
        job_timeout_seconds: float = DEFAULT_JOB_TIMEOUT_SECONDS,
        queues: list[str] | None = None,
    ) -> None:
        self._conn = conn
        self._vt = visibility_timeout
        self._job_timeout = job_timeout_seconds
        self._running = True
        selected = queues or list(HANDLERS)
        unknown = set(selected) - set(HANDLERS)
        if unknown:
            raise ValueError(f"unknown worker queues: {sorted(unknown)}")
        self._handlers = {queue: HANDLERS[queue] for queue in selected}

    def stop(self, _sig: int | None = None, _frm: types.FrameType | None = None) -> None:
        self._running = False

    def poll_once(self) -> bool:
        """每个进程只轮询显式选择的队列；队列级并发由多进程部署提供。"""
        got = False
        for queue, fn in self._handlers.items():
            row = self._read(queue)
            if row is None:
                continue
            self._conn.commit()
            got = True
            msg_id = int(row["msg_id"])
            read_count = int(row["read_ct"])
            payload = row["message"]
            if isinstance(payload, str):
                payload = json.loads(payload)
            if not isinstance(payload, dict):
                payload = {"value": payload}

            context = JobContext.from_message(queue, msg_id, read_count, payload, self._job_timeout)
            if self._already_completed(context.job_id):
                self._delete(queue, msg_id)
                self._conn.commit()
                log.info("job_duplicate_deleted", queue=queue, msg_id=msg_id)
                continue

            lease_seconds = math.ceil(self._job_timeout + DEFAULT_VISIBILITY_GRACE_SECONDS)
            self._set_visibility(queue, msg_id, lease_seconds)
            _workflow_event(
                self._conn,
                context,
                event_type="started",
                status="running",
                message="Agent 已开始执行",
                payload={"queue": queue, "attempt": read_count},
            )
            self._conn.commit()
            started_at = datetime.now(UTC)
            token = set_current_job(context)
            log.info(
                "job_start",
                queue=queue,
                msg_id=msg_id,
                job_id=str(context.job_id),
                correlation_id=str(context.correlation_id),
                attempt=read_count,
            )
            try:
                with (
                    telemetry.job_span(context, payload),
                    hard_deadline(context.remaining_seconds()),
                ):
                    result = fn(self._conn, payload)
                _record_workflow_decision(self._conn, context, payload, result)
                self._record_source_fetch_run(
                    context, payload, started_at, result=result, error=None
                )
                _workflow_event(
                    self._conn,
                    context,
                    event_type="succeeded",
                    status="succeeded",
                    message="Agent 执行完成",
                    payload=_workflow_result_payload(queue, result, payload),
                )
                self._mark_completed(context)
                self._delete(queue, msg_id)
                self._conn.commit()
                telemetry.record_job_outcome(context, "completed")
                log.info("job_done", queue=queue, msg_id=msg_id)
            except Exception as exc:  # noqa: BLE001 — job 边界必须兜住异常
                self._conn.rollback()
                retry = should_retry(exc, read_count)
                error_type = _error_type(exc)
                log.exception(
                    "job_failed",
                    queue=queue,
                    msg_id=msg_id,
                    read_count=read_count,
                    retry=retry,
                    error_type=error_type,
                )
                self._record_source_fetch_run(context, payload, started_at, result=None, error=exc)
                if retry:
                    delay = retry_delay_seconds(exc, read_count)
                    _workflow_event(
                        self._conn,
                        context,
                        event_type="retrying",
                        status="queued",
                        message=f"执行失败，将在 {delay}s 后重试",
                        payload={"errorType": _error_type(exc), "delaySeconds": delay},
                    )
                    self._set_visibility(queue, msg_id, delay)
                    self._conn.commit()
                    telemetry.record_job_outcome(context, "retry", error_type)
                    log.warning("job_retry_scheduled", queue=queue, msg_id=msg_id, delay=delay)
                else:
                    _record_workflow_failure_decision(self._conn, context, payload, exc)
                    _workflow_event(
                        self._conn,
                        context,
                        event_type="failed",
                        status="failed",
                        message="Agent 执行失败，任务已进入死信",
                        payload={
                            "errorType": _error_type(exc),
                            "error": str(exc)[:2000],
                            **{
                                key: payload[key]
                                for key in ("sourceId", "topicId", "articleId", "platform")
                                if key in payload
                            },
                        },
                    )
                    self._archive_failure(context, payload, exc)
                    self._conn.commit()
                    telemetry.record_job_outcome(context, "dead_letter", error_type)
                    log.error("job_dead_lettered", queue=queue, msg_id=msg_id)
            finally:
                reset_current_job(token)
        return got

    def _read(self, queue: str) -> dict[str, Any] | None:
        with (
            telemetry.span("messaging.read", **{"messaging.destination.name": queue}),
            self._conn.cursor(row_factory=dict_row) as cur,
        ):
            cur.execute(
                "select msg_id, read_ct, message from pgmq.read(%s, %s, 1)",
                (queue, self._vt),
            )
            return cur.fetchone()

    def _set_visibility(self, queue: str, msg_id: int, seconds: int) -> None:
        with (
            telemetry.span("messaging.extend_visibility", **{"messaging.destination.name": queue}),
            self._conn.cursor() as cur,
        ):
            cur.execute("select pgmq.set_vt(%s, %s, %s)", (queue, msg_id, seconds))

    def _delete(self, queue: str, msg_id: int) -> None:
        with (
            telemetry.span("messaging.delete", **{"messaging.destination.name": queue}),
            self._conn.cursor() as cur,
        ):
            cur.execute("select pgmq.delete(%s, %s)", (queue, msg_id))

    def _already_completed(self, job_id: UUID) -> bool:
        with self._conn.cursor() as cur:
            cur.execute("select 1 from job_receipts where job_id = %s", (job_id,))
            return cur.fetchone() is not None

    def _mark_completed(self, context: JobContext) -> None:
        with self._conn.cursor() as cur:
            cur.execute(
                """
                insert into job_receipts (job_id, queue, msg_id, correlation_id)
                values (%s, %s, %s, %s)
                on conflict (job_id) do nothing
                """,
                (context.job_id, context.queue, context.msg_id, context.correlation_id),
            )

    def _archive_failure(
        self, context: JobContext, payload: dict[str, Any], exc: BaseException
    ) -> None:
        retryable = exc.retryable if isinstance(exc, JobError) else True
        with self._conn.cursor() as cur:
            cur.execute(
                """
                insert into job_failures
                    (queue, msg_id, job_id, correlation_id, payload, read_count,
                     error_type, error_message, retryable, archived)
                values (%s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, true)
                on conflict (queue, msg_id) do update set
                    read_count = excluded.read_count,
                    error_type = excluded.error_type,
                    error_message = excluded.error_message,
                    retryable = excluded.retryable,
                    archived = true
                """,
                (
                    context.queue,
                    context.msg_id,
                    context.job_id,
                    context.correlation_id,
                    json.dumps(payload, ensure_ascii=False, default=str),
                    context.read_count,
                    _error_type(exc),
                    str(exc)[:2000],
                    retryable,
                ),
            )
            cur.execute("select pgmq.archive(%s, %s)", (context.queue, context.msg_id))

    def _record_source_fetch_run(
        self,
        context: JobContext,
        payload: dict[str, Any],
        started_at: datetime,
        *,
        result: object,
        error: BaseException | None,
    ) -> None:
        if context.queue != "source_fetch":
            return
        try:
            source_id = UUID(str(payload.get("sourceId")))
        except (TypeError, ValueError):
            return
        stats = result.as_dict() if isinstance(result, SourcingStats) else {}
        with self._conn.cursor() as cur:
            cur.execute(
                """
                insert into source_fetch_runs
                    (source_id, job_id, correlation_id, attempt, ok, stats,
                     error_type, error_message, started_at, finished_at)
                select %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, now()
                where exists (select 1 from sources where id = %s)
                """,
                (
                    source_id,
                    context.job_id,
                    context.correlation_id,
                    context.read_count,
                    error is None,
                    json.dumps(stats),
                    _error_type(error) if error else None,
                    str(error)[:2000] if error else None,
                    started_at,
                    source_id,
                ),
            )


def _error_type(exc: BaseException | None) -> str:
    if exc is None:
        return ""
    if isinstance(exc, ProviderError):
        return exc.error_code or f"provider_{exc.status_code or 'transport'}"
    return type(exc).__name__


def _selected_queues() -> list[str] | None:
    raw = os.environ.get("WORKER_QUEUES", "").strip()
    return [value.strip() for value in raw.split(",") if value.strip()] or None


def main() -> None:
    dsn = os.environ.get("DATABASE_URL")
    if not dsn:
        raise SystemExit("DATABASE_URL is required")
    try:
        telemetry.init_telemetry()
    except Exception as exc:  # noqa: BLE001 — 观测失败不能阻断业务
        log.warning("telemetry_initialization_failed", error=str(exc))
    try:
        with _connect_worker_database(dsn) as conn:
            worker = Worker(
                conn,
                visibility_timeout=int(os.environ.get("PGMQ_VISIBILITY_TIMEOUT_SECONDS", "300")),
                job_timeout_seconds=float(
                    os.environ.get("JOB_TIMEOUT_SECONDS", str(DEFAULT_JOB_TIMEOUT_SECONDS))
                ),
                queues=_selected_queues(),
            )
            signal.signal(signal.SIGINT, worker.stop)
            signal.signal(signal.SIGTERM, worker.stop)
            for queue in worker._handlers:  # noqa: SLF001
                telemetry.worker_concurrency.add(1, {"queue": queue})
            log.info("worker_started", queues=sorted(worker._handlers))  # noqa: SLF001
            try:
                while worker._running:  # noqa: SLF001
                    if not worker.poll_once():
                        time.sleep(1.0)
                log.info("worker_stopped")
            finally:
                for queue in worker._handlers:  # noqa: SLF001
                    telemetry.worker_concurrency.add(-1, {"queue": queue})
    finally:
        telemetry.shutdown_telemetry()


if __name__ == "__main__":
    main()
