"""M3 Reflector：确定性统计负责算数，LLM 只负责有证据的归因与经验提炼。"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from datetime import datetime
from typing import Any, Protocol
from uuid import UUID

from scholar_contracts.models import MemoryReflectOutput

from scholar_agents.db_access import (
    AgentRunInsert,
    InsightRecord,
    PerformanceSampleRecord,
)
from scholar_agents.embedding import embed
from scholar_agents.errors import PermanentJobError
from scholar_agents.observability import TraceRecorder
from scholar_agents.providers.base import ModelProvider, Usage
from scholar_agents.runtime.structured import StructuredOutputError, complete_structured

PROMPT_VERSION = "memory-reflector@v1"
ACTIVE_MIN_PUBLICATIONS = 5
ACTIVE_MIN_CONFIDENCE = 0.65
RETIRE_BELOW_CONFIDENCE = 0.35


class ReflectorRepository(Protocol):
    def list_reflection_samples(
        self, period_start: datetime, period_end: datetime
    ) -> list[PerformanceSampleRecord]: ...

    def list_reflection_insights(self, limit: int = 100) -> list[InsightRecord]: ...

    def create_agent_run(self, run: AgentRunInsert) -> UUID: ...

    def update_agent_run(self, run_id: UUID, run: AgentRunInsert) -> None: ...

    def create_insight(
        self,
        *,
        kind: str,
        platform: str | None,
        content: str,
        evidence: list[dict[str, Any]],
        confidence: float,
        status: str,
        embedding: list[float],
    ) -> UUID: ...

    def update_insight_from_reflection(
        self,
        insight_id: UUID,
        *,
        evidence: list[dict[str, Any]],
        confidence: float,
        status: str,
    ) -> bool: ...

    def create_weekly_report(
        self,
        *,
        period_start: datetime,
        period_end: datetime,
        sample_count: int,
        summary_markdown: str,
        calibration: dict[str, Any],
        agent_run_id: UUID,
    ) -> UUID: ...


@dataclass(frozen=True, slots=True)
class ReflectorResult:
    report_id: UUID
    insights_created: int
    insights_updated: int
    usage: Usage


def run_reflector(
    period_start: datetime,
    period_end: datetime,
    provider: ModelProvider,
    model: str,
    repository: ReflectorRepository,
    *,
    recorder: TraceRecorder,
) -> ReflectorResult:
    if period_end <= period_start:
        raise PermanentJobError("memory_reflect periodEnd must be after periodStart")
    samples = repository.list_reflection_samples(period_start, period_end)
    existing = repository.list_reflection_insights()
    calibration = build_calibration(samples)
    run_id = repository.create_agent_run(
        AgentRunInsert(
            job_type="memory.reflect",
            entity_type="weekly_report",
            entity_id=None,
            model=model,
            prompt_version=PROMPT_VERSION,
            tokens_in=0,
            tokens_out=0,
            cost_usd=None,
            langfuse_trace_id=recorder.trace_id,
            status="running",
        )
    )
    usage = Usage()
    created = 0
    updated = 0
    try:
        if samples:
            system, user = build_reflector_prompt(samples, existing, calibration)
            raw, usage = complete_structured(
                provider,
                model,
                system,
                user,
                MemoryReflectOutput.model_json_schema(),
                max_tokens=8192,
            )
            output = MemoryReflectOutput.model_validate(raw)
            validate_reflector_output(output, samples, existing)
            created, updated = apply_reflector_insights(output, existing, repository)
            summary = output.summaryMarkdown
        else:
            summary = (
                "# 本周数据回流报告\n\n"
                "本周期没有可比较的标准窗口快照。请先补录 24h / 72h / 7d 数据；"
                "当前不生成经验，也不提出权重调整建议。"
            )
        report_id = repository.create_weekly_report(
            period_start=period_start,
            period_end=period_end,
            sample_count=len(samples),
            summary_markdown=summary,
            calibration=calibration,
            agent_run_id=run_id,
        )
        repository.update_agent_run(
            run_id, _run(run_id, model, recorder.trace_id, usage, "succeeded")
        )
        return ReflectorResult(report_id, created, updated, usage)
    except StructuredOutputError as exc:
        usage.input_tokens += exc.usage.input_tokens
        usage.output_tokens += exc.usage.output_tokens
        repository.update_agent_run(run_id, _run(run_id, model, recorder.trace_id, usage, "failed"))
        raise
    except Exception:
        repository.update_agent_run(run_id, _run(run_id, model, recorder.trace_id, usage, "failed"))
        raise


def build_calibration(samples: list[PerformanceSampleRecord]) -> dict[str, Any]:
    correlations: list[dict[str, Any]] = []
    series: dict[str, list[tuple[float, float]]] = {
        "topic.total": [],
        "article.total": [],
    }
    for sample in samples:
        if sample.performance_percentile is None:
            continue
        p = sample.performance_percentile
        if sample.topic_score is not None:
            series["topic.total"].append((sample.topic_score, p))
        if sample.article_score is not None:
            series["article.total"].append((sample.article_score, p))
        for key, score in sample.topic_dimensions.items():
            series.setdefault(f"topic.dimension.{key}", []).append((score, p))
        for key, score in sample.article_dimensions.items():
            series.setdefault(f"article.dimension.{key}", []).append((score, p))
    for key in sorted(series):
        pairs = series[key]
        correlations.append(
            {"key": key, "sampleSize": len(pairs), "coefficient": pearson(pairs)}
        )
    high = [
        _case(sample, "high")
        for sample in samples
        if (sample.performance_percentile or 0) >= 75
    ]
    low = [
        _case(sample, "low")
        for sample in samples
        if sample.performance_percentile is not None and sample.performance_percentile <= 25
    ]
    high.sort(key=lambda item: float(item["percentile"]), reverse=True)
    low.sort(key=lambda item: float(item["percentile"]))
    return {
        "coldStart": len(samples) < 30,
        "correlations": correlations,
        "highCases": high[:10],
        "lowCases": low[:10],
    }


def pearson(pairs: list[tuple[float, float]]) -> float | None:
    if len(pairs) < 3:
        return None
    xs = [pair[0] for pair in pairs]
    ys = [pair[1] for pair in pairs]
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    numerator = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    denominator = math.sqrt(
        sum((x - mean_x) ** 2 for x in xs) * sum((y - mean_y) ** 2 for y in ys)
    )
    if denominator == 0:
        return None
    return round(numerator / denominator, 6)


def build_reflector_prompt(
    samples: list[PerformanceSampleRecord],
    existing: list[InsightRecord],
    calibration: dict[str, Any],
) -> tuple[str, str]:
    system = """你是 Scholars AI 的 Reflector。你的任务是从真实发布数据中找出可复用、可执行的规律。

硬性纪律：
1. 只引用输入中出现的 articleId/publicationId，不得发明证据。
2. 每条 insight 必须具体到下次 TopicScout 或 Writer 能执行的动作；“内容要高质量”无效。
3. 单篇偶然现象可以 create，但不能声称已被普遍证明。确定性代码会把证据不足的条目降为 candidate。
4. 区分可复制模式与热点/时机运气；相关性不等于因果。
5. support/contradict 必须填写 existingInsightId；create 必须为空。
6. 不提出自动调权。周报可提出人工检查建议，但必须标注样本量与冷启动状态。
7. summaryMarkdown 使用中文，包含：样本概况、高/低表现归因、评分预测力、
   下周可执行动作、风险/不确定性。"""
    sample_payload = [_sample_payload(sample) for sample in _reflection_subset(samples)]
    insight_payload = [
        {
            "id": str(item.id),
            "kind": item.kind,
            "platform": item.platform,
            "content": item.content,
            "confidence": item.confidence,
            "status": item.status,
            "evidence": item.evidence,
            "manualStatusOverride": item.manual_status_override,
        }
        for item in existing
    ]
    user = json.dumps(
        {
            "samples": sample_payload,
            "deterministicCalibration": calibration,
            "existingInsights": insight_payload,
        },
        ensure_ascii=False,
        indent=2,
    )
    return system, user


def validate_reflector_output(
    output: MemoryReflectOutput,
    samples: list[PerformanceSampleRecord],
    existing: list[InsightRecord],
) -> None:
    article_ids = {sample.article_id for sample in samples}
    publication_ids = {sample.publication_id for sample in samples}
    existing_ids = {item.id for item in existing if item.id is not None}
    for draft in output.insights:
        if draft.action.value == "create" and draft.existingInsightId is not None:
            raise PermanentJobError("Reflector create action cannot set existingInsightId")
        if draft.action.value != "create" and draft.existingInsightId not in existing_ids:
            raise PermanentJobError("Reflector support/contradict references unknown insight")
        for evidence in draft.evidence:
            if not evidence.articleIds and not evidence.publicationIds:
                raise PermanentJobError(
                    "Reflector evidence must reference an article or publication"
                )
            if not set(evidence.articleIds).issubset(article_ids):
                raise PermanentJobError(
                    "Reflector evidence references an article outside the period"
                )
            if not set(evidence.publicationIds).issubset(publication_ids):
                raise PermanentJobError(
                    "Reflector evidence references a publication outside the period"
                )


def apply_reflector_insights(
    output: MemoryReflectOutput,
    existing: list[InsightRecord],
    repository: ReflectorRepository,
) -> tuple[int, int]:
    by_id = {item.id: item for item in existing if item.id is not None}
    created = 0
    updated = 0
    for draft in output.insights:
        evidence = [item.model_dump(mode="json") for item in draft.evidence]
        if draft.action.value == "create":
            status = insight_status(evidence, draft.confidence)
            repository.create_insight(
                kind=draft.kind.value,
                platform=draft.platform.value if draft.platform is not None else None,
                content=draft.content,
                evidence=evidence,
                confidence=draft.confidence,
                status=status,
                embedding=embed(draft.content),
            )
            created += 1
            continue
        assert draft.existingInsightId is not None
        current = by_id[draft.existingInsightId]
        merged = merge_evidence(current.evidence, evidence)
        if draft.action.value == "support":
            confidence = min(1.0, max(current.confidence, draft.confidence) + 0.03)
            status = insight_status(merged, confidence)
        else:
            confidence = max(0.0, min(current.confidence, draft.confidence) - 0.15)
            status = (
                "retired"
                if confidence < RETIRE_BELOW_CONFIDENCE
                else (current.status or "candidate")
            )
        if repository.update_insight_from_reflection(
            draft.existingInsightId,
            evidence=merged,
            confidence=confidence,
            status=status,
        ):
            updated += 1
    return created, updated


def insight_status(evidence: list[dict[str, Any]], confidence: float) -> str:
    publications = {
        str(publication_id)
        for item in evidence
        for publication_id in item.get("publicationIds", [])
    }
    if len(publications) >= ACTIVE_MIN_PUBLICATIONS and confidence >= ACTIVE_MIN_CONFIDENCE:
        return "active"
    return "candidate"


def merge_evidence(
    old: list[dict[str, Any]], new: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    merged: list[dict[str, Any]] = []
    seen: set[str] = set()
    for item in [*old, *new]:
        key = json.dumps(item, ensure_ascii=False, sort_keys=True)
        if key in seen:
            continue
        seen.add(key)
        merged.append(item)
    return merged


def _reflection_subset(samples: list[PerformanceSampleRecord]) -> list[PerformanceSampleRecord]:
    high = sorted(
        (sample for sample in samples if (sample.performance_percentile or 0) >= 75),
        key=lambda sample: sample.performance_percentile or 0,
        reverse=True,
    )[:20]
    low = sorted(
        (
            sample
            for sample in samples
            if sample.performance_percentile is not None and sample.performance_percentile <= 25
        ),
        key=lambda sample: sample.performance_percentile or 0,
    )[:20]
    seen: set[tuple[UUID, str]] = set()
    result: list[PerformanceSampleRecord] = []
    for sample in [*high, *low]:
        key = (sample.publication_id, sample.snapshot_window)
        if key not in seen:
            seen.add(key)
            result.append(sample)
    return result


def _sample_payload(sample: PerformanceSampleRecord) -> dict[str, Any]:
    return {
        "publicationId": str(sample.publication_id),
        "articleId": str(sample.article_id),
        "platform": sample.platform,
        "snapshotWindow": sample.snapshot_window,
        "capturedAt": sample.captured_at.isoformat(),
        "performanceRaw": sample.performance_raw,
        "performancePercentile": sample.performance_percentile,
        "metrics": sample.metrics,
        "article": {
            "title": sample.article_title,
            "contentExcerpt": sample.article_content[:2000],
            "judgeScore": sample.article_score,
            "judgeDimensions": sample.article_dimensions,
        },
        "topic": {
            "id": str(sample.topic_id),
            "title": sample.topic_title,
            "angle": sample.topic_angle,
            "judgeScore": sample.topic_score,
            "judgeDimensions": sample.topic_dimensions,
        },
    }


def _case(sample: PerformanceSampleRecord, band: str) -> dict[str, Any]:
    return {
        "publicationId": str(sample.publication_id),
        "articleId": str(sample.article_id),
        "platform": sample.platform,
        "title": sample.article_title,
        "snapshotWindow": sample.snapshot_window,
        "percentile": sample.performance_percentile,
        "performanceRaw": sample.performance_raw,
        "capturedAt": sample.captured_at.isoformat(),
        "band": band,
    }


def _run(
    run_id: UUID,
    model: str,
    trace_id: str | None,
    usage: Usage,
    status: str,
) -> AgentRunInsert:
    del run_id
    return AgentRunInsert(
        job_type="memory.reflect",
        entity_type="weekly_report",
        entity_id=None,
        model=model,
        prompt_version=PROMPT_VERSION,
        tokens_in=usage.input_tokens,
        tokens_out=usage.output_tokens,
        cost_usd=None,
        langfuse_trace_id=trace_id,
        status=status,  # type: ignore[arg-type]
    )
