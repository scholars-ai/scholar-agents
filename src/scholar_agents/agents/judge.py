"""TopicJudge 的 rubric、上下文和确定性评分逻辑。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol

import yaml
from scholar_contracts.models import TopicJudgeOutput

from scholar_agents import telemetry
from scholar_agents.db_access import (
    AgentRunInsert,
    RawItemRecord,
    TopicEvaluationInsert,
    TopicRecord,
    WeightSetRecord,
)
from scholar_agents.observability import TraceRecorder
from scholar_agents.providers.base import ModelProvider, Usage
from scholar_agents.runtime.structured import StructuredOutputError, complete_structured

RUBRIC_VERSION = "topic@v2"
PROMPT_VERSION = "topic-judge@v2"


class TopicJudgeError(ValueError):
    """Rubric、权重或模型评分无法安全处理。"""


@dataclass(frozen=True, slots=True)
class TopicDimension:
    key: str
    name: str
    description: str
    initial_weight: float
    veto_below: float | None


@dataclass(frozen=True, slots=True)
class TopicRubric:
    rubric_id: str
    version: str
    pass_threshold: float
    dimensions: tuple[TopicDimension, ...]

    @property
    def dimension_keys(self) -> tuple[str, ...]:
        return tuple(dimension.key for dimension in self.dimensions)

    @property
    def veto_thresholds(self) -> dict[str, float]:
        return {
            dimension.key: dimension.veto_below
            for dimension in self.dimensions
            if dimension.veto_below is not None
        }


@dataclass(frozen=True, slots=True)
class TopicScore:
    total_score: float
    effective_weights: dict[str, float]
    vetoed_dimension: str | None


class JudgeRepository(Protocol):
    """Judge 所需的最小 repository 协议，便于纯单测和真实 DB 解耦。"""

    def get_topic(self, topic_id: Any) -> TopicRecord | None: ...

    def list_topic_raw_items(self, topic: TopicRecord) -> list[RawItemRecord]: ...

    def get_active_weight_set(self, rubric_id: str) -> WeightSetRecord | None: ...

    def create_agent_run(self, run: AgentRunInsert) -> Any: ...

    def update_agent_run(self, run_id: Any, run: AgentRunInsert) -> None: ...

    def create_topic_evaluation(self, evaluation: TopicEvaluationInsert) -> Any: ...


@dataclass(frozen=True, slots=True)
class JudgeResult:
    evaluation_id: Any
    agent_run_id: Any
    usage: Usage
    total_score: float
    vetoed_dimension: str | None
    passed: bool
    pass_threshold: float = 60.0


def run_judge(
    topic_id: Any,
    provider: ModelProvider,
    model: str,
    repository: JudgeRepository,
    rubric_path: Path,
    *,
    recorder: TraceRecorder | None = None,
    cold_start: bool = True,
    pass_threshold_override: float | None = None,
) -> JudgeResult:
    with telemetry.span("rubric.load"):
        rubric = load_topic_rubric(rubric_path)
    if pass_threshold_override is not None:
        if not 0 <= pass_threshold_override <= 100:
            raise TopicJudgeError("pass threshold override must be between 0 and 100")
        rubric = replace(rubric, pass_threshold=pass_threshold_override)
    with telemetry.span("topic.load", **{"topic.id": str(topic_id)}):
        topic = repository.get_topic(topic_id)
    if topic is None:
        raise TopicJudgeError(f"topic {topic_id} not found")
    with telemetry.span("weight_set.load"):
        weight_set = repository.get_active_weight_set(rubric.rubric_id)
    if weight_set is None:
        raise TopicJudgeError(f"no active weight set for {rubric.rubric_id!r}")
    with telemetry.span("topic.materials.load", **{"topic.id": str(topic.id)}):
        raw_items = repository.list_topic_raw_items(topic)
    system, user = build_judge_prompt(topic, raw_items, rubric)
    trace = recorder or TraceRecorder(name="topic-judge")
    run_id = repository.create_agent_run(
        AgentRunInsert(
            job_type="topic.evaluate",
            entity_type="topic",
            entity_id=topic.id,
            model=model,
            prompt_version=PROMPT_VERSION,
            tokens_in=0,
            tokens_out=0,
            cost_usd=None,
            langfuse_trace_id=trace.trace_id,
            status="running",
        )
    )
    try:
        data, usage = complete_structured(
            provider,
            model,
            system,
            user,
            TopicJudgeOutput.model_json_schema(),
        )
        output = parse_judge_output(data)
        scores = {key: getattr(output.dimensionScores, key).score for key in rubric.dimension_keys}
        reasons = {
            key: getattr(output.dimensionScores, key).reason for key in rubric.dimension_keys
        }
        with telemetry.span("score.recompute"):
            score = recompute_topic_score(rubric, weight_set, scores, cold_start=cold_start)
        with telemetry.span("evaluation.insert", **{"topic.id": str(topic.id)}):
            evaluation_id = repository.create_topic_evaluation(
                TopicEvaluationInsert(
                    topic_id=topic.id,
                    rubric_version=rubric.version,
                    dimension_scores=scores,
                    dimension_reasons=reasons,
                    total_score=score.total_score,
                    rationale=output.rationale,
                    judge_model=model,
                    agent_run_id=run_id,
                    weight_version=weight_set.version,
                    vetoed_dimension=score.vetoed_dimension,
                )
            )
        with telemetry.span("agent_run.update"):
            repository.update_agent_run(
                run_id,
                AgentRunInsert(
                    job_type="topic.evaluate",
                    entity_type="topic",
                    entity_id=topic.id,
                    model=model,
                    prompt_version=PROMPT_VERSION,
                    tokens_in=usage.input_tokens,
                    tokens_out=usage.output_tokens,
                    cost_usd=None,
                    langfuse_trace_id=trace.trace_id,
                    status="succeeded",
                ),
            )
        trace.score(name="topic_total_score", value=score.total_score, comment=output.rationale)
        return JudgeResult(
            evaluation_id,
            run_id,
            usage,
            score.total_score,
            score.vetoed_dimension,
            score.vetoed_dimension is None and score.total_score >= rubric.pass_threshold,
            rubric.pass_threshold,
        )
    except StructuredOutputError as exc:
        repository.update_agent_run(
            run_id,
            AgentRunInsert(
                job_type="topic.evaluate",
                entity_type="topic",
                entity_id=topic.id,
                model=model,
                prompt_version=PROMPT_VERSION,
                tokens_in=exc.usage.input_tokens,
                tokens_out=exc.usage.output_tokens,
                cost_usd=None,
                langfuse_trace_id=trace.trace_id,
                status="failed",
            ),
        )
        raise
    except Exception:
        repository.update_agent_run(
            run_id,
            AgentRunInsert(
                job_type="topic.evaluate",
                entity_type="topic",
                entity_id=topic.id,
                model=model,
                prompt_version=PROMPT_VERSION,
                tokens_in=None,
                tokens_out=None,
                cost_usd=None,
                langfuse_trace_id=trace.trace_id,
                status="failed",
            ),
        )
        raise


def load_topic_rubric(path: Path) -> TopicRubric:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise TopicJudgeError("topic rubric must be a mapping")
    rubric_id = _required_string(data, "id")
    version = _required_string(data, "version")
    dimensions_data = data.get("dimensions")
    if not isinstance(dimensions_data, list) or not dimensions_data:
        raise TopicJudgeError("topic rubric dimensions must be a non-empty list")

    dimensions: list[TopicDimension] = []
    for raw_dimension in dimensions_data:
        if not isinstance(raw_dimension, dict):
            raise TopicJudgeError("each rubric dimension must be a mapping")
        key = _required_string(raw_dimension, "key")
        dimensions.append(
            TopicDimension(
                key=key,
                name=_required_string(raw_dimension, "name"),
                description=_required_string(raw_dimension, "description"),
                initial_weight=float(raw_dimension.get("initialWeight", 0)),
                veto_below=_optional_float(raw_dimension.get("vetoBelow")),
            )
        )
    keys = [dimension.key for dimension in dimensions]
    if len(set(keys)) != len(keys):
        raise TopicJudgeError("topic rubric dimension keys must be unique")
    return TopicRubric(
        rubric_id=rubric_id,
        version=f"{rubric_id}@{version}",
        pass_threshold=float(data.get("passThreshold", 60)),
        dimensions=tuple(dimensions),
    )


def recompute_topic_score(
    rubric: TopicRubric,
    weight_set: WeightSetRecord,
    dimension_scores: dict[str, float],
    *,
    cold_start: bool = False,
) -> TopicScore:
    expected = set(rubric.dimension_keys)
    actual = set(dimension_scores)
    if actual != expected:
        raise TopicJudgeError(
            f"dimension keys mismatch: expected {sorted(expected)}, got {sorted(actual)}"
        )
    if weight_set.rubric_id != rubric.rubric_id:
        raise TopicJudgeError("weight set rubric does not match topic rubric")
    if set(weight_set.weights) != expected:
        raise TopicJudgeError("weight set dimension keys mismatch")
    if any(not 0 <= score <= 10 for score in dimension_scores.values()):
        raise TopicJudgeError("dimension scores must be between 0 and 10")

    weights = dict(weight_set.weights)
    if cold_start:
        weights["history_signal"] = 0.0
    total_weight = sum(weights.values())
    if total_weight <= 0:
        raise TopicJudgeError("effective topic weights must sum to a positive value")
    effective_weights = {key: value / total_weight for key, value in weights.items()}
    total_score = sum(dimension_scores[key] * 10 * effective_weights[key] for key in expected)
    vetoed = next(
        (
            dimension.key
            for dimension in rubric.dimensions
            if dimension.veto_below is not None
            and dimension_scores[dimension.key] < dimension.veto_below
        ),
        None,
    )
    return TopicScore(
        total_score=total_score,
        effective_weights=effective_weights,
        vetoed_dimension=vetoed,
    )


def build_judge_prompt(
    topic: TopicRecord,
    raw_items: list[RawItemRecord],
    rubric: TopicRubric,
) -> tuple[str, str]:
    dimensions = "\n".join(
        f"- {dimension.key}（{dimension.name}）：{dimension.description}"
        for dimension in rubric.dimensions
    )
    system = f"""你是 TopicJudge，负责评估一个候选选题是否值得进入文章创作。

请严格按照以下 rubric 对每个维度给出 0–10 分和具体理由：
{dimensions}

当前 rubric 版本是 {rubric.version}。不要输出总分，代码会根据生效权重重新计算。
当前 rubric 没有一票否决维度；不要自行创造 veto 规则。
评分时必须遵守以下要求：
- 面向中文社区判断受众价值和平台适配，不要因为英文标题或名词陌生就自动高分；
  需要说明普通中文读者为什么会关心，或需要怎样本地化解释。
- 检查候选是否只是素材标题或同一事件的换词改写；如果没有独立切口，应降低差异化空间，
  并在理由中明确指出。
- 优先评价具体、可展开、有明确冲突或影响的切口；“繁荣与隐忧”“科技与自然的博弈”等宽泛表述
  不能仅凭宏大主题高分。
- 每个维度理由都要引用候选选题或关联素材中的具体事实；总体 rationale 必须解释最高分和最低分维度，
  且与各维度理由一致。
输出必须符合 TopicJudgeOutput schema。"""
    material_sections = []
    for index, item in enumerate(raw_items, start=1):
        published = item.published_at.isoformat() if item.published_at else "未知"
        material_sections.append(
            "\n".join(
                [
                    f"素材 {index}",
                    f"标题：{item.title}",
                    f"来源：{item.source_name}（权重 {item.source_weight:.2f}）",
                    f"发布时间：{published}",
                    f"正文：{item.content}",
                ]
            )
        )
    user = "\n".join(
        [
            f"候选选题：{topic.title}",
            f"创作角度：{topic.angle}",
            f"摘要：{topic.summary}",
            f"建议平台：{', '.join(topic.target_platforms) or '未指定'}",
            "",
            "关联素材：",
            "\n\n---\n\n".join(material_sections) or "无",
        ]
    )
    return system, user


def parse_judge_output(data: dict[str, Any]) -> TopicJudgeOutput:
    try:
        return TopicJudgeOutput.model_validate(data)
    except Exception as exc:  # pydantic's validation error is intentionally normalized here
        raise TopicJudgeError(f"invalid TopicJudge output: {exc}") from exc


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise TopicJudgeError(f"rubric field {key!r} must be a non-empty string")
    return value


def _optional_float(value: object) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float, str)):
        return float(value)
    raise TopicJudgeError("rubric numeric field must be a number")
