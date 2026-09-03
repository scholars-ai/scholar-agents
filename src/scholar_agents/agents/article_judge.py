"""ArticleJudge：按平台 rubric 独立评分，代码重算总分与 veto。"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Protocol
from uuid import UUID

import yaml

from scholar_agents.db_access import (
    AgentRunInsert,
    AgentRunStatus,
    ArticleEvaluationInsert,
    ArticleRecord,
    RawItemRecord,
    TopicRecord,
    WeightSetRecord,
)
from scholar_agents.errors import PermanentJobError
from scholar_agents.observability import TraceRecorder
from scholar_agents.providers.base import ModelProvider, Usage
from scholar_agents.runtime.structured import StructuredOutputError, complete_structured

PROMPT_VERSION = "article-judge@v1"
AGENT_VERSION = "article-judge@v1"


class ArticleJudgeError(PermanentJobError):
    """实体、rubric 或权重配置无法安全评分。"""


@dataclass(frozen=True, slots=True)
class ArticleDimension:
    key: str
    name: str
    description: str
    initial_weight: float
    veto_below: float | None


@dataclass(frozen=True, slots=True)
class ArticleRubric:
    rubric_id: str
    version: str
    pass_threshold: float
    dimensions: tuple[ArticleDimension, ...]

    @property
    def dimension_keys(self) -> tuple[str, ...]:
        return tuple(item.key for item in self.dimensions)


@dataclass(frozen=True, slots=True)
class ArticleScore:
    total_score: float
    vetoed_dimension: str | None
    passed: bool
    pass_threshold: float = 70.0


@dataclass(frozen=True, slots=True)
class ArticleJudgeResult:
    evaluation_id: UUID
    agent_run_id: UUID
    usage: Usage
    score: ArticleScore


def _version_override(value: str | None, default: str, kind: str) -> str:
    if value is None:
        return default
    normalized = value.strip()
    if not normalized:
        raise ArticleJudgeError(f"{kind} version override must not be empty")
    return normalized


def _validate_rubric_version(actual: str, requested: str | None) -> None:
    if requested is not None and requested.strip() != actual:
        raise ArticleJudgeError(
            f"requested rubric version {requested!r} does not match loaded rubric {actual!r}"
        )


class ArticleJudgeRepository(Protocol):
    def get_article(self, article_id: UUID) -> ArticleRecord | None: ...

    def get_topic(self, topic_id: UUID) -> TopicRecord | None: ...

    def list_topic_raw_items(self, topic: TopicRecord) -> list[RawItemRecord]: ...

    def get_active_weight_set(self, rubric_id: str) -> WeightSetRecord | None: ...

    def create_agent_run(self, run: AgentRunInsert) -> UUID: ...

    def update_agent_run(self, run_id: UUID, run: AgentRunInsert) -> None: ...

    def create_article_evaluation(self, evaluation: ArticleEvaluationInsert) -> UUID: ...


def run_article_judge(
    article_id: UUID,
    provider: ModelProvider,
    model: str,
    repository: ArticleJudgeRepository,
    rubric_path: Path,
    *,
    recorder: TraceRecorder,
    pass_threshold_override: float | None = None,
    prompt_version_override: str | None = None,
    rubric_version_override: str | None = None,
    weight_version_override: int | None = None,
    agent_version_override: str | None = None,
) -> ArticleJudgeResult:
    rubric = load_article_rubric(rubric_path)
    prompt_version = _version_override(prompt_version_override, PROMPT_VERSION, "prompt")
    agent_version = _version_override(agent_version_override, AGENT_VERSION, "agent")
    _validate_rubric_version(rubric.version, rubric_version_override)
    if pass_threshold_override is not None:
        if not 0 <= pass_threshold_override <= 100:
            raise ArticleJudgeError("pass threshold override must be between 0 and 100")
        rubric = replace(rubric, pass_threshold=pass_threshold_override)
    article = repository.get_article(article_id)
    if article is None:
        raise ArticleJudgeError(f"article {article_id} not found")
    if article.status not in {"draft", "pending_review"}:
        raise ArticleJudgeError(f"article {article_id} cannot be evaluated: {article.status}")
    if rubric.rubric_id != f"article/{article.platform}":
        raise ArticleJudgeError("article platform does not match rubric")
    topic = repository.get_topic(article.topic_id)
    if topic is None:
        raise ArticleJudgeError(f"topic {article.topic_id} not found")
    weight_set = repository.get_active_weight_set(rubric.rubric_id)
    if weight_set is None:
        raise ArticleJudgeError(f"no active weight set for {rubric.rubric_id}")
    if weight_version_override is not None and weight_set.version != weight_version_override:
        raise ArticleJudgeError(
            "requested article weight version "
            f"{weight_version_override} is not active (active={weight_set.version})"
        )
    raw_items = repository.list_topic_raw_items(topic)
    system, user = build_article_judge_prompt(article, topic, raw_items, rubric)
    run_id = repository.create_agent_run(
        AgentRunInsert(
            job_type="article.evaluate",
            entity_type="article",
            entity_id=article.id,
            model=model,
            agent_version=agent_version,
            prompt_version=prompt_version,
            tokens_in=0,
            tokens_out=0,
            cost_usd=None,
            langfuse_trace_id=recorder.trace_id,
            status="running",
        )
    )
    usage = Usage()
    try:
        data, usage = complete_structured(
            provider,
            model,
            system,
            user,
            article_judge_output_schema(rubric),
            max_tokens=4096,
        )
        scores, reasons, rationale = parse_article_judge_output(data, rubric)
        score = recompute_article_score(rubric, weight_set, scores)
        evaluation_id = repository.create_article_evaluation(
            ArticleEvaluationInsert(
                article_id=article.id,
                rubric_version=rubric.version,
                dimension_scores=scores,
                dimension_reasons=reasons,
                total_score=score.total_score,
                rationale=rationale,
                judge_model=model,
                agent_run_id=run_id,
                weight_version=weight_set.version,
                vetoed_dimension=score.vetoed_dimension,
                pass_threshold=rubric.pass_threshold,
                passed=score.passed,
            )
        )
        repository.update_agent_run(
            run_id,
            _run_update(
                article.id,
                model,
                recorder.trace_id,
                usage,
                "succeeded",
                prompt_version,
                agent_version,
            ),
        )
        recorder.score(name="article_total_score", value=score.total_score, comment=rationale)
        return ArticleJudgeResult(evaluation_id, run_id, usage, score)
    except StructuredOutputError as exc:
        repository.update_agent_run(
            run_id,
            _run_update(
                article.id,
                model,
                recorder.trace_id,
                exc.usage,
                "failed",
                prompt_version,
                agent_version,
            ),
        )
        raise
    except Exception:
        repository.update_agent_run(
            run_id,
            _run_update(
                article.id, model, recorder.trace_id, usage, "failed", prompt_version, agent_version
            ),
        )
        raise


def load_article_rubric(path: Path) -> ArticleRubric:
    data = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ArticleJudgeError("article rubric must be a mapping")
    dimensions_raw = data.get("dimensions")
    if not isinstance(dimensions_raw, list) or not dimensions_raw:
        raise ArticleJudgeError("article rubric dimensions must be non-empty")
    dimensions: list[ArticleDimension] = []
    for raw in dimensions_raw:
        if not isinstance(raw, dict):
            raise ArticleJudgeError("article rubric dimension must be a mapping")
        dimensions.append(
            ArticleDimension(
                key=_required_string(raw, "key"),
                name=_required_string(raw, "name"),
                description=_required_string(raw, "description"),
                initial_weight=float(raw.get("initialWeight", 0)),
                veto_below=(float(raw["vetoBelow"]) if raw.get("vetoBelow") is not None else None),
            )
        )
    keys = [item.key for item in dimensions]
    if len(keys) != len(set(keys)):
        raise ArticleJudgeError("article rubric dimension keys must be unique")
    rubric_id = _required_string(data, "id")
    version = _required_string(data, "version")
    return ArticleRubric(
        rubric_id=rubric_id,
        version=f"{rubric_id}@{version}",
        pass_threshold=float(data.get("passThreshold", 70)),
        dimensions=tuple(dimensions),
    )


def article_judge_output_schema(rubric: ArticleRubric) -> dict[str, Any]:
    score_schema = {
        "type": "object",
        "additionalProperties": False,
        "required": ["score", "reason"],
        "properties": {
            "score": {"type": "number", "minimum": 0, "maximum": 10},
            "reason": {"type": "string", "minLength": 1},
        },
    }
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["dimensionScores", "rationale"],
        "properties": {
            "dimensionScores": {
                "type": "object",
                "additionalProperties": False,
                "required": list(rubric.dimension_keys),
                "properties": {key: score_schema for key in rubric.dimension_keys},
            },
            "rationale": {"type": "string", "minLength": 1},
        },
    }


def parse_article_judge_output(
    data: dict[str, Any], rubric: ArticleRubric
) -> tuple[dict[str, float], dict[str, str], str]:
    raw_scores = data.get("dimensionScores")
    if not isinstance(raw_scores, dict):
        raise ArticleJudgeError("dimensionScores must be an object")
    if set(raw_scores) != set(rubric.dimension_keys):
        raise ArticleJudgeError("article judge dimension keys mismatch")
    scores: dict[str, float] = {}
    reasons: dict[str, str] = {}
    for key in rubric.dimension_keys:
        item = raw_scores[key]
        if not isinstance(item, dict):
            raise ArticleJudgeError(f"dimension {key} must be an object")
        scores[key] = float(item["score"])
        reasons[key] = str(item["reason"])
    rationale = str(data.get("rationale") or "").strip()
    if not rationale:
        raise ArticleJudgeError("article judge rationale is required")
    return scores, reasons, rationale


def recompute_article_score(
    rubric: ArticleRubric,
    weight_set: WeightSetRecord,
    scores: dict[str, float],
) -> ArticleScore:
    expected = set(rubric.dimension_keys)
    if set(scores) != expected or set(weight_set.weights) != expected:
        raise ArticleJudgeError("rubric, scores and weight set dimensions must match")
    if weight_set.rubric_id != rubric.rubric_id:
        raise ArticleJudgeError("weight set rubric mismatch")
    if any(not 0 <= value <= 10 for value in scores.values()):
        raise ArticleJudgeError("dimension scores must be between 0 and 10")
    total_weight = sum(weight_set.weights.values())
    if total_weight <= 0:
        raise ArticleJudgeError("article weights must sum to a positive value")
    total = sum(scores[key] * 10 * weight_set.weights[key] / total_weight for key in expected)
    vetoed = next(
        (
            item.key
            for item in rubric.dimensions
            if item.veto_below is not None and scores[item.key] < item.veto_below
        ),
        None,
    )
    return ArticleScore(
        total_score=total,
        vetoed_dimension=vetoed,
        passed=total >= rubric.pass_threshold and vetoed is None,
        pass_threshold=rubric.pass_threshold,
    )


def build_article_judge_prompt(
    article: ArticleRecord,
    topic: TopicRecord,
    raw_items: list[RawItemRecord],
    rubric: ArticleRubric,
) -> tuple[str, str]:
    dimensions = "\n".join(
        f"- {item.key}（{item.name}）：{item.description}"
        + (f"；低于 {item.veto_below} 一票否决" if item.veto_below is not None else "")
        for item in rubric.dimensions
    )
    system = f"""你是独立 ArticleJudge，不参与写作，也不知道 Writer 的内部过程。
请按 {rubric.version} 对每个维度给出 0–10 分和具体理由：
{dimensions}

只依据文章、选题和原始素材评分。不要输出总分或 passed，代码会按生效权重、过审线
{rubric.pass_threshold:.0f} 和 veto 规则确定性计算。事实准确性必须逐项对照素材。"""
    materials = (
        "\n\n---\n\n".join(
            f"素材 {index}\n标题：{item.title}\n来源：{item.source_name}\n"
            f"URL：{item.url or '无'}\n正文：{item.content}"
            for index, item in enumerate(raw_items, start=1)
        )
        or "无额外原文；只能核对选题标题、角度和摘要中明确提供的事实。"
    )
    user = f"""平台：{article.platform}
文章版本：{article.version}
选题：{topic.title}
角度：{topic.angle}
摘要：{topic.summary}

待评文章标题：{article.title}
待评正文：
{article.content_md}

原始素材：
{materials}
"""
    return system, user


def _run_update(
    article_id: UUID,
    model: str,
    trace_id: str,
    usage: Usage,
    status: AgentRunStatus,
    prompt_version: str = PROMPT_VERSION,
    agent_version: str = AGENT_VERSION,
) -> AgentRunInsert:
    return AgentRunInsert(
        job_type="article.evaluate",
        entity_type="article",
        entity_id=article_id,
        model=model,
        agent_version=agent_version,
        prompt_version=prompt_version,
        tokens_in=usage.input_tokens,
        tokens_out=usage.output_tokens,
        cost_usd=None,
        langfuse_trace_id=trace_id,
        status=status,
    )


def _required_string(data: dict[str, Any], key: str) -> str:
    value = data.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ArticleJudgeError(f"rubric field {key!r} must be a non-empty string")
    return value
