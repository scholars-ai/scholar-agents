from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from scholar_agents.agents.judge import (
    RUBRIC_VERSION,
    TopicJudgeError,
    build_judge_prompt,
    load_topic_rubric,
    recompute_topic_score,
)
from scholar_agents.db_access import RawItemRecord, TopicRecord, WeightSetRecord

RUBRIC_PATH = Path(__file__).parents[2] / "scholar-shared/rubrics/topic.v1.yaml"


def test_load_topic_rubric_reads_six_dimensions_without_veto() -> None:
    rubric = load_topic_rubric(RUBRIC_PATH.resolve())

    assert rubric.version == RUBRIC_VERSION == "topic@v1"
    assert rubric.dimension_keys == (
        "timeliness",
        "audience_value",
        "platform_fit",
        "differentiation",
        "material_richness",
        "history_signal",
    )
    assert rubric.veto_thresholds == {}


def test_recompute_topic_score_normalizes_cold_start_weights() -> None:
    rubric = load_topic_rubric(RUBRIC_PATH.resolve())
    scores = {key: 5.0 for key in rubric.dimension_keys}
    scores["history_signal"] = 0.0
    weights = WeightSetRecord("topic", 4, {key: 0.0 for key in rubric.dimension_keys})
    weights = WeightSetRecord(
        "topic",
        4,
        {
            "timeliness": 0.2,
            "audience_value": 0.25,
            "platform_fit": 0.15,
            "differentiation": 0.15,
            "material_richness": 0.1,
            "history_signal": 0.15,
        },
    )

    result = recompute_topic_score(rubric, weights, scores, cold_start=True)

    assert result.total_score == pytest.approx(50.0)
    assert result.effective_weights["history_signal"] == 0.0
    assert sum(result.effective_weights.values()) == pytest.approx(1.0)
    assert result.vetoed_dimension is None


def test_recompute_topic_score_rejects_missing_or_unknown_dimensions() -> None:
    rubric = load_topic_rubric(RUBRIC_PATH.resolve())
    weights = WeightSetRecord("topic", 1, {key: 1 / 6 for key in rubric.dimension_keys})

    with pytest.raises(TopicJudgeError, match="dimension keys"):
        recompute_topic_score(rubric, weights, {"timeliness": 8.0})

    with pytest.raises(TopicJudgeError, match="dimension keys"):
        recompute_topic_score(
            rubric,
            weights,
            {key: 5.0 for key in rubric.dimension_keys} | {"unknown": 5.0},
        )


def test_build_judge_prompt_includes_topic_and_material_context() -> None:
    topic = TopicRecord(
        id=uuid4(),
        title="模型发布",
        angle="工程落地",
        summary="分析影响",
        raw_item_ids=[],
        target_platforms=["zhihu"],
        status="candidate",
        latest_score=None,
    )
    item = RawItemRecord(
        id=uuid4(),
        source_id=uuid4(),
        title="原始标题",
        url="https://example.com",
        author=None,
        content="原始正文",
        published_at=None,
        source_name="信源",
        source_weight=0.8,
        embedding=None,
    )
    rubric = load_topic_rubric(RUBRIC_PATH.resolve())

    system, user = build_judge_prompt(topic, [item], rubric)

    assert "TopicJudge" in system
    assert "timeliness" in system
    assert "模型发布" in user
    assert "原始正文" in user


class FakeJudgeProvider:
    name = "fake"

    def complete(self, model: str, request: object) -> object:
        import json

        from scholar_agents.providers.base import ChatResponse, TextBlock, Usage

        del model, request
        return ChatResponse(
            content=[
                TextBlock(
                    text=json.dumps(
                        {
                            "dimensionScores": {
                                key: {"score": 5, "reason": f"{key} reason"}
                                for key in (
                                    "timeliness",
                                    "audience_value",
                                    "platform_fit",
                                    "differentiation",
                                    "material_richness",
                                    "history_signal",
                                )
                            },
                            "rationale": "总体理由",
                        }
                    )
                )
            ],
            stop_reason="end_turn",
            usage=Usage(input_tokens=12, output_tokens=34),
            model="fake-model",
        )


class FakeTrace:
    trace_id = "trace-j-1"
    scores: list[tuple[str, float]] = []

    def score(self, *, name: str, value: float, comment: str | None = None) -> None:
        del comment
        self.scores.append((name, value))


class FakeJudgeRepository:
    def __init__(
        self, topic: TopicRecord, item: RawItemRecord, weight_set: WeightSetRecord
    ) -> None:
        self.topic = topic
        self.item = item
        self.weight_set = weight_set
        self.runs = []
        self.evaluations = []
        self.updates = []

    def get_topic(self, topic_id: object) -> TopicRecord | None:
        return self.topic if topic_id == self.topic.id else None

    def list_topic_raw_items(self, topic: TopicRecord) -> list[RawItemRecord]:
        assert topic.id == self.topic.id
        return [self.item]

    def get_active_weight_set(self, rubric_id: str) -> WeightSetRecord | None:
        return self.weight_set if rubric_id == self.weight_set.rubric_id else None

    def create_agent_run(self, run: object) -> str:
        self.runs.append(run)
        return "run-j-1"

    def update_agent_run(self, run_id: object, run: object) -> None:
        self.updates.append((run_id, run))

    def create_topic_evaluation(self, evaluation: object) -> str:
        self.evaluations.append(evaluation)
        return "evaluation-j-1"


def test_run_judge_recomputes_and_persists_score() -> None:
    topic = TopicRecord(uuid4(), "标题", "角度", "摘要", [], ["zhihu"], "candidate", None)
    item = RawItemRecord(uuid4(), uuid4(), "素材", None, None, "正文", None, "信源", 0.8, None)
    rubric = load_topic_rubric(RUBRIC_PATH.resolve())
    weights = WeightSetRecord("topic", 7, {key: 1 / 6 for key in rubric.dimension_keys})
    repository = FakeJudgeRepository(topic, item, weights)
    trace = FakeTrace()

    from scholar_agents.agents.judge import run_judge

    result = run_judge(
        topic.id,
        FakeJudgeProvider(),
        "fake-model",
        repository,
        RUBRIC_PATH.resolve(),
        recorder=trace,
    )

    assert result.total_score == pytest.approx(50.0)
    assert result.usage.input_tokens == 12
    assert result.usage.output_tokens == 34
    assert repository.evaluations[0].weight_version == 7
    assert repository.evaluations[0].vetoed_dimension is None
    assert repository.updates[-1][1].status == "succeeded"
    assert trace.scores == [("topic_total_score", 50.0)]
