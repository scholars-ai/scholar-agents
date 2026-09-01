from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest

from scholar_agents.agents.article_judge import (
    PROMPT_VERSION,
    ArticleJudgeError,
    load_article_rubric,
    recompute_article_score,
    run_article_judge,
)
from scholar_agents.db_access import (
    AgentRunInsert,
    ArticleEvaluationInsert,
    ArticleRecord,
    RawItemRecord,
    TopicRecord,
    WeightSetRecord,
)
from scholar_agents.providers.base import ChatResponse, TextBlock, Usage

RUBRICS = Path(__file__).resolve().parents[2] / "scholar-shared" / "rubrics"


class FakeArticleJudgeProvider:
    name = "fake"

    def __init__(self, scores: dict[str, float]) -> None:
        self.scores = scores

    def complete(self, _model: str, _request: object) -> ChatResponse:
        return ChatResponse(
            content=[
                TextBlock(
                    text=json.dumps(
                        {
                            "dimensionScores": {
                                key: {"score": score, "reason": f"{key} reason"}
                                for key, score in self.scores.items()
                            },
                            "rationale": "独立评分理由",
                        }
                    )
                )
            ],
            stop_reason="end_turn",
            usage=Usage(input_tokens=21, output_tokens=34),
            model="fake-model",
        )


class FakeTrace:
    trace_id = "trace-article-judge"

    def __init__(self) -> None:
        self.scores: list[tuple[str, float, str | None]] = []

    def score(self, *, name: str, value: float, comment: str | None = None) -> None:
        self.scores.append((name, value, comment))


class FakeArticleJudgeRepository:
    def __init__(
        self,
        article: ArticleRecord,
        topic: TopicRecord,
        item: RawItemRecord,
        weights: WeightSetRecord,
    ) -> None:
        self.article = article
        self.topic = topic
        self.item = item
        self.weights = weights
        self.run_id = uuid4()
        self.evaluation_id = uuid4()
        self.runs: list[AgentRunInsert] = []
        self.run_updates: list[AgentRunInsert] = []
        self.evaluations: list[ArticleEvaluationInsert] = []

    def get_article(self, article_id: UUID) -> ArticleRecord | None:
        return self.article if article_id == self.article.id else None

    def get_topic(self, topic_id: UUID) -> TopicRecord | None:
        return self.topic if topic_id == self.topic.id else None

    def list_topic_raw_items(self, _topic: TopicRecord) -> list[RawItemRecord]:
        return [self.item]

    def get_active_weight_set(self, rubric_id: str) -> WeightSetRecord | None:
        return self.weights if rubric_id == self.weights.rubric_id else None

    def create_agent_run(self, run: AgentRunInsert) -> UUID:
        self.runs.append(run)
        return self.run_id

    def update_agent_run(self, _run_id: UUID, run: AgentRunInsert) -> None:
        self.run_updates.append(run)

    def create_article_evaluation(self, evaluation: ArticleEvaluationInsert) -> UUID:
        self.evaluations.append(evaluation)
        return self.evaluation_id


def _fixtures(platform: str = "xiaohongshu") -> tuple[ArticleRecord, TopicRecord, RawItemRecord]:
    topic = TopicRecord(
        uuid4(), "选题", "工程视角", "摘要", [], [platform], "written", 82.0
    )
    article = ArticleRecord(
        uuid4(), topic.id, platform, 1, "文章标题", "文章正文", "writer", "draft", None, None
    )
    item = RawItemRecord(
        uuid4(), uuid4(), "素材", "https://example.com", None, "素材正文", None, "信源", 0.8, None
    )
    return article, topic, item


@pytest.mark.parametrize("platform", ["xiaohongshu", "zhihu", "wechat"])
def test_loads_platform_rubric(platform: str) -> None:
    rubric = load_article_rubric(RUBRICS / f"article-{platform}.v1.yaml")

    assert rubric.rubric_id == f"article/{platform}"
    assert rubric.version == f"article/{platform}@v1"
    assert rubric.pass_threshold == 70
    assert "accuracy" in rubric.dimension_keys


def test_recompute_article_score_applies_accuracy_veto() -> None:
    rubric = load_article_rubric(RUBRICS / "article-xiaohongshu.v1.yaml")
    scores = {key: 9.0 for key in rubric.dimension_keys}
    scores["accuracy"] = 5.9
    weights = WeightSetRecord(
        rubric.rubric_id,
        2,
        {dimension.key: dimension.initial_weight for dimension in rubric.dimensions},
    )

    result = recompute_article_score(rubric, weights, scores)

    assert result.total_score > rubric.pass_threshold
    assert result.vetoed_dimension == "accuracy"
    assert not result.passed


def test_run_article_judge_recomputes_and_persists_deterministic_decision() -> None:
    article, topic, item = _fixtures()
    rubric = load_article_rubric(RUBRICS / "article-xiaohongshu.v1.yaml")
    weights = WeightSetRecord(
        rubric.rubric_id,
        3,
        {dimension.key: dimension.initial_weight for dimension in rubric.dimensions},
    )
    scores = {key: 8.0 for key in rubric.dimension_keys}
    repository = FakeArticleJudgeRepository(article, topic, item, weights)
    trace = FakeTrace()

    result = run_article_judge(
        article.id,
        FakeArticleJudgeProvider(scores),
        "fake-model",
        repository,
        RUBRICS / "article-xiaohongshu.v1.yaml",
        recorder=trace,  # type: ignore[arg-type]
    )

    assert result.evaluation_id == repository.evaluation_id
    assert result.score.total_score == pytest.approx(80.0)
    assert result.score.passed
    assert result.usage == Usage(input_tokens=21, output_tokens=34)
    evaluation = repository.evaluations[0]
    assert evaluation.weight_version == 3
    assert evaluation.pass_threshold == 70
    assert evaluation.passed
    assert set(evaluation.dimension_reasons) == set(rubric.dimension_keys)
    assert repository.runs[0].prompt_version == PROMPT_VERSION
    assert repository.run_updates[-1].status == "succeeded"
    assert trace.scores == [("article_total_score", 80.0, "独立评分理由")]


def test_run_article_judge_allows_pending_review_for_replay() -> None:
    article, topic, item = _fixtures()
    article = ArticleRecord(
        article.id,
        article.topic_id,
        article.platform,
        article.version,
        article.title,
        article.content_md,
        article.writer_agent,
        "pending_review",
        article.latest_score,
        article.previous_article_id,
    )
    rubric = load_article_rubric(RUBRICS / "article-xiaohongshu.v1.yaml")
    weights = WeightSetRecord(
        rubric.rubric_id,
        1,
        {dimension.key: dimension.initial_weight for dimension in rubric.dimensions},
    )
    repository = FakeArticleJudgeRepository(article, topic, item, weights)

    result = run_article_judge(
        article.id,
        FakeArticleJudgeProvider({key: 8.0 for key in rubric.dimension_keys}),
        "fake-model",
        repository,
        RUBRICS / "article-xiaohongshu.v1.yaml",
        recorder=FakeTrace(),  # type: ignore[arg-type]
    )

    assert result.score.passed
    assert len(repository.evaluations) == 1


def test_article_judge_rejects_platform_rubric_mismatch() -> None:
    article, topic, item = _fixtures("zhihu")
    rubric = load_article_rubric(RUBRICS / "article-xiaohongshu.v1.yaml")
    weights = WeightSetRecord(
        rubric.rubric_id,
        1,
        {dimension.key: dimension.initial_weight for dimension in rubric.dimensions},
    )
    repository = FakeArticleJudgeRepository(article, topic, item, weights)

    with pytest.raises(ArticleJudgeError, match="platform does not match"):
        run_article_judge(
            article.id,
            FakeArticleJudgeProvider({key: 8.0 for key in rubric.dimension_keys}),
            "fake-model",
            repository,
            RUBRICS / "article-xiaohongshu.v1.yaml",
            recorder=FakeTrace(),  # type: ignore[arg-type]
        )
