from __future__ import annotations

from datetime import UTC, datetime
from uuid import uuid4

from scholar_agents.db_access import (
    AgentRunInsert,
    RawItemRecord,
    TopicEvaluationInsert,
    TopicRecord,
    WeightSetRecord,
)


def test_raw_item_record_keeps_source_context_for_scout() -> None:
    raw_item_id = uuid4()
    source_id = uuid4()
    published_at = datetime(2026, 8, 13, 8, 30, tzinfo=UTC)

    item = RawItemRecord.from_row(
        {
            "id": raw_item_id,
            "source_id": source_id,
            "title": "新模型发布",
            "url": "https://example.com/item",
            "author": "作者",
            "content": "正文",
            "published_at": published_at,
            "source_name": "Example",
            "source_weight": 0.8,
            "embedding": [1.0, 0.0],
        }
    )

    assert item.id == raw_item_id
    assert item.source_id == source_id
    assert item.source_name == "Example"
    assert item.source_weight == 0.8
    assert item.published_at == published_at
    assert item.embedding == [1.0, 0.0]


def test_scout_can_limit_raw_items_to_targeted_ids() -> None:
    from scholar_agents.db_access import AgentRepository

    class Cursor:
        def __init__(self) -> None:
            self.query = ""
            self.params: tuple[object, ...] = ()

        def __enter__(self) -> Cursor:
            return self

        def __exit__(self, *_args: object) -> None:
            return None

        def execute(self, query: str, params: tuple[object, ...] = ()) -> None:
            self.query = query
            self.params = params

        def fetchall(self) -> list[dict[str, object]]:
            return []

    class Connection:
        def __init__(self) -> None:
            self.cursor_instance = Cursor()

        def cursor(self) -> Cursor:
            return self.cursor_instance

    connection = Connection()
    repository = AgentRepository(connection)  # type: ignore[arg-type]
    repository.list_new_raw_items(raw_item_ids=[uuid4()])

    assert "r.id = any(%s)" in connection.cursor_instance.query
    assert len(connection.cursor_instance.params) == 2


def test_topic_record_builds_judge_context_from_joined_row() -> None:
    topic_id = uuid4()
    raw_item_id = uuid4()
    topic = TopicRecord.from_row(
        {
            "id": topic_id,
            "title": "选题标题",
            "angle": "工程实践角度",
            "summary": "选题摘要",
            "raw_item_ids": [raw_item_id],
            "target_platforms": ["zhihu"],
            "status": "candidate",
            "latest_score": None,
        }
    )

    assert topic.id == topic_id
    assert topic.raw_item_ids == [raw_item_id]
    assert topic.target_platforms == ["zhihu"]
    assert topic.status == "candidate"


def test_weight_set_record_parses_json_weights() -> None:
    record = WeightSetRecord.from_row(
        {"version": 3, "weights": {"timeliness": 0.2}, "rubric_id": "topic"}
    )

    assert record.rubric_id == "topic"
    assert record.version == 3
    assert record.weights == {"timeliness": 0.2}


def test_result_insert_models_preserve_replay_metadata() -> None:
    run_id = uuid4()
    topic_id = uuid4()
    evaluation = TopicEvaluationInsert(
        topic_id=topic_id,
        rubric_version="topic@v1",
        dimension_scores={"timeliness": 8.0},
        total_score=80.0,
        rationale="理由",
        judge_model="claude-sonnet-5",
        agent_run_id=run_id,
        weight_version=3,
        vetoed_dimension=None,
    )
    run = AgentRunInsert(
        job_type="topic.evaluate",
        entity_type="topic",
        entity_id=topic_id,
        model="claude-sonnet-5",
        prompt_version="topic-judge@v1",
        tokens_in=100,
        tokens_out=30,
        cost_usd=None,
        langfuse_trace_id="trace-123",
        status="succeeded",
    )

    assert evaluation.agent_run_id == run_id
    assert evaluation.weight_version == 3
    assert run.entity_id == topic_id
    assert run.tokens_in == 100
    assert run.langfuse_trace_id == "trace-123"
