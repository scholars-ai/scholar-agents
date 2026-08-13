from __future__ import annotations

import json
from uuid import uuid4

import pytest

from scholar_agents.agents.scout import (
    ScoutOutputError,
    build_scout_prompt,
    cluster_raw_items,
    format_cluster_context,
    parse_scout_output,
    run_scout,
    scout_output_schema,
)
from scholar_agents.db_access import RawItemRecord
from scholar_agents.providers.base import ChatResponse, TextBlock, Usage


def _item(title: str, embedding: list[float]) -> RawItemRecord:
    return RawItemRecord(
        id=uuid4(),
        source_id=uuid4(),
        title=title,
        url=f"https://example.com/{title}",
        author=None,
        content=f"{title} 的正文内容。",
        published_at=None,
        source_name="Example",
        source_weight=0.8,
        embedding=embedding,
    )


def test_cluster_raw_items_groups_high_similarity_candidates() -> None:
    items = [
        _item("同一事件 A", [1.0, 0.0]),
        _item("同一事件 B", [0.99, 0.1]),
        _item("另一个事件", [0.0, 1.0]),
    ]

    clusters = cluster_raw_items(items, threshold=0.9)

    assert [[item.title for item in cluster] for cluster in clusters] == [
        ["同一事件 A", "同一事件 B"],
        ["另一个事件"],
    ]


def test_cluster_raw_items_keeps_items_without_embeddings_separate() -> None:
    item = _item("无向量素材", [1.0, 0.0])
    item_without_embedding = RawItemRecord(
        id=uuid4(),
        source_id=uuid4(),
        title="缺少向量",
        url=None,
        author=None,
        content="正文",
        published_at=None,
        source_name="Example",
        source_weight=0.5,
        embedding=None,
    )

    clusters = cluster_raw_items([item, item_without_embedding])

    assert [[entry.title for entry in cluster] for cluster in clusters] == [
        ["无向量素材"],
        ["缺少向量"],
    ]


def test_format_cluster_context_contains_source_and_content() -> None:
    items = [_item("模型发布", [1.0, 0.0])]

    context = format_cluster_context(items)

    assert "模型发布" in context
    assert "Example" in context
    assert "正文内容" in context


def test_build_scout_prompt_requires_event_judgment_and_limits_drafts() -> None:
    items = [_item("模型发布", [1.0, 0.0])]

    system, user = build_scout_prompt(items)

    assert "同一事件" in system
    assert "1–3" in system
    assert "模型发布" in user
    assert "rawItemIds" in system


def test_build_scout_prompt_treats_cross_language_reports_as_one_event() -> None:
    items = [
        _item("NVIDIA VoiceChat English", [1.0, 0.0]),
        _item("NVIDIA VoiceChat 中文", [0.99, 0.1]),
    ]

    system, user = build_scout_prompt(items)

    assert "跨语言" in system
    assert str(items[0].id) in user
    assert str(items[1].id) in user


def test_build_scout_prompt_requires_one_angle_for_a_coherent_event() -> None:
    items = [_item("同一事件 A", [1.0, 0.0]), _item("同一事件 B", [0.99, 0.1])]

    system, _ = build_scout_prompt(items)

    assert "至少输出一个可写角度" in system
    assert "不要因为角度不够差异化而丢弃" in system


def test_scout_output_schema_limits_raw_item_ids_to_input_cluster() -> None:
    items = [_item("模型发布 A", [1.0, 0.0]), _item("模型发布 B", [0.99, 0.1])]

    schema = scout_output_schema(items)

    raw_item_schema = schema["$defs"]["TopicDraft"]["properties"]["rawItemIds"]
    assert raw_item_schema["items"]["enum"] == [str(item.id) for item in items]


def test_parse_scout_output_rejects_raw_item_id_outside_cluster() -> None:
    item = _item("模型发布", [1.0, 0.0])
    other_id = uuid4()

    with pytest.raises(ScoutOutputError, match="not in the input cluster"):
        parse_scout_output(
            {
                "topics": [
                    {
                        "title": "选题",
                        "angle": "角度",
                        "summary": "摘要",
                        "rawItemIds": [str(other_id)],
                        "targetPlatforms": ["zhihu"],
                    }
                ]
            },
            [item],
        )


def test_parse_scout_output_deduplicates_raw_ids_and_keeps_valid_drafts() -> None:
    item = _item("模型发布", [1.0, 0.0])

    drafts = parse_scout_output(
        {
            "topics": [
                {
                    "title": "选题",
                    "angle": "角度",
                    "summary": "摘要",
                    "rawItemIds": [str(item.id), str(item.id)],
                    "targetPlatforms": ["zhihu"],
                }
            ]
        },
        [item],
    )

    assert len(drafts) == 1
    assert drafts[0].rawItemIds == [item.id]


class FakeProvider:
    name = "fake"

    def __init__(self, outputs: list[str]) -> None:
        self.outputs = list(outputs)
        self.calls = 0

    def complete(self, model: str, request: object) -> ChatResponse:
        del model, request
        self.calls += 1
        return ChatResponse(
            content=[TextBlock(text=self.outputs.pop(0))],
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=20),
            model="fake-model",
        )


class FakeRepository:
    def __init__(self) -> None:
        self.created_topics: list[dict[str, object]] = []
        self.clustered: list[object] = []
        self.runs: list[object] = []

    def find_similar_topic(self, embedding: list[float], threshold: float = 0.92) -> object:
        del embedding, threshold
        return None

    def create_topic(self, **kwargs: object) -> object:
        self.created_topics.append(kwargs)
        return uuid4()

    def mark_raw_items_clustered(self, raw_item_ids: list[object]) -> None:
        self.clustered.extend(raw_item_ids)

    def create_agent_run(self, run: object) -> object:
        self.runs.append(run)
        return uuid4()


def test_run_scout_writes_draft_and_records_usage() -> None:
    item = _item("模型发布", [1.0, 0.0])
    provider = FakeProvider(
        [
            json.dumps(
                {
                    "topics": [
                        {
                            "title": "模型发布意味着什么",
                            "angle": "工程实践",
                            "summary": "摘要",
                            "rawItemIds": [str(item.id)],
                            "targetPlatforms": ["zhihu"],
                        }
                    ]
                }
            )
        ]
    )
    repository = FakeRepository()

    result = run_scout(
        [item],
        provider,
        "fake-model",
        repository,
        embed_fn=lambda text: [1.0, 0.0],
    )

    assert result.created_topics == 1
    assert result.usage.input_tokens == 10
    assert result.usage.output_tokens == 20
    assert len(repository.created_topics) == 1
    assert repository.clustered == [item.id]
    assert len(repository.runs) == 1


def test_run_scout_does_not_write_duplicate_topic() -> None:
    item = _item("模型发布", [1.0, 0.0])
    provider = FakeProvider(
        [
            json.dumps(
                {
                    "topics": [
                        {
                            "title": "重复选题",
                            "angle": "角度",
                            "summary": "摘要",
                            "rawItemIds": [str(item.id)],
                            "targetPlatforms": ["zhihu"],
                        }
                    ]
                }
            )
        ]
    )
    repository = FakeRepository()
    repository.find_similar_topic = lambda embedding, threshold=0.92: (uuid4(), 0.95)  # type: ignore[method-assign]

    result = run_scout(
        [item],
        provider,
        "fake-model",
        repository,
        embed_fn=lambda text: [1.0, 0.0],
    )

    assert result.created_topics == 0
    assert repository.created_topics == []
    assert repository.clustered == [item.id]


def test_run_scout_stops_processing_clusters_at_max_topics() -> None:
    first = _item("第一个事件", [1.0, 0.0])
    second = _item("第二个事件", [0.0, 1.0])
    provider = FakeProvider(
        [
            json.dumps(
                {
                    "topics": [
                        {
                            "title": "第一个选题",
                            "angle": "角度",
                            "summary": "摘要",
                            "rawItemIds": [str(first.id)],
                            "targetPlatforms": ["zhihu"],
                        }
                    ]
                }
            )
        ]
    )
    repository = FakeRepository()

    result = run_scout(
        [first, second],
        provider,
        "fake-model",
        repository,
        max_topics=1,
        embed_fn=lambda text: [1.0, 0.0],
    )

    assert result.created_topics == 1
    assert result.clusters_processed == 1
    assert provider.calls == 1
    assert repository.clustered == [first.id]
