from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

import pytest
from scholar_contracts.models import Platform

from scholar_agents.agents.writer import WriterModels, WriterProviders, run_writer
from scholar_agents.db_access import AgentRunInsert, ArticleInsert, TopicRecord
from scholar_agents.observability import TraceRecorder
from scholar_agents.providers.base import ChatResponse, TextBlock, Usage
from scholar_agents.writing.formatter import WriterConstraintError, format_article
from scholar_agents.writing.profiles import load_platform_profile

PROFILES = Path(__file__).resolve().parents[2] / "scholar-shared" / "profiles"


class FakeProvider:
    name = "fake"

    def __init__(self, payload: dict[str, object]) -> None:
        self.payload = payload
        self.calls = 0

    def complete(self, _model: str, _req: object) -> ChatResponse:
        self.calls += 1
        return ChatResponse(
            content=[TextBlock(text=json.dumps(self.payload, ensure_ascii=False))],
            stop_reason="end_turn",
            usage=Usage(input_tokens=10, output_tokens=20),
            model="fake-model",
        )


class FakeRepository:
    def __init__(self, topic: TopicRecord) -> None:
        self.topic = topic
        self.run_id = uuid4()
        self.article_id = uuid4()
        self.article: ArticleInsert | None = None
        self.run_updates: list[AgentRunInsert] = []

    def get_topic(self, _topic_id: UUID) -> TopicRecord:
        return self.topic

    def list_topic_raw_items(self, _topic: TopicRecord) -> list[object]:
        return []

    def list_writing_insights(self, _platform: str, limit: int = 5) -> list[object]:
        return []

    def list_high_score_articles(self, _platform: str, limit: int = 3) -> list[object]:
        return []

    def create_agent_run(self, _run: AgentRunInsert) -> UUID:
        return self.run_id

    def update_agent_run(self, _run_id: UUID, run: AgentRunInsert) -> None:
        self.run_updates.append(run)

    def create_article(self, article: ArticleInsert) -> UUID:
        self.article = article
        return self.article_id


@pytest.mark.parametrize("platform", list(Platform))
def test_loads_every_declared_platform_profile(platform: Platform) -> None:
    profile = load_platform_profile(PROFILES, platform)

    assert profile.platform == platform
    assert profile.rubricRef == f"article/{platform.value}@v1"


def test_formatter_enforces_xiaohongshu_hard_constraints() -> None:
    profile = load_platform_profile(PROFILES, Platform.xiaohongshu)

    with pytest.raises(WriterConstraintError, match="低于下限"):
        format_article("短标题", "内容太短\n#标签一 #标签二 #标签三", profile)

    content = "这是基于给定素材整理的可执行工程说明。" * 45 + "\n#人工智能 #工程实践 #可观测性"
    formatted = format_article("工程链路为什么更可靠", content, profile)

    assert 600 <= formatted.character_count <= 1200
    assert formatted.tag_count == 3


def test_writer_runs_outline_draft_critic_and_persists_draft() -> None:
    topic_id = uuid4()
    topic = TopicRecord(
        id=topic_id,
        title="可观测 AI 流水线",
        angle="解释端到端追踪如何降低演进风险",
        summary="从状态机、队列和 Trace 的关系展开。",
        raw_item_ids=[],
        target_platforms=["xiaohongshu"],
        status="in_writing",
        latest_score=82.0,
    )
    outline = FakeProvider(
        {
            "title": "工程链路为什么更可靠",
            "sections": [
                {"heading": "先看问题", "purpose": "建立背景", "evidenceRawItemIds": []},
                {"heading": "拆开链路", "purpose": "解释机制", "evidenceRawItemIds": []},
                {"heading": "给出行动", "purpose": "总结建议", "evidenceRawItemIds": []},
            ],
        }
    )
    content = "这是基于给定素材整理的可执行工程说明。" * 45 + "\n#人工智能 #工程实践 #可观测性"
    draft = FakeProvider({"title": "工程链路为什么更可靠", "contentMarkdown": content})
    critic = FakeProvider(
        {
            "title": "工程链路为什么更可靠",
            "contentMarkdown": content,
            "changes": ["核对事实边界", "保留三个行动标签"],
        }
    )
    repository = FakeRepository(topic)

    result = run_writer(
        topic_id,
        Platform.xiaohongshu,
        load_platform_profile(PROFILES, Platform.xiaohongshu),
        WriterProviders(outline=outline, draft=draft, critic=critic),
        WriterModels(outline="outline-model", draft="draft-model", critic="critic-model"),
        repository,  # type: ignore[arg-type]
        recorder=TraceRecorder(name="writer-test"),
    )

    assert result.article_id == repository.article_id
    assert result.usage == Usage(input_tokens=30, output_tokens=60)
    assert repository.article is not None
    assert repository.article.platform == "xiaohongshu"
    assert repository.article.version == 1
    assert repository.run_updates[-1].status == "succeeded"
    assert outline.calls == draft.calls == critic.calls == 1
