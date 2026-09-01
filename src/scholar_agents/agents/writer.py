"""M2 WriterOrchestrator：共享流水线骨架，平台差异由 Profile 注入。"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field
from scholar_contracts.models import Platform, PlatformProfile, RewriteContext

from scholar_agents import telemetry
from scholar_agents.db_access import (
    AgentRunInsert,
    AgentRunStatus,
    ArticleInsert,
    ArticleRecord,
    ArticleReferenceRecord,
    InsightRecord,
    RawItemRecord,
    TopicRecord,
)
from scholar_agents.errors import PermanentJobError
from scholar_agents.observability import TraceRecorder
from scholar_agents.providers.base import ModelProvider, Usage
from scholar_agents.runtime.structured import StructuredOutputError, complete_structured
from scholar_agents.writing.formatter import FormattedArticle, WriterConstraintError, format_article
from scholar_agents.writing.profiles import profile_prompt

PROMPT_VERSION = "writer-orchestrator@v1"
MAX_MATERIAL_CHARACTERS = 20_000
MAX_REFERENCE_CHARACTERS = 4_000


class OutlineSection(BaseModel):
    model_config = ConfigDict(extra="forbid")
    heading: str = Field(min_length=1)
    purpose: str = Field(min_length=1)
    evidenceRawItemIds: list[UUID]


class WriterOutline(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1)
    sections: list[OutlineSection] = Field(min_length=3, max_length=7)


class WriterDraft(BaseModel):
    model_config = ConfigDict(extra="forbid")
    title: str = Field(min_length=1)
    contentMarkdown: str = Field(min_length=1)


class WriterRevision(WriterDraft):
    changes: list[str]


class WriterRepository(Protocol):
    def get_topic(self, topic_id: UUID) -> TopicRecord | None: ...

    def get_article(self, article_id: UUID) -> ArticleRecord | None: ...

    def get_latest_article(self, topic_id: UUID, platform: str) -> ArticleRecord | None: ...

    def list_topic_raw_items(self, topic: TopicRecord) -> list[RawItemRecord]: ...

    def list_writing_insights(
        self, platform: str, embedding: list[float] | None = None, limit: int = 5
    ) -> list[InsightRecord]: ...

    def list_high_score_articles(
        self, platform: str, limit: int = 3
    ) -> list[ArticleReferenceRecord]: ...

    def create_agent_run(self, run: AgentRunInsert) -> UUID: ...

    def update_agent_run(self, run_id: UUID, run: AgentRunInsert) -> None: ...

    def create_article(self, article: ArticleInsert) -> UUID: ...


@dataclass(frozen=True, slots=True)
class WriterModels:
    outline: str
    draft: str
    critic: str


@dataclass(frozen=True, slots=True)
class WriterProviders:
    outline: ModelProvider
    draft: ModelProvider
    critic: ModelProvider


@dataclass(frozen=True, slots=True)
class WriterResult:
    article_id: UUID
    agent_run_id: UUID
    usage: Usage
    formatted: FormattedArticle


def run_writer(
    topic_id: UUID,
    platform: Platform,
    profile: PlatformProfile,
    providers: WriterProviders,
    models: WriterModels,
    repository: WriterRepository,
    *,
    recorder: TraceRecorder,
    rewrite: RewriteContext | None = None,
    replay: bool = False,
) -> WriterResult:
    topic = repository.get_topic(topic_id)
    if topic is None:
        raise PermanentJobError(f"topic {topic_id} not found")
    if topic.status not in {"in_writing", "written"}:
        raise PermanentJobError(f"topic {topic_id} is not in writing state: {topic.status}")
    if platform.value not in topic.target_platforms:
        raise PermanentJobError(f"platform {platform.value} is not targeted by topic {topic_id}")

    previous: ArticleRecord | None = None
    version = 1
    if rewrite is not None:
        previous = repository.get_article(rewrite.previousArticleId)
        if previous is None:
            raise PermanentJobError(f"previous article {rewrite.previousArticleId} not found")
        if previous.topic_id != topic.id or previous.platform != platform.value:
            raise PermanentJobError("rewrite previous article does not match topic and platform")
        if previous.status != "rewrite_queued":
            raise PermanentJobError(
                f"previous article {previous.id} is not rewrite_queued: {previous.status}"
            )
        if previous.version >= 3:
            raise PermanentJobError("article rewrite limit already reached")
        version = previous.version + 1
    elif replay:
        # A node replay must create an isolated article version. The parent
        # article may be pending review and therefore is not a RewriteContext.
        previous = repository.get_latest_article(topic.id, platform.value)
        if previous is not None:
            version = previous.version + 1

    with telemetry.span("writer.context.build", **{"topic.id": str(topic_id)}):
        raw_items = repository.list_topic_raw_items(topic)
        context = build_writer_context(
            topic,
            raw_items,
            repository.list_writing_insights(platform.value, topic.embedding),
            repository.list_high_score_articles(platform.value),
        )
    writer_agent = (
        f"{PROMPT_VERSION};profile={profile.id}@{profile.version};"
        f"outline={models.outline};draft={models.draft};critic={models.critic}"
    )
    run_id = repository.create_agent_run(
        AgentRunInsert(
            job_type="article.write",
            entity_type="topic",
            entity_id=topic.id,
            model=writer_agent,
            prompt_version=PROMPT_VERSION,
            tokens_in=0,
            tokens_out=0,
            cost_usd=None,
            langfuse_trace_id=recorder.trace_id,
            status="running",
        )
    )
    usage = Usage()
    try:
        platform_instructions = profile_prompt(profile)
        rewrite_context = _rewrite_prompt(previous, rewrite)
        outline: WriterOutline | None = None
        if rewrite is None or rewrite.redoOutline:
            outline_data, outline_usage = complete_structured(
                providers.outline,
                models.outline,
                "你是 Outliner，只设计文章的证据链和结构，不写正文。\n\n" + platform_instructions,
                context + rewrite_context,
                outline_output_schema(raw_items),
            )
            _add_usage(usage, outline_usage)
            outline = WriterOutline.model_validate(outline_data)
            _validate_outline_evidence(outline, raw_items)

        drafting_plan = (
            f"写作大纲：\n{outline.model_dump_json(indent=2)}"
            if outline is not None
            else "保持上一版本的主体结构，只修复评审指出的问题，不重新设计大纲。"
        )
        draft_data, draft_usage = complete_structured(
            providers.draft,
            models.draft,
            "你是 Drafter，严格依据素材、大纲和平台档案写成 Markdown。\n\n" + platform_instructions,
            f"{context}{rewrite_context}\n\n{drafting_plan}",
            WriterDraft.model_json_schema(),
            max_tokens=8192,
        )
        _add_usage(usage, draft_usage)
        draft = WriterDraft.model_validate(draft_data)

        revision = _critic_revision(
            providers.critic,
            models.critic,
            platform_instructions,
            context,
            draft,
            usage,
        )
        try:
            formatted = format_article(revision.title, revision.contentMarkdown, profile)
        except WriterConstraintError as first_error:
            # Formatter 的确定性结果反馈给 SelfCritic 做一次有边界的修复，不重跑大纲和初稿。
            telemetry.formatter_violations.add(1, {"platform": platform.value, "stage": "initial"})
            revision = _critic_revision(
                providers.critic,
                models.critic,
                platform_instructions,
                context,
                revision,
                usage,
                formatter_feedback=str(first_error),
            )
            try:
                formatted = format_article(revision.title, revision.contentMarkdown, profile)
            except WriterConstraintError:
                telemetry.formatter_violations.add(
                    1, {"platform": platform.value, "stage": "repair"}
                )
                raise

        article_id = repository.create_article(
            ArticleInsert(
                topic_id=topic.id,
                platform=platform.value,
                version=version,
                title=formatted.title,
                content_md=formatted.content_md,
                writer_agent=writer_agent,
                previous_article_id=previous.id if previous is not None else None,
            )
        )
        repository.update_agent_run(
            run_id,
            _run_update(topic.id, writer_agent, recorder.trace_id, usage, "succeeded"),
        )
        return WriterResult(article_id, run_id, usage, formatted)
    except StructuredOutputError as exc:
        _add_usage(usage, exc.usage)
        repository.update_agent_run(
            run_id, _run_update(topic.id, writer_agent, recorder.trace_id, usage, "failed")
        )
        raise
    except Exception:
        repository.update_agent_run(
            run_id, _run_update(topic.id, writer_agent, recorder.trace_id, usage, "failed")
        )
        raise


def _critic_revision(
    provider: ModelProvider,
    model: str,
    platform_instructions: str,
    context: str,
    draft: WriterDraft,
    usage: Usage,
    *,
    formatter_feedback: str | None = None,
) -> WriterRevision:
    feedback = (
        f"\n\nFormatter 检测到以下硬约束问题，必须逐项修复：{formatter_feedback}"
        if formatter_feedback
        else ""
    )
    data, critic_usage = complete_structured(
        provider,
        model,
        "你是 SelfCritic。检查事实准确、结构、可读性和平台硬约束，直接返回修订稿。\n\n"
        + platform_instructions,
        f"素材上下文：\n{context}\n\n待审稿：\n{draft.model_dump_json(indent=2)}{feedback}",
        WriterRevision.model_json_schema(),
        max_tokens=8192,
    )
    _add_usage(usage, critic_usage)
    return WriterRevision.model_validate(data)


def build_writer_context(
    topic: TopicRecord,
    raw_items: list[RawItemRecord],
    insights: list[InsightRecord],
    references: list[ArticleReferenceRecord],
) -> str:
    materials: list[str] = []
    remaining = MAX_MATERIAL_CHARACTERS
    for index, item in enumerate(raw_items, start=1):
        if remaining <= 0:
            break
        content = item.content[:remaining]
        remaining -= len(content)
        materials.append(
            f"素材 {index}（rawItemId={item.id}）\n"
            f"标题：{item.title}\n来源：{item.source_name}\n"
            f"URL：{item.url or '无'}\n正文：{content}"
        )
    insight_text = (
        "\n".join(f"- {item.content}（confidence={item.confidence:.2f}）" for item in insights)
        or "无"
    )
    reference_text = (
        "\n\n".join(
            f"高分参考 {index}（{item.latest_score:.1f} 分）：{item.title}\n"
            f"{item.content_md[:MAX_REFERENCE_CHARACTERS]}"
            for index, item in enumerate(references, start=1)
        )
        or "无"
    )
    return f"""选题标题：{topic.title}
创作角度：{topic.angle}
摘要：{topic.summary}

事实素材（唯一事实来源）：
{chr(10).join(materials) or "无额外原文，只能使用选题标题、角度和摘要中的事实"}

已生效写作经验：
{insight_text}

同平台历史高分参考（只学习结构和表达，禁止复制事实）：
{reference_text}
"""


def _rewrite_prompt(previous: ArticleRecord | None, rewrite: RewriteContext | None) -> str:
    if previous is None or rewrite is None:
        return ""
    return f"""

这是第 {previous.version + 1} 版回炉任务。
上一版本标题：{previous.title}
上一版本正文：
{previous.content_md}

独立 ArticleJudge 的评审反馈：
{rewrite.evaluationFeedback}

是否重做大纲：{"是" if rewrite.redoOutline else "否"}
必须针对反馈修订，不能仅做同义改写。
"""


def outline_output_schema(raw_items: list[RawItemRecord]) -> dict[str, Any]:
    schema = deepcopy(WriterOutline.model_json_schema())
    section = schema["$defs"]["OutlineSection"]
    evidence = section["properties"]["evidenceRawItemIds"]
    allowed = [str(item.id) for item in raw_items]
    evidence["items"] = {"type": "string", "format": "uuid", "enum": allowed}
    if not allowed:
        evidence["maxItems"] = 0
    return schema


def _validate_outline_evidence(outline: WriterOutline, raw_items: list[RawItemRecord]) -> None:
    allowed = {item.id for item in raw_items}
    referenced = {
        raw_item_id for section in outline.sections for raw_item_id in section.evidenceRawItemIds
    }
    unknown = referenced - allowed
    if unknown:
        unknown_ids = sorted(map(str, unknown))
        raise ValueError(f"outline references raw items outside context: {unknown_ids}")


def _add_usage(total: Usage, increment: Usage) -> None:
    total.input_tokens += increment.input_tokens
    total.output_tokens += increment.output_tokens


def _run_update(
    topic_id: UUID,
    writer_agent: str,
    trace_id: str,
    usage: Usage,
    status: AgentRunStatus,
) -> AgentRunInsert:
    return AgentRunInsert(
        job_type="article.write",
        entity_type="topic",
        entity_id=topic_id,
        model=writer_agent,
        prompt_version=PROMPT_VERSION,
        tokens_in=usage.input_tokens,
        tokens_out=usage.output_tokens,
        cost_usd=None,
        langfuse_trace_id=trace_id,
        status=status,
    )
