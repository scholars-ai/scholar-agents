"""TopicScout 的纯函数基础能力。

这里的 embedding 聚类只负责召回潜在相关素材，不能替代 LLM 的事件判断：
相似度阈值刻意保持较低，避免把跨语言报道或不同角度的文章提前丢掉。
"""

from __future__ import annotations

from collections.abc import Callable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, cast
from uuid import UUID

from scholar_contracts.models import TopicDraft, TopicScoutOutput

from scholar_agents.db_access import AgentRunInsert, RawItemRecord
from scholar_agents.embedding import cosine
from scholar_agents.providers.base import ModelProvider, Usage
from scholar_agents.runtime.structured import StructuredOutputError, complete_structured

DEFAULT_CLUSTER_THRESHOLD = 0.80


class ScoutOutputError(ValueError):
    """模型输出无法安全映射到当前素材簇。"""


class ScoutRepository(Protocol):
    def find_similar_topic(
        self, embedding: list[float], threshold: float = 0.92
    ) -> tuple[UUID, float] | None: ...

    def create_topic(
        self,
        *,
        title: str,
        angle: str,
        summary: str,
        raw_item_ids: list[UUID],
        target_platforms: list[str],
        embedding: list[float],
    ) -> UUID: ...

    def mark_raw_items_clustered(self, raw_item_ids: list[UUID]) -> None: ...

    def create_agent_run(self, run: AgentRunInsert) -> UUID: ...


@dataclass(frozen=True, slots=True)
class ScoutResult:
    created_topics: int
    clusters_processed: int
    usage: Usage


EmbedFn = Callable[[str], list[float]]


def cluster_raw_items(
    items: list[RawItemRecord],
    threshold: float = DEFAULT_CLUSTER_THRESHOLD,
) -> list[list[RawItemRecord]]:
    """按 embedding 做高召回粗聚类，返回稳定顺序的候选簇。

    使用代表向量的贪心聚类：新素材只需与簇中任一素材相似即可加入。
    没有 embedding 的素材无法安全判断，单独形成候选簇交给后续处理。
    """
    if not 0 < threshold <= 1:
        raise ValueError("threshold must be in (0, 1]")

    clusters: list[list[RawItemRecord]] = []
    for item in items:
        if item.embedding is None:
            clusters.append([item])
            continue

        matching_cluster: list[RawItemRecord] | None = None
        for cluster in clusters:
            if any(
                candidate.embedding is not None
                and cosine(item.embedding, candidate.embedding) >= threshold
                for candidate in cluster
            ):
                matching_cluster = cluster
                break
        if matching_cluster is None:
            clusters.append([item])
        else:
            matching_cluster.append(item)
    return clusters


def format_cluster_context(items: list[RawItemRecord]) -> str:
    """把素材簇格式化为可注入 Scout prompt 的上下文。"""
    sections: list[str] = []
    for index, item in enumerate(items, start=1):
        published = item.published_at.isoformat() if item.published_at else "未知"
        sections.append(
            "\n".join(
                [
                    f"素材 {index}",
                    f"标题：{item.title}",
                    f"来源：{item.source_name}（权重 {item.source_weight:.2f}）",
                    f"发布时间：{published}",
                    f"URL：{item.url or '无'}",
                    f"正文：{item.content}",
                ]
            )
        )
    return "\n\n---\n\n".join(sections)


def build_scout_prompt(items: list[RawItemRecord]) -> tuple[str, str]:
    """构造 TopicScout 的系统提示和素材上下文。"""
    system = """你是 TopicScout，负责把资讯素材聚合成可创作的选题。

先判断素材是否属于同一事件：相似主题不等于同一事件，不要为了凑簇而强行合并。
如果素材并非同一事件，输出空 topics，并在 discardReason 说明原因。
如果属于同一事件，最多输出 1–3 个不同创作角度。
每个角度必须引用输入素材中的 rawItemIds，只能建议 xiaohongshu、zhihu、wechat 平台。
输出必须符合 TopicScoutOutput schema，字段名使用 rawItemIds 和 targetPlatforms。"""
    allowed_ids = "、".join(str(item.id) for item in items)
    user = (
        f"请分析以下素材簇。rawItemIds 只能从这个列表中选择：{allowed_ids}\n\n"
        f"{format_cluster_context(items)}"
    )
    return system, user


def scout_output_schema(items: list[RawItemRecord]) -> dict[str, Any]:
    """把当前素材簇的 ID 集合约束注入结构化输出 schema。"""
    schema = deepcopy(TopicScoutOutput.model_json_schema())
    raw_item_ids = schema["$defs"]["TopicDraft"]["properties"]["rawItemIds"]
    raw_item_ids["items"] = {
        "type": "string",
        "format": "uuid",
        "enum": [str(item.id) for item in items],
    }
    return cast(dict[str, Any], schema)


def parse_scout_output(data: dict[str, Any], items: list[RawItemRecord]) -> list[TopicDraft]:
    """校验结构化输出，并拒绝模型引用簇外素材。"""
    output = TopicScoutOutput.model_validate(data)
    allowed_ids = {item.id for item in items}
    drafts: list[TopicDraft] = []
    for draft in output.topics:
        unique_ids: list[UUID] = []
        for raw_item_id in draft.rawItemIds:
            if raw_item_id not in allowed_ids:
                raise ScoutOutputError(
                    f"raw item {raw_item_id} is not in the input cluster"
                )
            if raw_item_id not in unique_ids:
                unique_ids.append(raw_item_id)
        drafts.append(draft.model_copy(update={"rawItemIds": unique_ids}))
    return drafts


def run_scout(
    items: list[RawItemRecord],
    provider: ModelProvider,
    model: str,
    repository: ScoutRepository,
    *,
    max_topics: int | None = None,
    embed_fn: EmbedFn | None = None,
    langfuse_trace_id: str | None = None,
) -> ScoutResult:
    """运行一轮 TopicScout；调用方负责事务提交。"""
    if not items:
        return ScoutResult(created_topics=0, clusters_processed=0, usage=Usage())

    embedder = embed_fn or _default_embed
    created_topics = 0
    total_usage = Usage()
    clusters = cluster_raw_items(items)
    try:
        for cluster in clusters:
            system, user = build_scout_prompt(cluster)
            data, usage = complete_structured(
                provider,
                model,
                system,
                user,
                scout_output_schema(cluster),
            )
            total_usage.input_tokens += usage.input_tokens
            total_usage.output_tokens += usage.output_tokens
            drafts = parse_scout_output(data, cluster)
            for draft in drafts:
                if max_topics is not None and created_topics >= max_topics:
                    break
                topic_embedding = embedder(f"{draft.title}\n\n{draft.angle}\n\n{draft.summary}")
                if repository.find_similar_topic(topic_embedding) is not None:
                    continue
                repository.create_topic(
                    title=draft.title,
                    angle=draft.angle,
                    summary=draft.summary,
                    raw_item_ids=draft.rawItemIds,
                    target_platforms=[platform.value for platform in draft.targetPlatforms],
                    embedding=topic_embedding,
                )
                created_topics += 1
            repository.mark_raw_items_clustered([item.id for item in cluster])
    except StructuredOutputError as exc:
        total_usage.input_tokens += exc.usage.input_tokens
        total_usage.output_tokens += exc.usage.output_tokens
        repository.create_agent_run(
            AgentRunInsert(
                job_type="topic.scout",
                entity_type="raw_item_cluster",
                entity_id=None,
                model=model,
                prompt_version="topic-scout@v1",
                tokens_in=total_usage.input_tokens,
                tokens_out=total_usage.output_tokens,
                cost_usd=None,
                langfuse_trace_id=langfuse_trace_id,
                status="failed",
            )
        )
        raise
    except Exception:
        repository.create_agent_run(
            AgentRunInsert(
                job_type="topic.scout",
                entity_type="raw_item_cluster",
                entity_id=None,
                model=model,
                prompt_version="topic-scout@v1",
                tokens_in=total_usage.input_tokens,
                tokens_out=total_usage.output_tokens,
                cost_usd=None,
                langfuse_trace_id=langfuse_trace_id,
                status="failed",
            )
        )
        raise
    else:
        repository.create_agent_run(
            AgentRunInsert(
                job_type="topic.scout",
                entity_type="raw_item_cluster",
                entity_id=None,
                model=model,
                prompt_version="topic-scout@v1",
                tokens_in=total_usage.input_tokens,
                tokens_out=total_usage.output_tokens,
                cost_usd=None,
                langfuse_trace_id=langfuse_trace_id,
                status="succeeded",
            )
        )
    return ScoutResult(
        created_topics=created_topics,
        clusters_processed=len(clusters),
        usage=total_usage,
    )


def _default_embed(text: str) -> list[float]:
    from scholar_agents.embedding import embed

    return embed(text)
