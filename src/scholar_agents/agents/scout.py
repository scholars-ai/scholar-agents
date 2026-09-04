"""TopicScout 的纯函数基础能力。

这里的 embedding 聚类只负责召回潜在相关素材，不能替代 LLM 的事件判断：
相似度阈值刻意保持较低，避免把跨语言报道或不同角度的文章提前丢掉。
"""

from __future__ import annotations

from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Protocol, cast
from uuid import UUID

import structlog
from scholar_contracts.models import TopicDraft, TopicScoutOutput

from scholar_agents import telemetry
from scholar_agents.db_access import AgentRunInsert, InsightRecord, RawItemRecord
from scholar_agents.embedding import cosine
from scholar_agents.providers.base import ModelProvider, Usage
from scholar_agents.runtime.structured import StructuredOutputError, complete_structured

DEFAULT_CLUSTER_THRESHOLD = 0.80
AGENT_VERSION = "topic-scout@v1"
DEFAULT_MAX_CONCURRENCY = 6
log = structlog.get_logger()


class ScoutOutputError(ValueError):
    """模型输出无法安全映射到当前素材簇。"""


class ScoutRepository(Protocol):
    def list_topic_insights(
        self, embedding: list[float] | None = None, limit: int = 5
    ) -> list[InsightRecord]: ...

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
    failed_clusters: list[dict[str, Any]]


EmbedFn = Callable[[str], list[float]]


@dataclass(frozen=True, slots=True)
class _PreparedCluster:
    index: int
    items: list[RawItemRecord]
    system: str
    user: str
    schema: dict[str, Any]


def _run_cluster(
    prepared: _PreparedCluster, provider: ModelProvider, model: str
) -> tuple[dict[str, Any], Usage]:
    return complete_structured(
        provider,
        model,
        prepared.system,
        prepared.user,
        prepared.schema,
    )


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


def build_scout_prompt(
    items: list[RawItemRecord],
    *,
    targeted: bool = False,
    insights: list[InsightRecord] | None = None,
) -> tuple[str, str]:
    """构造 TopicScout 的系统提示和素材上下文。"""
    system = """你是 TopicScout，负责把资讯素材聚合成可创作的选题。

先判断素材是否属于同一事件：相似主题不等于同一事件，不要为了凑簇而强行合并。
跨语言、不同来源对同一产品发布或同一新闻事件的报道，仍应视为同一事件；语言差异本身不是拆分理由。
如果素材并非同一事件，输出空 topics，并在 discardReason 说明原因。
如果属于同一事件，至少输出一个可写角度，最多输出 1–3 个不同创作角度；
不要因为角度不够差异化而丢弃同一事件。
每个角度必须引用输入素材中的 rawItemIds，只能建议 xiaohongshu、zhihu、wechat 平台。
输出必须符合 TopicScoutOutput schema，字段名使用 rawItemIds 和 targetPlatforms。"""
    if targeted:
        system += """

当前是定向手动投喂模式。即使输入只有一篇素材，也必须输出 1 个可写选题；
不要因为缺少其他报道而输出空 topics。选题只能基于这篇素材，不得虚构未提供的事实。"""
    memory = "\n".join(
        f"- {item.content}（confidence={item.confidence:.2f}）" for item in insights or []
    ) or "无"
    system += f"""

以下是由真实发布数据支持的已生效经验。它们是选角度的参考，不是事实来源；
若与当前素材冲突，以当前素材为准：
{memory}"""
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
                raise ScoutOutputError(f"raw item {raw_item_id} is not in the input cluster")
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
    targeted: bool = False,
    agent_version_override: str | None = None,
    max_concurrency: int = DEFAULT_MAX_CONCURRENCY,
) -> ScoutResult:
    """运行一轮 TopicScout；调用方负责事务提交。"""
    if not items:
        return ScoutResult(
            created_topics=0, clusters_processed=0, usage=Usage(), failed_clusters=[]
        )
    if max_concurrency < 1:
        raise ScoutOutputError("max concurrency must be at least 1")

    embedder = embed_fn or _default_embed
    created_topics = 0
    clusters_processed = 0
    total_usage = Usage()
    failed_clusters: list[dict[str, Any]] = []
    agent_version = (agent_version_override or AGENT_VERSION).strip()
    if not agent_version:
        raise ScoutOutputError("agent version override must not be empty")
    with telemetry.span("embedding.cluster"):
        clusters = cluster_raw_items(items)
    try:
        prepared_clusters: list[_PreparedCluster] = []
        for index, cluster in enumerate(clusters):
            if max_topics is not None and created_topics >= max_topics:
                break
            insight_embedding = _cluster_embedding(cluster)
            insights = repository.list_topic_insights(insight_embedding)
            system, user = build_scout_prompt(cluster, targeted=targeted, insights=insights)
            prepared_clusters.append(
                _PreparedCluster(
                    index=index,
                    items=cluster,
                    system=system,
                    user=user,
                    schema=scout_output_schema(cluster),
                )
            )

        # maxTopics preserves the historical short-circuit semantics. Scheduled
        # runs have no cap and can safely overlap provider calls.
        if max_topics is None and max_concurrency > 1:
            with ThreadPoolExecutor(max_workers=max_concurrency) as executor:
                cluster_results = [
                    (prepared, executor.submit(_run_cluster, prepared, provider, model))
                    for prepared in prepared_clusters
                ]
        else:
            cluster_results = [
                (prepared, None) for prepared in prepared_clusters
            ]

        for prepared, future in cluster_results:
            if max_topics is not None and created_topics >= max_topics:
                break
            cluster = prepared.items
            clusters_processed += 1
            try:
                data, usage = (
                    future.result()
                    if future is not None
                    else _run_cluster(prepared, provider, model)
                )
                drafts = parse_scout_output(data, cluster)
            except (StructuredOutputError, ScoutOutputError) as exc:
                usage = exc.usage if isinstance(exc, StructuredOutputError) else Usage()
                failed_clusters.append(
                    {
                        "cluster_index": prepared.index,
                        "raw_item_ids": [str(item.id) for item in cluster],
                        "reason": str(exc)[:2000],
                        "error_type": type(exc).__name__,
                    }
                )
                log.warning(
                    "topic_scout_cluster_failed",
                    cluster_index=prepared.index,
                    raw_item_count=len(cluster),
                    error_type=type(exc).__name__,
                    error=str(exc),
                )
                total_usage.input_tokens += usage.input_tokens
                total_usage.output_tokens += usage.output_tokens
                continue
            total_usage.input_tokens += usage.input_tokens
            total_usage.output_tokens += usage.output_tokens
            for draft in drafts:
                if max_topics is not None and created_topics >= max_topics:
                    break
                with telemetry.span("topic.embedding"):
                    topic_embedding = embedder(f"{draft.title}\n\n{draft.angle}\n\n{draft.summary}")
                with telemetry.span("topic.duplicate_check"):
                    duplicate = repository.find_similar_topic(topic_embedding)
                if duplicate is not None:
                    continue
                with telemetry.span("topic.insert"):
                    repository.create_topic(
                        title=draft.title,
                        angle=draft.angle,
                        summary=draft.summary,
                        raw_item_ids=draft.rawItemIds,
                        target_platforms=[platform.value for platform in draft.targetPlatforms],
                        embedding=topic_embedding,
                    )
                created_topics += 1
            with telemetry.span("raw_items.mark_clustered"):
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
                agent_version=agent_version,
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
                agent_version=agent_version,
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
                agent_version=agent_version,
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
        clusters_processed=clusters_processed,
        usage=total_usage,
        failed_clusters=failed_clusters,
    )


def _default_embed(text: str) -> list[float]:
    from scholar_agents.embedding import embed

    return embed(text)


def _cluster_embedding(items: list[RawItemRecord]) -> list[float] | None:
    vectors = [item.embedding for item in items if item.embedding]
    if not vectors:
        return None
    size = len(vectors[0])
    compatible = [vector for vector in vectors if len(vector) == size]
    if not compatible:
        return None
    return [sum(vector[index] for vector in compatible) / len(compatible) for index in range(size)]
