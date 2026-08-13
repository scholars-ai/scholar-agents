"""source_fetch 的 job handler（SPEC-003 §3）。

流程：拉 feed → 逐条清洗 → 精确去重（content_hash）→ embedding → 语义去重
（近 14 天 cos > 0.92）→ 写入 raw_items(status=new)。

两条纪律：
1. **单源失败隔离**：单条 entry 出错只跳过该条并计数，不让整个 job 失败；
   整个 feed 拉不动才抛异常（交由 worker 重试）。
2. **只写 raw_items**：不碰 topics/状态机（那是 core 的事）。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta
from typing import Any
from uuid import UUID

import structlog
from psycopg import Connection

from scholar_agents.db import to_pgvector
from scholar_agents.embedding import EmbeddingError, embed
from scholar_agents.sourcing.fetcher import (
    FetchError,
    FullText,
    Role,
    build_item,
    build_manual_item,
    fetch_feed,
)
from scholar_agents.sourcing.fetcher import (
    published_at as entry_published,
)

log = structlog.get_logger()

#: 语义去重阈值与回看窗口（SPEC-003 §3）
SIMILARITY_THRESHOLD = 0.92
LOOKBACK_DAYS = 14

#: 单次采集的条目上限（可被 sources.fetch_config.max_items 覆盖）。
#: 必要性来自实测：arXiv 的 RSS 一次给出当天全部论文（cs.AI 单次 295 篇、
#: cs.CL 119 篇），两个源就能一天灌进 400+ 条，淹没选题池并浪费 embedding 额度。
#: feed 通常按时间倒序，取前 N 条即最新的 N 条。
DEFAULT_MAX_ITEMS = 30

#: 时效性窗口：早于此天数的条目直接丢弃，**不做 embedding**（省额度）。
#: 必要性来自实测：OpenAI 的 news RSS 返回**全部历史归档**（单次 1115 条，
#: 回溯数年），若不过滤会把多年前的旧闻灌进选题池。
#: 这是比 max_items 更本质的约束 —— 选题系统要的是新鲜资讯。
#: 可被 sources.fetch_config.max_age_days 覆盖（如常青教程类源可放宽）。
DEFAULT_MAX_AGE_DAYS = 14


@dataclass(slots=True)
class SourcingStats:
    fetched: int = 0
    inserted: int = 0
    dup_exact: int = 0
    dup_semantic: int = 0
    skipped: int = 0
    too_old: int = 0
    failed: int = 0
    item_ids: list[UUID] = field(default_factory=list)

    def as_dict(self) -> dict[str, int]:
        return {
            "fetched": self.fetched,
            "inserted": self.inserted,
            "dup_exact": self.dup_exact,
            "dup_semantic": self.dup_semantic,
            "skipped": self.skipped,
            "too_old": self.too_old,
            "failed": self.failed,
        }


@dataclass(slots=True)
class SourceConfig:
    role: Role
    full_text: FullText
    max_items: int
    max_age_days: int


def _positive_int(raw: object, default: int, field: str) -> int:
    """读一个正整数配置项；非法值退回默认而不是让整轮采集崩掉。"""
    if raw is None:
        return default
    if not isinstance(raw, int | str):
        log.warning("bad_config_value", field=field, value=repr(raw), using=default)
        return default
    try:
        return max(1, int(raw))
    except ValueError:
        log.warning("bad_config_value", field=field, value=raw, using=default)
        return default


def _source_config(row: dict[str, Any]) -> SourceConfig:
    """从 sources.fetch_config 读配置，缺省保守取值（signal + 不抓页面 + 默认上限）。"""
    cfg = row.get("fetch_config") or {}
    role: Role = "material" if cfg.get("role") == "material" else "signal"
    full_text: FullText = (
        "fetch_page" if cfg.get("full_text") == "fetch_page" else "rss_description"
    )
    return SourceConfig(
        role=role,
        full_text=full_text,
        max_items=_positive_int(cfg.get("max_items"), DEFAULT_MAX_ITEMS, "max_items"),
        max_age_days=_positive_int(
            cfg.get("max_age_days"), DEFAULT_MAX_AGE_DAYS, "max_age_days"
        ),
    )


def _semantic_duplicate(
    conn: Connection[dict[str, Any]], vec: list[float]
) -> tuple[str, float] | None:
    """近 LOOKBACK_DAYS 天内最相似的一条；超过阈值则视为重复。

    pgvector 的 `<=>` 是余弦距离，相似度 = 1 - 距离。
    """
    with conn.cursor() as cur:
        cur.execute(
            """
            select id::text as id, 1 - (embedding <=> %s::vector) as similarity
            from raw_items
            where embedding is not null
              and created_at > now() - make_interval(days => %s)
            order by embedding <=> %s::vector
            limit 1
            """,
            (to_pgvector(vec), LOOKBACK_DAYS, to_pgvector(vec)),
        )
        row = cur.fetchone()
    if row and row["similarity"] is not None and float(row["similarity"]) >= SIMILARITY_THRESHOLD:
        return row["id"], float(row["similarity"])
    return None


def handle_source_fetch(
    conn: Connection[dict[str, Any]], payload: dict[str, Any]
) -> SourcingStats:
    """采集一个信源，或处理手动投喂 payload 的单个 URL。"""
    source_id = payload["sourceId"]
    with conn.cursor() as cur:
        cur.execute(
            "select id::text as id, name, url, fetch_config, enabled from sources where id = %s",
            (source_id,),
        )
        source = cur.fetchone()
    if source is None:
        raise ValueError(f"source {source_id} not found")
    if not source["enabled"]:
        log.info("source_disabled_skip", source=source["name"])
        return SourcingStats()
    manual_url = payload.get("url")
    if not source["url"] and not manual_url:
        raise ValueError(f"source {source['name']} has no url")

    cfg = _source_config(source)
    role, full_text = cfg.role, cfg.full_text
    stats = SourcingStats()
    if manual_url:
        item = build_manual_item(str(manual_url))
        stats.fetched = 1
        if item is None:
            stats.failed = 1
            return stats
        return _store_item(conn, source, item, stats)

    all_entries = fetch_feed(source["url"])  # 整个 feed 拉不动 → 抛 FetchError，由 worker 重试
    entries = all_entries[: cfg.max_items]
    if len(all_entries) > cfg.max_items:
        log.info(
            "feed_truncated",
            source=source["name"],
            total=len(all_entries),
            taken=cfg.max_items,
        )
    stats.fetched = len(entries)
    cutoff = datetime.now(UTC) - timedelta(days=cfg.max_age_days)

    for entry in entries:
        try:
            # 时效性过滤放在 build_item 之前：早于窗口的条目连页面都不抓、
            # 更不做 embedding（OpenAI 的 news RSS 实测回溯数年，1115 条历史归档）
            published = entry_published(entry)
            if published is not None and published < cutoff:
                stats.too_old += 1
                continue

            item = build_item(entry, role=role, full_text=full_text)
            if item is None:
                stats.skipped += 1
                continue

            # 精确去重：content_hash 有 unique 约束，先查省掉一次异常往返
            _store_item(conn, source, item, stats)

        except (EmbeddingError, FetchError) as exc:
            # 单条失败不拖垮整个源（SPEC-008 §6：单源失败隔离的条目级版本）
            stats.failed += 1
            conn.rollback()
            log.warning("item_failed", source=source["name"], error=str(exc)[:150])
        except Exception as exc:  # noqa: BLE001 — 逐条兜底，避免一条脏数据毁掉整批
            stats.failed += 1
            conn.rollback()
            log.warning(
                "item_failed_unexpected",
                source=source["name"],
                error=f"{type(exc).__name__}: {str(exc)[:150]}",
            )

    log.info("source_fetch_done", source=source["name"], role=role, **stats.as_dict())
    return stats


def _store_item(
    conn: Connection[dict[str, Any]], source: dict[str, Any], item: Any, stats: SourcingStats
) -> SourcingStats:
    with conn.cursor() as cur:
        cur.execute("select 1 from raw_items where content_hash = %s", (item.content_hash,))
        existing = cur.fetchone()
        if existing:
            stats.dup_exact += 1
            cur.execute("select id from raw_items where content_hash = %s", (item.content_hash,))
            row = cur.fetchone()
            if row is not None:
                stats.item_ids.append(UUID(str(row["id"])))
            return stats

    vec = embed(f"{item.title}\n\n{item.content}")
    dup = _semantic_duplicate(conn, vec)
    if dup is not None:
        stats.dup_semantic += 1
        log.info(
            "semantic_duplicate",
            title=item.title[:50],
            existing=dup[0],
            similarity=round(dup[1], 4),
        )
        return stats

    with conn.cursor() as cur:
        cur.execute(
            """
            insert into raw_items
                (source_id, title, url, author, content, published_at,
                 content_hash, embedding, status)
            values (%s, %s, %s, %s, %s, %s, %s, %s::vector, 'new')
            on conflict (content_hash) do nothing
            returning id
            """,
            (
                source["id"],
                item.title,
                item.url,
                item.author,
                item.content,
                item.published_at,
                item.content_hash,
                to_pgvector(vec),
            ),
        )
        row = cur.fetchone()
        if row is not None:
            stats.inserted += 1
            stats.item_ids.append(UUID(str(row["id"])))
    conn.commit()
    return stats
