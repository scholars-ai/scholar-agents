"""source_fetch 的 job handler（SPEC-003 §3）。

流程：拉 feed → 逐条清洗 → 精确去重（content_hash）→ embedding → 语义去重
（近 14 天 cos > 0.92）→ 写入 raw_items(status=new)。

两条纪律：
1. **单源失败隔离**：单条 entry 出错只跳过该条并计数，不让整个 job 失败；
   整个 feed 拉不动才抛异常（交由 worker 重试）。
2. **只写 raw_items**：不碰 topics/状态机（那是 core 的事）。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import structlog
from psycopg import Connection

from scholar_agents.db import to_pgvector
from scholar_agents.embedding import EmbeddingError, embed
from scholar_agents.sourcing.fetcher import (
    FetchError,
    FullText,
    Role,
    build_item,
    fetch_feed,
)

log = structlog.get_logger()

#: 语义去重阈值与回看窗口（SPEC-003 §3）
SIMILARITY_THRESHOLD = 0.92
LOOKBACK_DAYS = 14


@dataclass(slots=True)
class SourcingStats:
    fetched: int = 0
    inserted: int = 0
    dup_exact: int = 0
    dup_semantic: int = 0
    skipped: int = 0
    failed: int = 0

    def as_dict(self) -> dict[str, int]:
        return {
            "fetched": self.fetched,
            "inserted": self.inserted,
            "dup_exact": self.dup_exact,
            "dup_semantic": self.dup_semantic,
            "skipped": self.skipped,
            "failed": self.failed,
        }


def _source_config(row: dict[str, Any]) -> tuple[Role, FullText]:
    """从 sources.fetch_config 读角色配置，缺省保守取值（signal + 不抓页面）。"""
    cfg = row.get("fetch_config") or {}
    role: Role = "material" if cfg.get("role") == "material" else "signal"
    full_text: FullText = (
        "fetch_page" if cfg.get("full_text") == "fetch_page" else "rss_description"
    )
    return role, full_text


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
    """采集一个信源。payload: {"sourceId": uuid}"""
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
    if not source["url"]:
        raise ValueError(f"source {source['name']} has no url")

    role, full_text = _source_config(source)
    stats = SourcingStats()
    entries = fetch_feed(source["url"])  # 整个 feed 拉不动 → 抛 FetchError，由 worker 重试
    stats.fetched = len(entries)

    for entry in entries:
        try:
            item = build_item(entry, role=role, full_text=full_text)
            if item is None:
                stats.skipped += 1
                continue

            # 精确去重：content_hash 有 unique 约束，先查省掉一次异常往返
            with conn.cursor() as cur:
                cur.execute("select 1 from raw_items where content_hash = %s", (item.content_hash,))
                if cur.fetchone():
                    stats.dup_exact += 1
                    continue

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
                continue

            with conn.cursor() as cur:
                cur.execute(
                    """
                    insert into raw_items
                        (source_id, title, url, author, content, published_at,
                         content_hash, embedding, status)
                    values (%s, %s, %s, %s, %s, %s, %s, %s::vector, 'new')
                    on conflict (content_hash) do nothing
                    """,
                    (
                        source_id,
                        item.title,
                        item.url,
                        item.author,
                        item.content,
                        item.published_at,
                        item.content_hash,
                        to_pgvector(vec),
                    ),
                )
                stats.inserted += cur.rowcount
            conn.commit()

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
