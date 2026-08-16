"""开发/运维 CLI。

M1 期间 core 的 cron 尚未落地，用这里手动触发采集；cron 上线后 seed-sources 仍
保留（信源清单以 config/sources.yaml 为准，便于 review 与版本化）。

    uv run python -m scholar_agents.cli seed-sources
    uv run python -m scholar_agents.cli fetch --all
    uv run python -m scholar_agents.cli fetch --name "AIHOT 精选"
    uv run python -m scholar_agents.cli stats
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

import structlog
import yaml

from scholar_agents.db import connection
from scholar_agents.sourcing.fetcher import FetchError
from scholar_agents.sourcing.handler import handle_source_fetch

log = structlog.get_logger()

CONFIG = Path(__file__).resolve().parent.parent.parent / "config" / "sources.yaml"
RSSHUB_BASE = os.environ.get("RSSHUB_BASE", "http://127.0.0.1:1200")


def _load_sources() -> list[dict[str, Any]]:
    data = yaml.safe_load(CONFIG.read_text())
    out = []
    for s in data["sources"]:
        s = dict(s)
        s["url"] = s["url"].replace("${RSSHUB_BASE}", RSSHUB_BASE)
        out.append(s)
    return out


def cmd_seed_sources(_: argparse.Namespace) -> int:
    """把 config/sources.yaml 灌入 sources 表（按 name 幂等 upsert）。"""
    sources = _load_sources()
    with connection() as conn, conn.cursor() as cur:
        for s in sources:
            cur.execute(
                """
                insert into sources (name, type, url, category, weight, enabled, fetch_config)
                values (%s, %s, %s, %s, %s, true, %s)
                on conflict (name) do update set
                    type = excluded.type, url = excluded.url,
                    category = excluded.category, weight = excluded.weight,
                    fetch_config = excluded.fetch_config, updated_at = now()
                """,
                (
                    s["name"],
                    s["type"],
                    s["url"],
                    s["category"],
                    s["weight"],
                    json.dumps(s.get("fetch_config") or {}),
                ),
            )
        cur.execute("select count(*) as n, count(*) filter (where enabled) as on_ from sources")
        row = cur.fetchone() or {"n": 0, "on_": 0}
    print(f"seeded {len(sources)} sources; table now has {row['n']} ({row['on_']} enabled)")
    return 0


def cmd_fetch(args: argparse.Namespace) -> int:
    """采集：--all 全部启用的源，或 --name 指定单个源。"""
    with connection() as conn:
        with conn.cursor() as cur:
            if args.name:
                cur.execute(
                    "select id::text as id, name from sources where name = %s and enabled",
                    (args.name,),
                )
            else:
                cur.execute("select id::text as id, name from sources where enabled order by name")
            targets = cur.fetchall()

        if not targets:
            print("no matching enabled source", file=sys.stderr)
            return 1

        totals: dict[str, int] = {}
        failures: list[str] = []
        for t in targets:
            try:
                stats = handle_source_fetch(conn, {"sourceId": t["id"]})
                for k, v in stats.as_dict().items():
                    totals[k] = totals.get(k, 0) + v
                print(
                    f"  {t['name']:34} fetched={stats.fetched:3} inserted={stats.inserted:3} "
                    f"dup={stats.dup_exact + stats.dup_semantic:3} "
                    f"skip={stats.skipped:2} fail={stats.failed:2}"
                )
            except (FetchError, ValueError) as exc:
                # 单源失败隔离（SPEC-008 §6）：记录并继续下一个源
                conn.rollback()
                failures.append(t["name"])
                print(f"  {t['name']:34} FAILED: {str(exc)[:70]}", file=sys.stderr)

    print(f"\ntotals: {totals}")
    if failures:
        print(f"failed sources ({len(failures)}/{len(targets)}): {', '.join(failures)}")
    return 0


def cmd_stats(_: argparse.Namespace) -> int:
    """看库里现有的采集情况。"""
    with connection() as conn, conn.cursor() as cur:
        cur.execute("""
            select s.name, s.category, s.fetch_config->>'role' as role,
                   count(r.id) as items,
                   coalesce(round(avg(length(r.content))), 0) as avg_len,
                   max(r.created_at) as latest
            from sources s left join raw_items r on r.source_id = s.id
            group by s.id, s.name, s.category, s.fetch_config
            order by items desc, s.name
        """)
        rows = cur.fetchall()
        cur.execute("select status, count(*) as n from raw_items group by status")
        by_status = {r["status"]: r["n"] for r in cur.fetchall()}

    print(f"{'源':34} {'类别':10} {'角色':9} {'条数':>5} {'平均正文':>9}  最近采集")
    print("-" * 96)
    for r in rows:
        latest = r["latest"].strftime("%m-%d %H:%M") if r["latest"] else "-"
        print(
            f"{r['name'][:33]:34} {r['category']:10} {(r['role'] or '-'):9} "
            f"{r['items']:5} {r['avg_len']:9}  {latest}"
        )
    print(f"\nraw_items by status: {by_status or '(empty)'}")
    return 0


def main() -> int:
    structlog.configure(wrapper_class=structlog.make_filtering_bound_logger(20))
    parser = argparse.ArgumentParser(prog="scholar_agents.cli")
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_seed = sub.add_parser("seed-sources", help="灌入 config/sources.yaml")
    p_seed.set_defaults(fn=cmd_seed_sources)

    p_fetch = sub.add_parser("fetch", help="采集信源")
    g = p_fetch.add_mutually_exclusive_group(required=True)
    g.add_argument("--all", action="store_true", help="全部启用的源")
    g.add_argument("--name", help="指定源名")
    p_fetch.set_defaults(fn=cmd_fetch)

    sub.add_parser("stats", help="查看采集统计").set_defaults(fn=cmd_stats)

    args = parser.parse_args()
    return int(args.fn(args))


if __name__ == "__main__":
    raise SystemExit(main())
