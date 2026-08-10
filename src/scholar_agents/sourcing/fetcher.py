"""信源抓取与清洗（SPEC-003 §3）。

设计要点：
- **feed 是不可信输入**：解析走 feedparser（内部禁用了实体解析），不用 stdlib
  ElementTree 裸解析，防 XXE / billion-laughs。
- **两类信源角色**（SPEC-003 §2.1）：`material` 拿原文、`signal` 只存二手摘要。
  `full_text=fetch_page` 时才额外抓原文页（trafilatura），省掉无谓请求。
- **单源失败隔离**：本模块只抛异常，是否影响其他源由 handler 决定（SPEC-008 §6）。
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Literal

import feedparser
import httpx
import structlog
import trafilatura

log = structlog.get_logger()

Role = Literal["material", "signal"]
FullText = Literal["rss_description", "fetch_page"]

UA = "Mozilla/5.0 (compatible; scholars-ai/0.1; +https://github.com/scholars-ai)"
#: 正文入库上限：TopicScout 只需要判断素材厚薄与要点，超长截断（SPEC-003 §3）
MAX_CONTENT_CHARS = 20_000
#: 低于此长度视为"没拿到正文"，material 源会尝试抓页面补救
THIN_CONTENT_CHARS = 200


@dataclass(slots=True)
class FetchedItem:
    """一条采集结果，对应 raw_items 的一行（SPEC-002）。"""

    title: str
    url: str | None
    author: str | None
    content: str
    published_at: datetime | None
    #: 去重键：优先 feed 的 guid（实测聚合源唯一性 50/50），否则 sha256(正文)
    content_hash: str


class FetchError(RuntimeError):
    """整个 feed 拉取失败（网络/格式）。单条 entry 的问题不用这个，跳过即可。"""


def fetch_feed(url: str, *, timeout: float = 45.0) -> list[dict[str, Any]]:
    """拉取并解析 feed，返回 entry 列表。"""
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True, headers={"User-Agent": UA})
        resp.raise_for_status()
    except httpx.HTTPError as exc:
        raise FetchError(f"feed request failed: {exc}") from exc

    parsed = feedparser.parse(resp.content)
    # feedparser 对轻微畸形会置 bozo 但仍可能解析出条目：有条目就继续，没条目才算失败
    if not parsed.entries:
        reason = getattr(parsed, "bozo_exception", None)
        raise FetchError(f"feed has no entries ({reason or 'unknown'})")
    if getattr(parsed, "bozo", 0):
        log.warning("feed_bozo", url=url, reason=str(getattr(parsed, "bozo_exception", ""))[:120])
    return list(parsed.entries)


def _entry_body(entry: dict[str, Any]) -> str:
    """取 entry 里最长的正文字段：content:encoded > content > summary > description。"""
    candidates: list[str] = []
    for c in entry.get("content") or []:
        if isinstance(c, dict) and c.get("value"):
            candidates.append(str(c["value"]))
    for key in ("summary", "description"):
        if entry.get(key):
            candidates.append(str(entry[key]))
    return max(candidates, key=len, default="")


def _clean_html(html: str) -> str:
    """HTML → 纯文本。短片段走正则去标签（trafilatura 对短文本会返回空）。"""
    if not html:
        return ""
    if len(html) > 1200:
        extracted = trafilatura.extract(html, include_comments=False, include_tables=True)
        if extracted:
            return extracted.strip()
    text = re.sub(r"<br\s*/?>|</p>", "\n", html, flags=re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = re.sub(r"&nbsp;", " ", text)
    text = re.sub(r"&amp;", "&", text)
    text = re.sub(r"&quot;", '"', text)
    text = re.sub(r"&#39;", "'", text)
    text = re.sub(r"&lt;", "<", text)
    text = re.sub(r"&gt;", ">", text)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def fetch_page_text(url: str, *, timeout: float = 45.0) -> str:
    """抓原文页并提取正文。失败返回空串——由调用方决定要不要降级用摘要。"""
    try:
        resp = httpx.get(url, timeout=timeout, follow_redirects=True, headers={"User-Agent": UA})
        resp.raise_for_status()
        extracted = trafilatura.extract(resp.text, include_comments=False, include_tables=True)
        return (extracted or "").strip()
    except (httpx.HTTPError, ValueError) as exc:
        log.warning("page_fetch_failed", url=url, error=str(exc)[:120])
        return ""


def published_at(entry: dict[str, Any]) -> datetime | None:
    for key in ("published_parsed", "updated_parsed"):
        st = entry.get(key)
        if not st:
            continue
        try:
            y, mo, d, h, mi, s = (int(x) for x in st[:6])
            return datetime(y, mo, d, h, mi, s, tzinfo=UTC)
        except (TypeError, ValueError):
            continue
    return None


def build_item(entry: dict[str, Any], *, role: Role, full_text: FullText) -> FetchedItem | None:
    """把一条 feed entry 转成 FetchedItem。标题或正文为空则返回 None（跳过该条）。"""
    title = _clean_html(str(entry.get("title") or "")).strip()
    link = entry.get("link") or None
    content = _clean_html(_entry_body(entry))

    # material 源且 RSS 正文太薄时，抓原文页补救（signal 源不抓，它本来就只有摘要）
    needs_page = full_text == "fetch_page" or len(content) < THIN_CONTENT_CHARS
    if role == "material" and link and needs_page:
        page = fetch_page_text(link)
        if len(page) > len(content):
            content = page

    if not title or not content:
        return None

    content = content[:MAX_CONTENT_CHARS]
    # guid 优先：聚合源的 guid 稳定且唯一，比正文 hash 更能识别"同一条被改写"
    guid = entry.get("id") or entry.get("guid")
    content_hash = (
        f"guid:{hashlib.sha256(str(guid).encode()).hexdigest()}"
        if guid
        else f"body:{hashlib.sha256(content.encode()).hexdigest()}"
    )

    return FetchedItem(
        title=title,
        url=link,
        author=(str(entry.get("author")) if entry.get("author") else None),
        content=content,
        published_at=published_at(entry),
        content_hash=content_hash,
    )
