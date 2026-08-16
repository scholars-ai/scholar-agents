"""Formatter：不用 LLM，做 Markdown 规范化与平台硬约束检查。"""

from __future__ import annotations

import re
from dataclasses import dataclass

from scholar_contracts.models import PlatformProfile

from scholar_agents.errors import JobError

_MARKDOWN_SYNTAX = re.compile(r"[`*_>#\[\]()!|~-]")
_TAG = re.compile(r"(?<![\w#])#([\w\u3400-\u9fff-]+)")
_IMAGE_PLACEHOLDER = re.compile(r"<!--\s*image\s*:", re.IGNORECASE)


class WriterConstraintError(JobError):
    """模型成文仍违反硬约束；重试可能由新的模型采样修复。"""


@dataclass(frozen=True, slots=True)
class FormattedArticle:
    title: str
    content_md: str
    character_count: int
    tag_count: int
    image_placeholder_count: int


def visible_character_count(markdown: str) -> int:
    without_comments = re.sub(r"<!--.*?-->", "", markdown, flags=re.DOTALL)
    without_urls = re.sub(r"https?://\S+", "", without_comments)
    return len(re.sub(r"\s+", "", _MARKDOWN_SYNTAX.sub("", without_urls)))


def format_article(title: str, content_md: str, profile: PlatformProfile) -> FormattedArticle:
    normalized_title = re.sub(r"^#+\s*", "", title.strip()).strip()
    normalized_content = content_md.replace("\r\n", "\n").strip() + "\n"
    hard = profile.hardConstraints
    characters = visible_character_count(normalized_content)
    tags = len(_TAG.findall(normalized_content))
    placeholders = len(_IMAGE_PLACEHOLDER.findall(normalized_content))
    violations: list[str] = []
    if not normalized_title:
        violations.append("标题不能为空")
    if len(normalized_title) > hard.titleMaxCharacters:
        violations.append(
            f"标题 {len(normalized_title)} 字，超过上限 {hard.titleMaxCharacters}"
        )
    if characters < hard.minCharacters:
        violations.append(f"正文 {characters} 字，低于下限 {hard.minCharacters}")
    if characters > hard.maxCharacters:
        violations.append(f"正文 {characters} 字，超过上限 {hard.maxCharacters}")
    if tags < hard.minTags or tags > hard.maxTags:
        violations.append(f"话题标签 {tags} 个，要求 {hard.minTags}–{hard.maxTags} 个")
    if placeholders < hard.minImagePlaceholders:
        violations.append(
            f"图片占位符 {placeholders} 个，至少需要 {hard.minImagePlaceholders} 个"
        )
    if violations:
        raise WriterConstraintError("；".join(violations))
    return FormattedArticle(
        title=normalized_title,
        content_md=normalized_content,
        character_count=characters,
        tag_count=tags,
        image_placeholder_count=placeholders,
    )
