"""加载并校验 scholar-shared 中的平台档案。"""

from __future__ import annotations

from pathlib import Path

import yaml
from scholar_contracts.models import Platform, PlatformProfile

from scholar_agents.errors import PermanentJobError


def load_platform_profile(directory: Path, platform: Platform) -> PlatformProfile:
    path = directory / f"{platform.value}.v1.yaml"
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        profile = PlatformProfile.model_validate(raw)
    except (OSError, ValueError, TypeError) as exc:
        raise PermanentJobError(f"invalid platform profile {path}: {exc}") from exc
    if profile.platform != platform:
        raise PermanentJobError(
            f"profile {path} declares {profile.platform.value}, expected {platform.value}"
        )
    return profile


def profile_prompt(profile: PlatformProfile) -> str:
    structure = "\n".join(f"- {item.root}" for item in profile.structureTemplate)
    rules = "\n".join(f"- {item.root}" for item in profile.styleRules)
    hard = profile.hardConstraints
    return f"""平台：{profile.platform.value}
平台档案：{profile.id}@{profile.version}
写作人设：{profile.voice}

推荐结构：
{structure}

风格规则：
{rules}

确定性硬约束：
- 正文可见字符数：{hard.minCharacters}–{hard.maxCharacters}
- 标题最多 {hard.titleMaxCharacters} 个字符
- 话题标签：{hard.minTags}–{hard.maxTags} 个
- 图片占位符至少 {hard.minImagePlaceholders} 个，格式为 <!-- image: 描述 -->
- 输出 Markdown，所有事实必须能追溯到给定素材，不得虚构经历、数字或来源
"""
