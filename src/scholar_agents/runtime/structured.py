"""结构化输出：跨协议统一的"给我符合 schema 的 JSON"（ADR-002）。

校验失败带着错误信息重试（最多 max_attempts 次），仍失败则抛出 —— 契约错误必须炸出来。
"""

from __future__ import annotations

import json
from typing import Any

import structlog
from jsonschema import Draft7Validator

from scholar_agents import telemetry
from scholar_agents.providers.base import (
    ChatRequest,
    Message,
    ModelProvider,
    Usage,
    UserMessage,
)

log = structlog.get_logger()


class StructuredOutputError(Exception):
    """重试耗尽仍拿不到合法 JSON。"""

    def __init__(self, message: str, usage: Usage | None = None) -> None:
        super().__init__(message)
        self.usage = usage or Usage()


def complete_structured(
    provider: ModelProvider,
    model: str,
    system: str,
    user_prompt: str,
    json_schema: dict[str, Any],
    max_attempts: int = 3,
    max_tokens: int = 4096,
) -> tuple[dict[str, Any], Usage]:
    """返回 (符合 schema 的 dict, 累计 usage)。"""
    validator = Draft7Validator(json_schema)
    messages: list[Message] = [UserMessage(content=user_prompt)]
    total = Usage()

    for attempt in range(1, max_attempts + 1):
        resp = provider.complete(
            model,
            ChatRequest(
                messages=messages, system=system, json_schema=json_schema, max_tokens=max_tokens
            ),
        )
        total.input_tokens += resp.usage.input_tokens
        total.output_tokens += resp.usage.output_tokens

        errors: list[str] = []
        try:
            data = json.loads(resp.text)
        except json.JSONDecodeError as exc:
            errors = [f"invalid JSON: {exc}"]
        else:
            errors = [e.message for e in validator.iter_errors(data)]
            if not errors:
                return data, total

        log.warning("structured_retry", attempt=attempt, errors=errors[:3])
        if attempt < max_attempts:
            telemetry.structured_retries.add(1, {"provider": provider.name, "model": model})
        messages = [
            UserMessage(
                content=(
                    f"{user_prompt}\n\n"
                    f"你上一次的输出未通过 schema 校验：{'; '.join(errors[:5])}\n"
                    "请修正并只输出符合 schema 的 JSON。"
                )
            )
        ]

    raise StructuredOutputError(f"no valid structured output after {max_attempts} attempts", total)
