"""Anthropic 协议 adapter。

协议特点（相对 OpenAI）：
- system 是独立参数
- 工具调用是 assistant 消息里的 tool_use content block
- 工具结果是 **user 消息**里的 tool_result block
- 结构化输出：无 response_format，用"强制调用唯一工具"模式（tool_choice）
"""

from __future__ import annotations

import os
from typing import Any, Literal, cast

import anthropic

from scholar_agents.errors import normalize_provider_error
from scholar_agents.job_context import current_job
from scholar_agents.providers.base import (
    AssistantMessage,
    ChatRequest,
    ChatResponse,
    ContentBlock,
    Message,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    Usage,
    UserMessage,
)

_STRUCTURED_TOOL = "emit_structured_output"
DEFAULT_REQUEST_TIMEOUT_SECONDS = 120.0


def _request_timeout() -> float:
    raw = os.environ.get("LLM_REQUEST_TIMEOUT_SECONDS", "")
    if not raw:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_REQUEST_TIMEOUT_SECONDS
    return value if value > 0 else DEFAULT_REQUEST_TIMEOUT_SECONDS


class AnthropicProvider:
    name = "anthropic"

    def __init__(self, api_key: str | None = None, base_url: str | None = None) -> None:
        self._client = anthropic.Anthropic(
            api_key=api_key,
            base_url=base_url,
            timeout=_request_timeout(),
            max_retries=0,
        )

    def complete(self, model: str, req: ChatRequest) -> ChatResponse:
        tools = [
            {"name": t.name, "description": t.description, "input_schema": t.input_schema}
            for t in req.tools
        ]
        tool_choice: dict[str, Any] | None = None

        # 结构化输出：注入唯一工具并强制调用
        if req.json_schema is not None:
            tools = [
                {
                    "name": _STRUCTURED_TOOL,
                    "description": "Emit the final structured result.",
                    "input_schema": req.json_schema,
                }
            ]
            tool_choice = {"type": "tool", "name": _STRUCTURED_TOOL}

        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": req.max_tokens,
            "messages": _to_anthropic_messages(req.messages),
        }
        if req.system is not None:
            kwargs["system"] = req.system
        if tools:
            kwargs["tools"] = tools
        if tool_choice is not None:
            kwargs["tool_choice"] = tool_choice
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature
        job = current_job()
        if job is not None:
            job.raise_if_expired()
            kwargs["timeout"] = max(1.0, min(_request_timeout(), job.remaining_seconds()))

        try:
            resp = self._client.messages.create(**kwargs)
        except anthropic.AnthropicError as exc:
            raise normalize_provider_error(self.name, exc) from exc

        content: list[ContentBlock] = []
        for block in resp.content:
            if block.type == "text":
                content.append(TextBlock(text=block.text))
            elif block.type == "tool_use":
                args = cast(dict[str, Any], block.input)
                if req.json_schema is not None and block.name == _STRUCTURED_TOOL:
                    # 结构化输出模式：把工具入参直接作为文本 JSON 返回，
                    # runtime.structured 统一做校验与重试
                    import json

                    content.append(TextBlock(text=json.dumps(args, ensure_ascii=False)))
                else:
                    content.append(ToolCallBlock(id=block.id, name=block.name, arguments=args))

        return ChatResponse(
            content=content,
            stop_reason=_map_stop(resp.stop_reason),
            usage=Usage(
                input_tokens=resp.usage.input_tokens, output_tokens=resp.usage.output_tokens
            ),
            model=resp.model,
            raw={"id": resp.id},
        )


def _map_stop(reason: str | None) -> Literal["end_turn", "tool_use", "max_tokens", "other"]:
    match reason:
        case "end_turn" | "stop_sequence":
            return "end_turn"
        case "tool_use":
            return "tool_use"
        case "max_tokens":
            return "max_tokens"
        case _:
            return "other"


def _to_anthropic_messages(messages: list[Message]) -> list[dict[str, Any]]:
    """归一消息 → Anthropic 消息。ToolResultMessage 转为 user 消息内的 tool_result block。"""
    out: list[dict[str, Any]] = []
    for m in messages:
        if isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AssistantMessage):
            blocks: list[dict[str, Any]] = []
            for b in m.content:
                if isinstance(b, TextBlock):
                    blocks.append({"type": "text", "text": b.text})
                else:
                    blocks.append(
                        {"type": "tool_use", "id": b.id, "name": b.name, "input": b.arguments}
                    )
            out.append({"role": "assistant", "content": blocks})
        elif isinstance(m, ToolResultMessage):
            out.append(
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": m.tool_call_id,
                            "content": m.content,
                            "is_error": m.is_error,
                        }
                    ],
                }
            )
    return out
