"""OpenAI 兼容协议 adapter。

一个 adapter 通吃所有 OpenAI 兼容端点（DeepSeek/Qwen/Kimi/GLM/OpenRouter/Ollama）：
只需不同的 base_url + api_key。

协议特点（相对 Anthropic）：
- system 是 messages 里的一条消息
- 工具调用在 assistant 消息的 tool_calls 字段，arguments 是 JSON **字符串**
- 工具结果是独立的 role="tool" 消息
- 结构化输出：response_format = json_schema
"""

from __future__ import annotations

import json
from typing import Any, Literal

import openai

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


class OpenAICompatProvider:
    def __init__(
        self,
        name: str = "openai",
        api_key: str | None = None,
        base_url: str | None = None,
        json_mode: str = "schema",
    ) -> None:
        self.name = name
        self.json_mode = json_mode
        self._client = openai.OpenAI(api_key=api_key, base_url=base_url)

    def complete(self, model: str, req: ChatRequest) -> ChatResponse:
        kwargs: dict[str, Any] = {
            "model": model,
            "max_tokens": req.max_tokens,
            "messages": _to_openai_messages(req.messages, req.system),
        }
        if req.tools and req.json_schema is None:
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": t.name,
                        "description": t.description,
                        "parameters": t.input_schema,
                    },
                }
                for t in req.tools
            ]
        if req.json_schema is not None and self.json_mode == "tool":
            kwargs["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": "emit_structured_output",
                        "description": "Return the required structured output.",
                        "parameters": req.json_schema,
                    },
                }
            ]
            kwargs["tool_choice"] = {
                "type": "function",
                "function": {"name": "emit_structured_output"},
            }
        if req.json_schema is not None:
            if self.json_mode == "tool":
                pass
            elif self.json_mode == "object":
                kwargs["response_format"] = {"type": "json_object"}
            else:
                kwargs["response_format"] = {
                    "type": "json_schema",
                    "json_schema": {"name": "structured_output", "schema": req.json_schema},
                }
        if req.temperature is not None:
            kwargs["temperature"] = req.temperature

        resp = self._client.chat.completions.create(**kwargs)
        choice = resp.choices[0]

        usage = Usage()
        if resp.usage is not None:
            usage = Usage(
                input_tokens=resp.usage.prompt_tokens, output_tokens=resp.usage.completion_tokens
            )

        content: list[ContentBlock] = []
        if req.json_schema is not None and self.json_mode == "tool":
            for tc in choice.message.tool_calls or []:
                if tc.function.name == "emit_structured_output":
                    content.append(TextBlock(text=tc.function.arguments))
            return ChatResponse(
                content=content,
                stop_reason=_map_stop(choice.finish_reason),
                usage=usage,
                model=resp.model,
                raw={"id": resp.id},
            )
        if choice.message.content:
            content.append(TextBlock(text=choice.message.content))
        for tc in choice.message.tool_calls or []:
            # OpenAI 的 arguments 是 JSON 字符串；坏 JSON 归一为空参并保留原文，让 runtime 决策
            try:
                args = json.loads(tc.function.arguments)
            except json.JSONDecodeError:
                args = {"__unparsed__": tc.function.arguments}
            content.append(ToolCallBlock(id=tc.id, name=tc.function.name, arguments=args))

        return ChatResponse(
            content=content,
            stop_reason=_map_stop(choice.finish_reason),
            usage=usage,
            model=resp.model,
            raw={"id": resp.id},
        )


def _map_stop(reason: str | None) -> Literal["end_turn", "tool_use", "max_tokens", "other"]:
    match reason:
        case "stop":
            return "end_turn"
        case "tool_calls":
            return "tool_use"
        case "length":
            return "max_tokens"
        case _:
            return "other"


def _to_openai_messages(messages: list[Message], system: str | None) -> list[dict[str, Any]]:
    """归一消息 → OpenAI 消息。system 前置；工具调用需重建 tool_calls 字段。"""
    out: list[dict[str, Any]] = []
    if system is not None:
        out.append({"role": "system", "content": system})
    for m in messages:
        if isinstance(m, UserMessage):
            out.append({"role": "user", "content": m.content})
        elif isinstance(m, AssistantMessage):
            text = "".join(b.text for b in m.content if isinstance(b, TextBlock))
            tool_calls = [
                {
                    "id": b.id,
                    "type": "function",
                    "function": {
                        "name": b.name,
                        "arguments": json.dumps(b.arguments, ensure_ascii=False),
                    },
                }
                for b in m.content
                if isinstance(b, ToolCallBlock)
            ]
            msg: dict[str, Any] = {"role": "assistant", "content": text or None}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            out.append(msg)
        elif isinstance(m, ToolResultMessage):
            out.append({"role": "tool", "tool_call_id": m.tool_call_id, "content": m.content})
    return out
