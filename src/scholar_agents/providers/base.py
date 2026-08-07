"""ModelProvider 抽象（ADR-002）。

统一 Anthropic 协议与 OpenAI 协议的差异，runtime 之上不感知协议：
- 工具调用：Anthropic tool_use block vs OpenAI tool_calls 字段 → 归一为 ToolCallBlock
- 工具结果：Anthropic user 消息内 tool_result vs OpenAI role="tool" → 归一为 ToolResultMessage
- system：Anthropic 独立参数 vs OpenAI messages[0] → ChatRequest.system
- 结构化输出：各 adapter 自行实现（Anthropic 强制 tool call / OpenAI json_schema），
  接口只承诺"返回符合 schema 的 JSON"，校验与重试在 runtime.structured 中。
"""

from __future__ import annotations

from typing import Any, Literal, Protocol

from pydantic import BaseModel, Field


class ToolSpec(BaseModel):
    """工具声明（协议无关）。input_schema 为 JSON Schema。"""

    name: str
    description: str
    input_schema: dict[str, Any]


class TextBlock(BaseModel):
    type: Literal["text"] = "text"
    text: str


class ToolCallBlock(BaseModel):
    """归一化的工具调用（Anthropic tool_use / OpenAI tool_calls）。"""

    type: Literal["tool_call"] = "tool_call"
    id: str
    name: str
    arguments: dict[str, Any]


ContentBlock = TextBlock | ToolCallBlock


class UserMessage(BaseModel):
    role: Literal["user"] = "user"
    content: str


class AssistantMessage(BaseModel):
    role: Literal["assistant"] = "assistant"
    content: list[ContentBlock]


class ToolResultMessage(BaseModel):
    """工具执行结果，回传给模型（协议差异最大处，由 adapter 转换）。"""

    role: Literal["tool"] = "tool"
    tool_call_id: str
    content: str
    is_error: bool = False


Message = UserMessage | AssistantMessage | ToolResultMessage


class Usage(BaseModel):
    input_tokens: int = 0
    output_tokens: int = 0


class ChatRequest(BaseModel):
    messages: list[Message]
    system: str | None = None
    tools: list[ToolSpec] = Field(default_factory=list)
    max_tokens: int = 4096
    temperature: float | None = None
    # 结构化输出：期望模型返回符合该 JSON Schema 的 JSON（adapter 各自实现）
    json_schema: dict[str, Any] | None = None


class ChatResponse(BaseModel):
    content: list[ContentBlock]
    stop_reason: Literal["end_turn", "tool_use", "max_tokens", "other"]
    usage: Usage
    model: str
    # 原始响应（调试/Langfuse 上报用），不参与业务逻辑
    raw: dict[str, Any] = Field(default_factory=dict)

    @property
    def text(self) -> str:
        return "".join(b.text for b in self.content if isinstance(b, TextBlock))

    @property
    def tool_calls(self) -> list[ToolCallBlock]:
        return [b for b in self.content if isinstance(b, ToolCallBlock)]


class ModelProvider(Protocol):
    """所有 provider 的统一接口。实现必须是纯 adapter：不做重试/预算/日志（runtime 负责）。"""

    name: str

    def complete(self, model: str, req: ChatRequest) -> ChatResponse: ...
