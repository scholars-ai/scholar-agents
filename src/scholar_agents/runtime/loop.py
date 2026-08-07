"""自研 agent loop（ADR-002）：while + 工具调度。

用于需要自主性的 Agent（TopicScout / Reflector）；
纯结构化调用（Judge / Writer 各步骤）直接用 runtime.structured，不需要 loop。
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import structlog

from scholar_agents.providers.base import (
    AssistantMessage,
    ChatRequest,
    ChatResponse,
    Message,
    ModelProvider,
    ToolResultMessage,
    ToolSpec,
    Usage,
    UserMessage,
)

log = structlog.get_logger()

ToolFn = Callable[[dict[str, object]], str]


@dataclass
class Tool:
    spec: ToolSpec
    fn: ToolFn


@dataclass
class LoopResult:
    final_text: str
    turns: int
    usage: Usage
    messages: list[Message] = field(default_factory=list)


class ToolNotFoundError(Exception):
    pass


def run_loop(
    provider: ModelProvider,
    model: str,
    system: str,
    user_prompt: str,
    tools: list[Tool],
    max_turns: int = 10,
    max_tokens: int = 4096,
) -> LoopResult:
    """经典 agent loop：调模型 → 有 tool_call 就执行并回传 → 直到 end_turn 或轮数耗尽。

    工具执行抛异常不炸 loop：错误文本作为 is_error 的 tool result 回传，让模型自行修正；
    但未知工具名视为契约错误，直接抛出。
    """
    by_name = {t.spec.name: t for t in tools}
    messages: list[Message] = [UserMessage(content=user_prompt)]
    total = Usage()

    for turn in range(1, max_turns + 1):
        resp: ChatResponse = provider.complete(
            model,
            ChatRequest(
                messages=messages,
                system=system,
                tools=[t.spec for t in tools],
                max_tokens=max_tokens,
            ),
        )
        total.input_tokens += resp.usage.input_tokens
        total.output_tokens += resp.usage.output_tokens
        messages.append(AssistantMessage(content=resp.content))

        if resp.stop_reason != "tool_use":
            return LoopResult(final_text=resp.text, turns=turn, usage=total, messages=messages)

        for call in resp.tool_calls:
            tool = by_name.get(call.name)
            if tool is None:
                raise ToolNotFoundError(f"model called unknown tool {call.name!r}")
            try:
                result = tool.fn(call.arguments)
                is_error = False
            except Exception as exc:  # noqa: BLE001 — 工具失败要回传模型而不是炸 loop
                log.warning("tool_failed", tool=call.name, error=str(exc))
                result, is_error = f"tool error: {exc}", True
            messages.append(
                ToolResultMessage(tool_call_id=call.id, content=result, is_error=is_error)
            )

    log.warning("loop_exhausted", max_turns=max_turns)
    return LoopResult(final_text="", turns=max_turns, usage=total, messages=messages)
