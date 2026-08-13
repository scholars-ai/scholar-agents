"""Provider 归一化契约测试：两个 adapter 面对各自协议的 mock 响应，产出一致的 ChatResponse。

不打真实 API —— mock SDK 客户端层。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from scholar_agents.providers.anthropic_provider import AnthropicProvider, _to_anthropic_messages
from scholar_agents.providers.base import (
    AssistantMessage,
    ChatRequest,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    ToolSpec,
    UserMessage,
)
from scholar_agents.providers.openai_provider import OpenAICompatProvider, _to_openai_messages
from scholar_agents.providers.router import ModelRouter

TOOLS = [ToolSpec(name="search", description="search the web", input_schema={"type": "object"})]
REQ = ChatRequest(messages=[UserMessage(content="hi")], system="be brief", tools=TOOLS)


def _anthropic_sdk_response(blocks: list[Any], stop: str = "end_turn") -> MagicMock:
    resp = MagicMock()
    resp.content = blocks
    resp.stop_reason = stop
    resp.usage.input_tokens = 10
    resp.usage.output_tokens = 5
    resp.model = "claude-sonnet-5"
    resp.id = "msg_1"
    return resp


def _text_block(text: str) -> MagicMock:
    b = MagicMock()
    b.type = "text"
    b.text = text
    return b


def _tool_use_block(name: str, args: dict[str, Any]) -> MagicMock:
    b = MagicMock()
    b.type = "tool_use"
    b.id = "tu_1"
    b.name = name
    b.input = args
    return b


class TestAnthropicNormalization:
    def _provider(self, sdk_resp: MagicMock) -> AnthropicProvider:
        p = AnthropicProvider.__new__(AnthropicProvider)
        p._client = MagicMock()
        p._client.messages.create.return_value = sdk_resp
        return p

    def test_text_response(self) -> None:
        p = self._provider(_anthropic_sdk_response([_text_block("hello")]))
        resp = p.complete("claude-sonnet-5", REQ)
        assert resp.text == "hello"
        assert resp.stop_reason == "end_turn"
        assert resp.usage.input_tokens == 10

    def test_tool_call_normalized(self) -> None:
        p = self._provider(
            _anthropic_sdk_response([_tool_use_block("search", {"q": "ai"})], stop="tool_use")
        )
        resp = p.complete("claude-sonnet-5", REQ)
        assert resp.stop_reason == "tool_use"
        [call] = resp.tool_calls
        assert (call.name, call.arguments) == ("search", {"q": "ai"})

    def test_structured_output_forces_tool(self) -> None:
        schema = {"type": "object", "properties": {"score": {"type": "number"}}}
        p = self._provider(
            _anthropic_sdk_response(
                [_tool_use_block("emit_structured_output", {"score": 8})], stop="tool_use"
            )
        )
        resp = p.complete("claude-sonnet-5", REQ.model_copy(update={"json_schema": schema}))
        # 结构化输出归一为文本 JSON，工具调用列表为空
        assert resp.tool_calls == []
        assert '"score": 8' in resp.text
        kwargs = p._client.messages.create.call_args.kwargs
        assert kwargs["tool_choice"] == {"type": "tool", "name": "emit_structured_output"}

    def test_tool_result_becomes_user_message(self) -> None:
        msgs = _to_anthropic_messages(
            [
                UserMessage(content="hi"),
                AssistantMessage(
                    content=[ToolCallBlock(id="tu_1", name="search", arguments={"q": "x"})]
                ),
                ToolResultMessage(tool_call_id="tu_1", content="result", is_error=False),
            ]
        )
        assert msgs[2]["role"] == "user"
        assert msgs[2]["content"][0]["type"] == "tool_result"
        assert msgs[2]["content"][0]["tool_use_id"] == "tu_1"


def _openai_sdk_response(
    text: str | None, tool_calls: list[Any] | None = None, finish: str = "stop"
) -> MagicMock:
    resp = MagicMock()
    choice = MagicMock()
    choice.message.content = text
    choice.message.tool_calls = tool_calls
    choice.finish_reason = finish
    resp.choices = [choice]
    resp.usage.prompt_tokens = 10
    resp.usage.completion_tokens = 5
    resp.model = "deepseek-chat"
    resp.id = "cmpl_1"
    return resp


def _openai_tool_call(name: str, args_json: str) -> MagicMock:
    tc = MagicMock()
    tc.id = "call_1"
    tc.function.name = name
    tc.function.arguments = args_json
    return tc


class TestOpenAINormalization:
    def _provider(self, sdk_resp: MagicMock) -> OpenAICompatProvider:
        p = OpenAICompatProvider.__new__(OpenAICompatProvider)
        p.name = "deepseek"
        p._client = MagicMock()
        p._client.chat.completions.create.return_value = sdk_resp
        return p

    def test_text_response(self) -> None:
        p = self._provider(_openai_sdk_response("hello"))
        resp = p.complete("deepseek-chat", REQ)
        assert resp.text == "hello"
        assert resp.stop_reason == "end_turn"
        assert resp.usage.input_tokens == 10

    def test_tool_call_arguments_parsed_from_json_string(self) -> None:
        p = self._provider(
            _openai_sdk_response(None, [_openai_tool_call("search", '{"q": "ai"}')], "tool_calls")
        )
        resp = p.complete("deepseek-chat", REQ)
        assert resp.stop_reason == "tool_use"
        [call] = resp.tool_calls
        assert (call.name, call.arguments) == ("search", {"q": "ai"})

    def test_json_object_mode_uses_generic_response_format(self) -> None:
        p = self._provider(_openai_sdk_response('{"ok": true}'))
        p.json_mode = "object"
        p.complete("qwen", REQ.model_copy(update={"json_schema": {"type": "object"}}))
        kwargs = p._client.chat.completions.create.call_args.kwargs
        assert kwargs["response_format"] == {"type": "json_object"}

    def test_tool_mode_forces_structured_tool_and_returns_json_text(self) -> None:
        p = self._provider(
            _openai_sdk_response(
                None,
                [_openai_tool_call("emit_structured_output", '{"ok": true}')],
                "tool_calls",
            )
        )
        p.json_mode = "tool"
        resp = p.complete("qwen", REQ.model_copy(update={"json_schema": {"type": "object"}}))
        kwargs = p._client.chat.completions.create.call_args.kwargs
        assert kwargs["tools"][0]["function"]["name"] == "emit_structured_output"
        assert kwargs["tool_choice"] == {
            "type": "function",
            "function": {"name": "emit_structured_output"},
        }
        assert resp.tool_calls == []
        assert resp.text == '{"ok": true}'

    def test_system_goes_first_and_tool_result_is_tool_role(self) -> None:
        msgs = _to_openai_messages(
            [
                UserMessage(content="hi"),
                AssistantMessage(
                    content=[
                        TextBlock(text="let me search"),
                        ToolCallBlock(id="call_1", name="search", arguments={"q": "x"}),
                    ]
                ),
                ToolResultMessage(tool_call_id="call_1", content="result"),
            ],
            system="be brief",
        )
        assert msgs[0] == {"role": "system", "content": "be brief"}
        assert msgs[2]["tool_calls"][0]["function"]["arguments"] == '{"q": "x"}'
        assert msgs[3] == {"role": "tool", "tool_call_id": "call_1", "content": "result"}


class TestCrossProviderConsistency:
    """同一逻辑响应经两个 adapter 归一后完全一致 —— Provider 抽象的核心承诺。"""

    def test_same_normalized_tool_call(self) -> None:
        a = AnthropicProvider.__new__(AnthropicProvider)
        a._client = MagicMock()
        a._client.messages.create.return_value = _anthropic_sdk_response(
            [_tool_use_block("search", {"q": "ai"})], stop="tool_use"
        )
        o = OpenAICompatProvider.__new__(OpenAICompatProvider)
        o.name = "deepseek"
        o._client = MagicMock()
        o._client.chat.completions.create.return_value = _openai_sdk_response(
            None, [_openai_tool_call("search", '{"q": "ai"}')], "tool_calls"
        )

        ra = a.complete("m", REQ)
        ro = o.complete("m", REQ)
        assert ra.stop_reason == ro.stop_reason == "tool_use"
        assert ra.tool_calls[0].name == ro.tool_calls[0].name
        assert ra.tool_calls[0].arguments == ro.tool_calls[0].arguments


def test_router_allows_provider_model_environment_override(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    config = tmp_path / "routing.yaml"
    config.write_text(
        """
providers:
  anthropic:
    protocol: anthropic
    api_key_env: TEST_ANTHROPIC_KEY
    model_env: TEST_ANTHROPIC_MODEL
tasks:
  topic_scout: anthropic/example-model
""",
        encoding="utf-8",
    )
    monkeypatch.setenv("TEST_ANTHROPIC_KEY", "test-key")
    monkeypatch.setenv("TEST_ANTHROPIC_MODEL", "claude-opus-4-8")

    _, model = ModelRouter.from_yaml(config).resolve("topic_scout")

    assert model == "claude-opus-4-8"
