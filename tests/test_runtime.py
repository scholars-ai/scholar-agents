"""runtime 测试：loop 的工具调度与 structured 的校验重试。用 FakeProvider，不打真实 API。"""

from __future__ import annotations

import json
from typing import Any

import pytest

from scholar_agents.providers.base import (
    ChatRequest,
    ChatResponse,
    ContentBlock,
    TextBlock,
    ToolCallBlock,
    ToolResultMessage,
    ToolSpec,
    Usage,
)
from scholar_agents.runtime.loop import LoopResult, Tool, ToolNotFoundError, run_loop
from scholar_agents.runtime.structured import StructuredOutputError, complete_structured


class FakeProvider:
    """按脚本顺序吐响应，并记录收到的请求。"""

    name = "fake"

    def __init__(self, script: list[ChatResponse]) -> None:
        self._script = list(script)
        self.requests: list[ChatRequest] = []

    def complete(self, model: str, req: ChatRequest) -> ChatResponse:
        self.requests.append(req)
        return self._script.pop(0)


def _resp(content: list[ContentBlock], stop: str) -> ChatResponse:
    return ChatResponse(
        content=content, stop_reason=stop, usage=Usage(input_tokens=1, output_tokens=1), model="m"
    )


ECHO_TOOL = Tool(
    spec=ToolSpec(name="echo", description="echo", input_schema={"type": "object"}),
    fn=lambda args: f"echo:{args['x']}",
)


class TestRunLoop:
    def test_tool_roundtrip_then_final_answer(self) -> None:
        provider = FakeProvider(
            [
                _resp([ToolCallBlock(id="1", name="echo", arguments={"x": "hi"})], "tool_use"),
                _resp([TextBlock(text="done")], "end_turn"),
            ]
        )
        result: LoopResult = run_loop(provider, "m", "sys", "go", [ECHO_TOOL])
        assert result.final_text == "done"
        assert result.turns == 2
        # 第二轮请求里带有工具结果
        tool_results = [
            m for m in provider.requests[1].messages if isinstance(m, ToolResultMessage)
        ]
        assert tool_results[0].content == "echo:hi"
        assert not tool_results[0].is_error

    def test_tool_exception_fed_back_as_error(self) -> None:
        def boom(_: dict[str, Any]) -> str:
            raise ValueError("nope")

        provider = FakeProvider(
            [
                _resp([ToolCallBlock(id="1", name="fail", arguments={})], "tool_use"),
                _resp([TextBlock(text="recovered")], "end_turn"),
            ]
        )
        tool = Tool(
            spec=ToolSpec(name="fail", description="", input_schema={"type": "object"}), fn=boom
        )
        result = run_loop(provider, "m", "sys", "go", [tool])
        assert result.final_text == "recovered"
        tool_results = [
            m for m in provider.requests[1].messages if isinstance(m, ToolResultMessage)
        ]
        assert tool_results[0].is_error
        assert "nope" in tool_results[0].content

    def test_unknown_tool_raises(self) -> None:
        provider = FakeProvider(
            [_resp([ToolCallBlock(id="1", name="ghost", arguments={})], "tool_use")]
        )
        with pytest.raises(ToolNotFoundError):
            run_loop(provider, "m", "sys", "go", [ECHO_TOOL])

    def test_max_turns_exhaustion_returns_empty(self) -> None:
        provider = FakeProvider(
            [
                _resp([ToolCallBlock(id=str(i), name="echo", arguments={"x": "a"})], "tool_use")
                for i in range(3)
            ]
        )
        result = run_loop(provider, "m", "sys", "go", [ECHO_TOOL], max_turns=3)
        assert result.final_text == ""
        assert result.turns == 3


SCHEMA = {
    "type": "object",
    "required": ["score"],
    "properties": {"score": {"type": "number", "minimum": 0, "maximum": 10}},
    "additionalProperties": False,
}


class TestCompleteStructured:
    def test_valid_first_try(self) -> None:
        provider = FakeProvider([_resp([TextBlock(text=json.dumps({"score": 8}))], "end_turn")])
        data, usage = complete_structured(provider, "m", "sys", "rate", SCHEMA)
        assert data == {"score": 8}
        assert usage.output_tokens == 1

    def test_retry_on_schema_violation_with_feedback(self) -> None:
        provider = FakeProvider(
            [
                _resp([TextBlock(text=json.dumps({"score": 99}))], "end_turn"),  # 超出 maximum
                _resp([TextBlock(text=json.dumps({"score": 7}))], "end_turn"),
            ]
        )
        data, _ = complete_structured(provider, "m", "sys", "rate", SCHEMA)
        assert data == {"score": 7}
        # 重试请求包含校验错误反馈
        retry_prompt = provider.requests[1].messages[0]
        assert "schema" in retry_prompt.content  # type: ignore[union-attr]

    def test_exhausted_raises(self) -> None:
        provider = FakeProvider(
            [_resp([TextBlock(text="not json")], "end_turn") for _ in range(3)]
        )
        with pytest.raises(StructuredOutputError):
            complete_structured(provider, "m", "sys", "rate", SCHEMA, max_attempts=3)


def test_observed_provider_records_each_generation() -> None:
    from scholar_agents.observability import ObservedProvider, TraceRecorder
    from scholar_agents.providers.base import ChatRequest, ChatResponse, TextBlock, Usage

    class Provider:
        name = "fake"

        def complete(self, model: str, req: ChatRequest) -> ChatResponse:
            del model, req
            return ChatResponse(
                content=[TextBlock(text="ok")],
                stop_reason="end_turn",
                usage=Usage(input_tokens=3, output_tokens=4),
                model="fake-model",
            )

    class Recorder(TraceRecorder):
        def __init__(self) -> None:
            super().__init__(trace_id="trace-r-1")
            self.calls = 0

        def generation(self, **kwargs: object) -> None:
            del kwargs
            self.calls += 1

    recorder = Recorder()
    provider = ObservedProvider(
        Provider(), recorder, observation_name="test", prompt_version="x@v1"
    )
    provider.complete("fake-model", ChatRequest(messages=[]))
    provider.complete("fake-model", ChatRequest(messages=[]))

    assert recorder.calls == 2


def test_trace_recorder_uses_current_langfuse_ingestion_shape(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import scholar_agents.observability as observability

    recorder = observability.TraceRecorder(trace_id="trace-test")
    monkeypatch.setattr(recorder, "_host", "http://langfuse")
    monkeypatch.setattr(recorder, "_public_key", "pk-test")
    monkeypatch.setattr(recorder, "_secret_key", "sk-test")
    calls: list[tuple[str, dict[str, object]]] = []
    monkeypatch.setattr(
        recorder,
        "_send",
        lambda event_type, body: calls.append((event_type, body)),
    )

    recorder.trace(metadata={"jobType": "topic.scout"})
    recorder.generation(
        model="model",
        request=ChatRequest(messages=[]),
        response=_resp([TextBlock(text="output")], "end_turn").model_copy(
            update={"raw": {"id": "provider-response-1"}}
        ),
        observation_name="scout",
        prompt_version="topic-scout@v1",
    )
    recorder.score(name="topic_total_score", value=80.0)

    assert [event_type for event_type, _ in calls] == [
        "trace-create",
        "generation-create",
        "score-create",
    ]
    generation = calls[1][1]
    assert generation["output"] == "output"
    assert generation["usage"] == {
        "input": 1,
        "output": 1,
        "total": 2,
        "unit": "TOKENS",
    }
    assert generation["usageDetails"] == {"input": 1, "output": 1, "total": 2}
    assert generation["completionStartTime"] == generation["startTime"]
    assert generation["metadata"] == {
        "promptVersion": "topic-scout@v1",
        "providerResponse": {"id": "provider-response-1"},
    }


def test_trace_recorder_sends_ingestion_event_timestamp() -> None:
    from scholar_agents.observability import TraceRecorder

    recorder = TraceRecorder(trace_id="trace-test")
    assert recorder.enabled is False


def test_trace_recorder_retries_temporary_ingestion_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    import scholar_agents.observability as observability

    recorder = observability.TraceRecorder(trace_id="trace-retry")
    monkeypatch.setattr(recorder, "_host", "http://langfuse")
    monkeypatch.setattr(recorder, "_public_key", "pk-test")
    monkeypatch.setattr(recorder, "_secret_key", "sk-test")
    attempts = 0

    def post(*args: object, **kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise httpx.ConnectError("temporary outage")

        class Response:
            def raise_for_status(self) -> None:
                return None

        return Response()

    monkeypatch.setattr(observability.httpx, "post", post)

    recorder.score(name="topic_total_score", value=80.0)

    assert attempts == 3


def test_trace_recorder_does_not_retry_ingestion_client_error(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import httpx

    import scholar_agents.observability as observability

    recorder = observability.TraceRecorder(trace_id="trace-client-error")
    monkeypatch.setattr(recorder, "_host", "http://langfuse")
    monkeypatch.setattr(recorder, "_public_key", "pk-test")
    monkeypatch.setattr(recorder, "_secret_key", "sk-test")
    attempts = 0
    request = httpx.Request("POST", "http://langfuse/api/public/ingestion")

    def post(*args: object, **kwargs: object) -> object:
        nonlocal attempts
        attempts += 1
        response = httpx.Response(401, request=request)
        response.raise_for_status()
        return response

    monkeypatch.setattr(observability.httpx, "post", post)

    recorder.score(name="topic_total_score", value=80.0)

    assert attempts == 1
