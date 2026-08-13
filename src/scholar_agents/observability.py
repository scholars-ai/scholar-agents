"""LLM 调用级 Langfuse 留痕；未配置 Langfuse 时安全降级为 no-op。"""

from __future__ import annotations

import os
from datetime import UTC, datetime
from typing import Any
from uuid import UUID, uuid4

import httpx
import structlog

from scholar_agents.providers.base import ChatRequest, ChatResponse, ModelProvider

log = structlog.get_logger()


class TraceRecorder:
    def __init__(self, *, trace_id: str | None = None, name: str = "agent-job") -> None:
        self.trace_id = trace_id or str(uuid4())
        self.name = name
        self._host = os.environ.get("LANGFUSE_HOST", "").rstrip("/")
        self._public_key = os.environ.get("LANGFUSE_PUBLIC_KEY")
        self._secret_key = os.environ.get("LANGFUSE_SECRET_KEY")

    @property
    def enabled(self) -> bool:
        return bool(self._host and self._public_key and self._secret_key)

    def generation(
        self,
        *,
        model: str,
        request: ChatRequest,
        response: ChatResponse,
        observation_name: str,
        prompt_version: str | None,
    ) -> None:
        if not self.enabled:
            return
        now = datetime.now(UTC).isoformat()
        input_payload = _request_payload(request)
        metadata: dict[str, Any] = {}
        if prompt_version:
            metadata["promptVersion"] = prompt_version
        if response.raw:
            metadata["providerResponse"] = response.raw
        body: dict[str, Any] = {
            "id": str(uuid4()),
            "traceId": self.trace_id,
            "name": observation_name,
            "startTime": now,
            "endTime": now,
            "completionStartTime": now,
            "model": response.model or model,
            "input": input_payload,
            "output": response.text,
            "usage": {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
                "total": response.usage.input_tokens + response.usage.output_tokens,
                "unit": "TOKENS",
            },
            "usageDetails": {
                "input": response.usage.input_tokens,
                "output": response.usage.output_tokens,
                "total": response.usage.input_tokens + response.usage.output_tokens,
            },
        }
        if metadata:
            body["metadata"] = metadata
        self._send("generation-create", body)

    def trace(self, *, input_payload: Any = None, metadata: dict[str, Any] | None = None) -> None:
        if not self.enabled:
            return
        body: dict[str, Any] = {"id": self.trace_id, "timestamp": datetime.now(UTC).isoformat()}
        if input_payload is not None:
            body["input"] = input_payload
        if metadata:
            body["metadata"] = metadata
        self._send("trace-create", body)

    def score(self, *, name: str, value: float, comment: str | None = None) -> None:
        if not self.enabled:
            return
        body: dict[str, Any] = {
            "id": str(uuid4()),
            "traceId": self.trace_id,
            "name": name,
            "value": value,
        }
        if comment:
            body["comment"] = comment
        self._send("score-create", body)

    def _send(self, event_type: str, body: dict[str, Any]) -> None:
        try:
            event = {
                "id": str(uuid4()),
                "type": event_type,
                "timestamp": datetime.now(UTC).isoformat(),
                "body": body,
            }
            response = httpx.post(
                f"{self._host}/api/public/ingestion",
                json={"batch": [event]},
                auth=(self._public_key or "", self._secret_key or ""),
                timeout=10,
            )
            response.raise_for_status()
        except Exception as exc:  # noqa: BLE001 — 观测系统故障不应阻断业务 job
            log.warning("langfuse_write_failed", trace_id=self.trace_id, error=str(exc))


class ObservedProvider:
    """在不改变 ModelProvider 协议的情况下记录每次 complete。"""

    def __init__(
        self,
        provider: ModelProvider,
        recorder: TraceRecorder,
        *,
        observation_name: str,
        prompt_version: str | None,
    ) -> None:
        self._provider = provider
        self._recorder = recorder
        self._observation_name = observation_name
        self._prompt_version = prompt_version
        self.name = provider.name

    def complete(self, model: str, req: ChatRequest) -> ChatResponse:
        response = self._provider.complete(model, req)
        self._recorder.generation(
            model=model,
            request=req,
            response=response,
            observation_name=self._observation_name,
            prompt_version=self._prompt_version,
        )
        return response


def _request_payload(request: ChatRequest) -> dict[str, Any]:
    return {
        "system": request.system,
        "messages": [message.model_dump(mode="json") for message in request.messages],
        "json_schema": request.json_schema,
        "max_tokens": request.max_tokens,
    }


def new_trace_id() -> str:
    return str(UUID(int=uuid4().int))
