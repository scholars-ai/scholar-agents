"""按任务路由模型（ADR-002）：task → provider/model 由 config/model_routing.yaml 决定。

切换模型 = 改配置，代码零改动。provider 的密钥从环境变量读取（绝不入库）。
"""

from __future__ import annotations

import os
from pathlib import Path

import yaml
from pydantic import BaseModel

from scholar_agents.providers.anthropic_provider import AnthropicProvider
from scholar_agents.providers.base import ChatRequest, ChatResponse, ModelProvider
from scholar_agents.providers.openai_provider import OpenAICompatProvider


class ProviderConfig(BaseModel):
    protocol: str  # anthropic | openai
    base_url: str | None = None
    api_key_env: str  # 环境变量名，如 ANTHROPIC_API_KEY / DEEPSEEK_API_KEY


class RoutingConfig(BaseModel):
    providers: dict[str, ProviderConfig]
    # task 名 → "provider/model"，如 topic_judge: "anthropic/claude-sonnet-5"
    tasks: dict[str, str]


class ModelRouter:
    def __init__(self, config: RoutingConfig) -> None:
        self._config = config
        self._instances: dict[str, ModelProvider] = {}

    @classmethod
    def from_yaml(cls, path: Path) -> ModelRouter:
        data = yaml.safe_load(path.read_text())
        return cls(RoutingConfig.model_validate(data))

    def resolve(self, task: str) -> tuple[ModelProvider, str]:
        """task 名 → (provider 实例, model 名)。未注册的 task 直接抛错，不静默兜底。"""
        route = self._config.tasks.get(task)
        if route is None:
            raise KeyError(f"no model route for task {task!r}; add it to model_routing.yaml")
        provider_name, _, model = route.partition("/")
        if not model:
            raise ValueError(f"bad route for task {task!r}: {route!r} (want 'provider/model')")
        return self._provider(provider_name), model

    def complete(self, task: str, req: ChatRequest) -> ChatResponse:
        provider, model = self.resolve(task)
        return provider.complete(model, req)

    def _provider(self, name: str) -> ModelProvider:
        if name in self._instances:
            return self._instances[name]
        cfg = self._config.providers.get(name)
        if cfg is None:
            raise KeyError(f"unknown provider {name!r} in model_routing.yaml")
        api_key = os.environ.get(cfg.api_key_env)
        if not api_key:
            raise RuntimeError(f"env {cfg.api_key_env} is required for provider {name!r}")

        provider: ModelProvider
        if cfg.protocol == "anthropic":
            provider = AnthropicProvider(api_key=api_key, base_url=cfg.base_url)
        elif cfg.protocol == "openai":
            provider = OpenAICompatProvider(name=name, api_key=api_key, base_url=cfg.base_url)
        else:
            raise ValueError(f"unknown protocol {cfg.protocol!r} for provider {name!r}")
        self._instances[name] = provider
        return provider
