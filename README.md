# scholar-agents

scholars-ai 的 Agent 运行时（Python 3.12）：**自研轻量 runtime + 双协议 ModelProvider**（[ADR-002](https://github.com/scholars-ai/spec/blob/main/adr/ADR-002-self-built-runtime-dual-provider.md)）。

纯 worker 纪律：消费 pgmq job → 跑 Agent → 写结果表。**绝不改业务状态机**（那是 [scholar-core](https://github.com/scholars-ai/scholar-core) 的唯一职责）。

## 结构

```
src/scholar_agents/
  providers/          ModelProvider 抽象与双协议 adapter
    base.py             ChatRequest/ChatResponse/归一化消息模型（协议无关）
    anthropic_provider  Anthropic 协议（tool_use block / 强制工具做结构化输出）
    openai_provider     OpenAI 兼容协议，一个 adapter 通吃 DeepSeek/Qwen/Kimi/GLM/...
    router.py           task → provider/model 路由（config/model_routing.yaml）
  runtime/
    loop.py             agent loop：while + 工具调度（工具失败回传模型，不炸 loop）
    structured.py       结构化输出：schema 校验失败带反馈重试，耗尽即抛
  worker/consumer.py  pgmq 消费循环（有限重试 + visibility timeout 语义）
  agents/             六类 Agent（M1 起逐个落地）
config/model_routing.yaml   模型路由：切换模型改配置，代码零改动
```

## 开发

```bash
uv sync
uv run pytest        # 15 个测试：双协议归一化一致性 + loop/structured 行为
uv run ruff check . && uv run mypy   # strict
```

## 设计约定

- **provider 是纯 adapter**：不做重试/预算/日志；runtime 汇总每个 LLM step 的 usage，worker 负责有限 job 重试。
- **不在应用层做 token 预算熔断**：供应商 API key 负责额度限制；每个 LLM step 的输入/输出 token、模型和成本由 Langfuse trace 记录，并汇总到 `agent_runs`。
- **协议差异只存在于 adapter 内**：runtime 之上看到的永远是归一化的 ChatRequest/ChatResponse。
- **契约错误必须炸出来**：未知 task 路由、未知工具名、结构化输出重试耗尽，一律抛异常。
- **重试必须有边界**：结构化输出最多 3 次；quota、余额、认证和模型不存在等永久错误不重试；临时 job 错误最多执行 3 次。
- 密钥全部走环境变量（`.env.example`），绝不入库。
