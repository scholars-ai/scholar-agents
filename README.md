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
  agents/             Scout / Judge / WriterOrchestrator 等业务 Agent
  writing/            Platform Profile 加载与确定性 Formatter
config/model_routing.yaml   模型路由：切换模型改配置，代码零改动
```

## 开发

```bash
uv sync
uv run pytest        # provider/runtime、采集、Topic 与 Writer 行为测试
uv run ruff check . && uv run mypy   # strict
```

## 设计约定

- **provider 是纯 adapter**：不做重试/预算/日志；runtime 汇总每个 LLM step 的 usage，worker 负责有限 job 重试。
- **不在应用层做 token 预算熔断**：供应商 API key 负责额度限制；每个 LLM step 的输入/输出 token、模型和成本由 Langfuse trace 记录，并汇总到 `agent_runs`。
- **协议差异只存在于 adapter 内**：runtime 之上看到的永远是归一化的 ChatRequest/ChatResponse。
- **契约错误必须炸出来**：未知 task 路由、未知工具名、结构化输出重试耗尽，一律抛异常。
- **重试必须有边界**：结构化输出最多 3 次；quota、余额、认证和模型不存在等永久错误不重试；临时 job 错误最多执行 3 次。
- **定时 Scout 批次有边界**：普通调度默认最多处理 20 条新素材，在单个 LLM job 可控的前提下支撑 M1 日均产量；手动定向投喂按指定 `rawItemIds` 处理，不受该默认批次限制。
- **Writer 同构不同魂**：Outliner → Drafter → SelfCritic → Formatter 共用一套骨架，平台差异只从 `scholar-shared/profiles/*.yaml` 注入；Agents 只写 `articles(draft)`，状态推进和 `article_evaluate` 投递仍由 Core 负责。
- **LLM 请求必须有超时**：默认单次请求 120 秒，可由 `LLM_REQUEST_TIMEOUT_SECONDS` 调整，避免超过 pgmq visibility timeout 后重复领取。
- 密钥全部走环境变量（`.env.example`），绝不入库。
