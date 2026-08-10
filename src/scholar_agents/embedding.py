"""Embedding：支持两种后端，由环境变量切换（ADR-005 修订版）。

默认使用**硅基流动 SiliconFlow API**（BAAI/bge-m3，1024 维，OpenAI 兼容协议），
不占 VPS 内存，采集不会被 embedding 拖死。

本机 Ollama 路径保留（设 EMBED_BACKEND=ollama），作为离线/调试备用。

维度与模型是"数据契约"：换模型/换维度必须走"新列 + 回填 + 切换"，不可原地
替换，否则库内既有向量全部失效。当前契约：1024 维。
"""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass

import structlog

log = structlog.get_logger()

# ── 与库表 vector(1024) 绑定，改动需同步迁移 ──────────────────────────────
EMBED_DIM = 1024

# ── 后端选择（默认 siliconflow）────────────────────────────────────────────
EMBED_BACKEND = os.environ.get("EMBED_BACKEND", "siliconflow")

# ── SiliconFlow（OpenAI 兼容）─────────────────────────────────────────────
SF_BASE_URL = os.environ.get("SF_BASE_URL", "https://api.siliconflow.cn/v1")
SF_API_KEY = os.environ.get("SF_API_KEY", "")
SF_MODEL = os.environ.get("SF_EMBED_MODEL", "BAAI/bge-m3")

# ── Ollama（本机备用）────────────────────────────────────────────────────
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434")
OLLAMA_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "qwen3-embedding:0.6b")
OLLAMA_KEEP_ALIVE = os.environ.get("OLLAMA_KEEP_ALIVE", "10m")

# ── 共用截断上限：embedding 只需语义指纹，不需要全文 ─────────────────────
MAX_CHARS = int(os.environ.get("EMBED_MAX_CHARS", "600"))


class EmbeddingError(RuntimeError):
    """embedding 服务不可用或返回非预期向量 —— 契约错误，不静默兜底。"""


def _l2_normalize(vec: list[float]) -> list[float]:
    norm = math.sqrt(sum(x * x for x in vec))
    if norm == 0.0:
        raise EmbeddingError("embedding norm is zero; degenerate vector refused")
    return [x / norm for x in vec]


def _embed_siliconflow(text: str) -> list[float]:
    """via OpenAI-compatible API（SiliconFlow / 任何兼容端点）。"""
    if not SF_API_KEY:
        raise EmbeddingError("SF_API_KEY is not set; set EMBED_BACKEND=ollama to use local Ollama")
    try:
        from openai import OpenAI  # openai 已在 agents 依赖里
        client = OpenAI(api_key=SF_API_KEY, base_url=SF_BASE_URL)
        resp = client.embeddings.create(model=SF_MODEL, input=text[:MAX_CHARS])
        vec = resp.data[0].embedding
    except Exception as exc:
        raise EmbeddingError(f"siliconflow embed failed: {exc}") from exc
    if len(vec) < EMBED_DIM:
        raise EmbeddingError(
            f"model returned {len(vec)} dims, need >= {EMBED_DIM}; "
            "update SF_EMBED_MODEL or EMBED_DIM together with DB migration"
        )
    raw = vec[:EMBED_DIM]
    # bge-m3 通常已归一化，但仍做一次确保（换模型时不需要改这里）
    norm = math.sqrt(sum(x * x for x in raw))
    return [x / norm for x in raw] if abs(norm - 1.0) > 1e-5 else raw


def _embed_ollama(text: str) -> list[float]:
    """via 本机 Ollama（离线/调试用，生产建议用 siliconflow）。"""
    import httpx
    payload = {"model": OLLAMA_MODEL, "input": text[:MAX_CHARS], "keep_alive": OLLAMA_KEEP_ALIVE}
    try:
        resp = httpx.post(f"{OLLAMA_HOST}/api/embed", json=payload, timeout=120.0)
        resp.raise_for_status()
        data = resp.json()
    except Exception as exc:
        raise EmbeddingError(f"ollama request failed: {exc}") from exc
    vectors = data.get("embeddings") or []
    if not vectors or not isinstance(vectors[0], list):
        raise EmbeddingError(f"ollama returned no embedding: keys={list(data)}")
    raw = vectors[0]
    if len(raw) < EMBED_DIM:
        raise EmbeddingError(
            f"model {OLLAMA_MODEL} returned {len(raw)} dims, need >= {EMBED_DIM}"
        )
    return _l2_normalize(raw[:EMBED_DIM])


def embed(text: str) -> list[float]:
    """返回 EMBED_DIM 维、L2 归一化的向量。后端由 EMBED_BACKEND 控制。"""
    if EMBED_BACKEND == "ollama":
        return _embed_ollama(text)
    return _embed_siliconflow(text)


def cosine(a: list[float], b: list[float]) -> float:
    if len(a) != len(b):
        raise ValueError(f"dimension mismatch: {len(a)} vs {len(b)}")
    return sum(x * y for x, y in zip(a, b, strict=True))
