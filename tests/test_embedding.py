"""embedding 契约测试：维度、归一化、失败即抛（不静默兜底）。

全部 mock，不打真实 API、不消耗额度。
"""

from __future__ import annotations

import math
from unittest.mock import MagicMock, patch

import pytest

from scholar_agents import embedding as emb


def _fake_sf_response(dims: int, scale: float = 1.0) -> MagicMock:
    """构造一个 OpenAI 兼容的 embeddings 响应。"""
    resp = MagicMock()
    item = MagicMock()
    # 造一个非归一化向量，验证我们会归一化它
    item.embedding = [scale * (i + 1) / dims for i in range(dims)]
    resp.data = [item]
    return resp


class TestSiliconFlowBackend:
    def _patch_client(self, resp: MagicMock) -> MagicMock:
        client = MagicMock()
        client.embeddings.create.return_value = resp
        return client

    def test_returns_normalized_1024_dims(self) -> None:
        client = self._patch_client(_fake_sf_response(1024, scale=7.0))
        with patch.object(emb, "SF_API_KEY", "sk-test"), \
             patch("openai.OpenAI", return_value=client):
            vec = emb._embed_siliconflow("hello")
        assert len(vec) == emb.EMBED_DIM == 1024
        assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, abs_tol=1e-6)

    def test_truncates_longer_vectors_to_contract_dim(self) -> None:
        # 模型返回 2560 维（如 Qwen3-Embedding-4B 默认）→ 截断到 1024
        client = self._patch_client(_fake_sf_response(2560))
        with patch.object(emb, "SF_API_KEY", "sk-test"), \
             patch("openai.OpenAI", return_value=client):
            vec = emb._embed_siliconflow("hello")
        assert len(vec) == 1024

    def test_raises_when_dims_too_few(self) -> None:
        # 返回 768 维（如 bge-large）→ 不足契约维度，必须炸而不是零填充
        client = self._patch_client(_fake_sf_response(768))
        with patch.object(emb, "SF_API_KEY", "sk-test"), \
             patch("openai.OpenAI", return_value=client), \
             pytest.raises(emb.EmbeddingError, match="768 dims"):
            emb._embed_siliconflow("hello")

    def test_raises_without_api_key(self) -> None:
        with patch.object(emb, "SF_API_KEY", ""), \
             pytest.raises(emb.EmbeddingError, match="SF_API_KEY"):
            emb._embed_siliconflow("hello")

    def test_truncates_input_to_max_chars(self) -> None:
        client = self._patch_client(_fake_sf_response(1024))
        with patch.object(emb, "SF_API_KEY", "sk-test"), \
             patch.object(emb, "MAX_CHARS", 10), \
             patch("openai.OpenAI", return_value=client):
            emb._embed_siliconflow("x" * 500)
        sent = client.embeddings.create.call_args.kwargs["input"]
        assert len(sent) == 10, "过长输入必须截断，否则 CPU/额度成本失控"

    def test_api_error_becomes_embedding_error(self) -> None:
        client = MagicMock()
        client.embeddings.create.side_effect = RuntimeError("upstream 503")
        with patch.object(emb, "SF_API_KEY", "sk-test"), \
             patch("openai.OpenAI", return_value=client), \
             pytest.raises(emb.EmbeddingError, match="siliconflow embed failed"):
            emb._embed_siliconflow("hello")


class TestOllamaBackend:
    def test_normalizes_after_mrl_truncation(self) -> None:
        """Ollama 的 qwen3-embedding 原生 2560 维且已归一化；
        截断前 1024 维后范数不再为 1（实测 ~0.63），必须重新归一化。"""
        raw = [1.0 / math.sqrt(2560)] * 2560  # 归一化的 2560 维向量
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"embeddings": [raw]}
        with patch("httpx.post", return_value=resp):
            vec = emb._embed_ollama("hello")
        assert len(vec) == 1024
        assert math.isclose(math.sqrt(sum(x * x for x in vec)), 1.0, abs_tol=1e-6)

    def test_zero_vector_refused(self) -> None:
        resp = MagicMock()
        resp.raise_for_status = MagicMock()
        resp.json.return_value = {"embeddings": [[0.0] * 1024]}
        with patch("httpx.post", return_value=resp), \
             pytest.raises(emb.EmbeddingError, match="norm is zero"):
            emb._embed_ollama("hello")


class TestBackendSwitch:
    def test_default_backend_is_siliconflow(self) -> None:
        assert emb.EMBED_BACKEND == "siliconflow"

    def test_env_switches_to_ollama(self) -> None:
        called = {}

        def fake_ollama(text: str) -> list[float]:
            called["ollama"] = True
            return [1.0] + [0.0] * 1023

        with patch.object(emb, "EMBED_BACKEND", "ollama"), \
             patch.object(emb, "_embed_ollama", fake_ollama):
            emb.embed("hello")
        assert called.get("ollama")


class TestCosine:
    def test_identical_vectors(self) -> None:
        v = [0.6, 0.8] + [0.0] * 1022
        assert math.isclose(emb.cosine(v, v), 1.0, abs_tol=1e-9)

    def test_orthogonal_vectors(self) -> None:
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert math.isclose(emb.cosine(a, b), 0.0, abs_tol=1e-9)

    def test_dimension_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="dimension mismatch"):
            emb.cosine([1.0, 0.0], [1.0, 0.0, 0.0])
