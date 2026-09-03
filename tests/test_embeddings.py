"""
测试 src/embeddings.py

覆盖：
1. local 正常创建（mock HuggingFaceEmbeddings）
2. deepseek 无 key 时报错
3. deepseek base_url 不带 /v1：自动补
4. deepseek base_url 已带 /v1：不重复补
5. deepseek base_url 带尾斜杠：正确处理
6. local 模型加载失败：给友好提示
"""

from unittest.mock import MagicMock

import pytest

from src import embeddings as emb_module


class TestLocalEmbeddings:
    def test_local_provider_creates_hf_embeddings(self, valid_env, monkeypatch):
        fake_instance = MagicMock(name="HFEmbeddingsInstance")
        fake_cls = MagicMock(return_value=fake_instance)

        monkeypatch.setattr(
            "langchain_huggingface.HuggingFaceEmbeddings", fake_cls
        )

        result = emb_module.get_embeddings()

        assert result is fake_instance
        fake_cls.assert_called_once()
        _, kwargs = fake_cls.call_args
        assert kwargs["model_name"] == "shibing624/text2vec-base-chinese"
        assert kwargs["encode_kwargs"] == {"normalize_embeddings": True}

    def test_local_model_load_failure_gives_friendly_error(
        self, valid_env, monkeypatch
    ):
        def boom(*args, **kwargs):
            raise OSError("模型下载失败")

        monkeypatch.setattr(
            "langchain_huggingface.HuggingFaceEmbeddings", boom
        )

        with pytest.raises(RuntimeError, match="本地 embedding 模型加载失败"):
            emb_module.get_embeddings()


class TestDeepseekEmbeddings:
    def _mock_openai_embeddings(self, monkeypatch):
        fake_instance = MagicMock(name="OpenAIEmbeddingsInstance")
        fake_cls = MagicMock(return_value=fake_instance)
        monkeypatch.setattr(
            "langchain_openai.OpenAIEmbeddings", fake_cls
        )
        return fake_cls, fake_instance

    def test_deepseek_without_key_raises(self, clean_env):
        clean_env.setenv("EMBEDDING_PROVIDER", "deepseek")
        # 不设 DEEPSEEK_API_KEY
        with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
            emb_module.get_embeddings()

    def test_base_url_without_v1_appends_v1(self, valid_env, monkeypatch):
        valid_env.setenv("EMBEDDING_PROVIDER", "deepseek")
        valid_env.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        fake_cls, _ = self._mock_openai_embeddings(monkeypatch)

        emb_module.get_embeddings()

        _, kwargs = fake_cls.call_args
        assert kwargs["base_url"] == "https://api.deepseek.com/v1"

    def test_base_url_with_v1_not_duplicated(self, valid_env, monkeypatch):
        valid_env.setenv("EMBEDDING_PROVIDER", "deepseek")
        valid_env.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
        fake_cls, _ = self._mock_openai_embeddings(monkeypatch)

        emb_module.get_embeddings()

        _, kwargs = fake_cls.call_args
        assert kwargs["base_url"] == "https://api.deepseek.com/v1"

    def test_base_url_with_trailing_slash(self, valid_env, monkeypatch):
        valid_env.setenv("EMBEDDING_PROVIDER", "deepseek")
        valid_env.setenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/")
        fake_cls, _ = self._mock_openai_embeddings(monkeypatch)

        emb_module.get_embeddings()

        _, kwargs = fake_cls.call_args
        assert kwargs["base_url"] == "https://api.deepseek.com/v1"

    def test_deepseek_passes_api_key(self, valid_env, monkeypatch):
        valid_env.setenv("EMBEDDING_PROVIDER", "deepseek")
        fake_cls, _ = self._mock_openai_embeddings(monkeypatch)

        emb_module.get_embeddings()

        _, kwargs = fake_cls.call_args
        assert kwargs["api_key"] == "sk-test-key"