"""
pytest 公共 fixture。

要点：
- autouse 清空 lru_cache：防止 get_config / get_embeddings 在测试间串味
- monkeypatch 会话结束自动还原环境变量
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _clear_caches():
    """每个测试前后清空所有 lru_cache 的模块级缓存。"""
    from src import config as config_mod
    from src import embeddings as embeddings_mod

    for mod in (config_mod, embeddings_mod):
        for name in dir(mod):
            obj = getattr(mod, name)
            cache_clear = getattr(obj, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()

    yield

    for mod in (config_mod, embeddings_mod):
        for name in dir(mod):
            obj = getattr(mod, name)
            cache_clear = getattr(obj, "cache_clear", None)
            if callable(cache_clear):
                cache_clear()


@pytest.fixture
def clean_env(monkeypatch):
    """
    清掉所有 DEEPSEEK_* / EMBEDDING_* 环境变量，
    保证 config.py 从零开始读。
    """
    for key in list(__import__("os").environ.keys()):
        if key.startswith(("DEEPSEEK_", "EMBEDDING_", "LOCAL_", "VECTORSTORE_",
                           "RETRIEVE_", "CHUNK_", "MEMORY_")):
            monkeypatch.delenv(key, raising=False)
    return monkeypatch
@pytest.fixture
def valid_env(clean_env):
    """
    预设一套合法的环境变量，供不想从零设置的测试用。
    """
    clean_env.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    clean_env.setenv("EMBEDDING_PROVIDER", "local")
    return clean_env