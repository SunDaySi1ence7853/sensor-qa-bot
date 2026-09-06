"""
pytest 公共 fixture。

要点：
- autouse 清空 lru_cache：防止 get_config / get_embeddings 在测试间串味
- clean_env 会同时禁用 load_dotenv，防止 .env 污染测试
"""

from __future__ import annotations

import os

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
    for key in list(os.environ.keys()):
        if key.startswith((
            "DEEPSEEK_", "EMBEDDING_", "LOCAL_", "VECTORSTORE_",
            "RETRIEVE_", "CHUNK_", "MEMORY_",
        )):
            monkeypatch.delenv(key, raising=False)

    # 从源头禁用：patch dotenv 库本身，
    # 这样 reload(src.config) 里的 `from dotenv import load_dotenv`
    # 拿到的也是空操作版本。
    import dotenv
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **kw: False)

    return monkeypatch


@pytest.fixture
def valid_env(clean_env):
    """预设一套合法的环境变量，供不想从零设置的测试用。"""
    clean_env.setenv("DEEPSEEK_API_KEY", "sk-test-key")
    clean_env.setenv("EMBEDDING_PROVIDER", "local")
    return clean_env