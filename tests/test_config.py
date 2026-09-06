"""
config.py 单元测试。

用 monkeypatch 精确控制环境变量，避开真实 .env 干扰。
"""

from __future__ import annotations

import importlib

import pytest


def _reload_config():
    """重新导入 config 模块以生效环境变量。"""
    import src.config as m
    importlib.reload(m)
    m.get_config.cache_clear()
    return m


def test_normal_config_loads(clean_env):
    clean_env.setenv("DEEPSEEK_API_KEY", "sk-test")
    clean_env.setenv("EMBEDDING_PROVIDER", "local")
    m = _reload_config()
    cfg = m.get_config()
    assert cfg.deepseek_api_key == "sk-test"
    assert cfg.embedding_provider == "local"


def test_missing_api_key_does_not_raise_on_get_config(clean_env):
    """API key 缺失时 get_config() 不报错。"""
    clean_env.setenv("EMBEDDING_PROVIDER", "local")
    m = _reload_config()
    cfg = m.get_config()
    assert cfg.deepseek_api_key == ""


def test_default_embedding_provider_is_local(clean_env):
    clean_env.setenv("DEEPSEEK_API_KEY", "sk-test")
    m = _reload_config()
    cfg = m.get_config()
    assert cfg.embedding_provider == "local"


def test_illegal_embedding_provider_raises(clean_env):
    clean_env.setenv("DEEPSEEK_API_KEY", "sk-test")
    clean_env.setenv("EMBEDDING_PROVIDER", "wtf")
    m = _reload_config()
    with pytest.raises(ValueError, match="EMBEDDING_PROVIDER"):
        m.get_config()


def test_price_zero_is_allowed(clean_env):
    clean_env.setenv("DEEPSEEK_API_KEY", "sk-test")
    clean_env.setenv("DEEPSEEK_INPUT_PRICE_PER_1M", "0")
    clean_env.setenv("DEEPSEEK_OUTPUT_PRICE_PER_1M", "0")
    m = _reload_config()
    cfg = m.get_config()
    assert cfg.input_price_per_1m == 0.0
    assert cfg.output_price_per_1m == 0.0


def test_negative_input_price_raises(clean_env):
    clean_env.setenv("DEEPSEEK_API_KEY", "sk-test")
    clean_env.setenv("DEEPSEEK_INPUT_PRICE_PER_1M", "-1.0")
    m = _reload_config()
    with pytest.raises(ValueError, match="DEEPSEEK_INPUT_PRICE_PER_1M"):
        m.get_config()


def test_top_k_must_be_positive(clean_env):
    clean_env.setenv("DEEPSEEK_API_KEY", "sk-test")
    clean_env.setenv("RETRIEVE_TOP_K", "0")
    m = _reload_config()
    with pytest.raises(ValueError, match="RETRIEVE_TOP_K"):
        m.get_config()