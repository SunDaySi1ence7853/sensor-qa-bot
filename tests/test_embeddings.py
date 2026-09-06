"""
embeddings.py 单元测试。

关键：全部 patch src.embeddings.XXX（而不是原库路径），
因为 embeddings.py 顶层已经用 from ... import XXX 把名字绑进了自己模块。
"""

from __future__ import annotations

import pytest


def _make_cfg(**overrides):
    """构造一个假的 config 对象，字段和真实 Config 对齐。"""
    class FakeCfg:
        embedding_provider = "local"
        deepseek_api_key = "sk-fake"
        deepseek_base_url = "https://api.deepseek.com"
        deepseek_embedding_model = "deepseek-embedding-v1"
        local_embedding_model = "shibing624/text2vec-base-chinese"

    cfg = FakeCfg()
    for k, v in overrides.items():
        setattr(cfg, k, v)
    return cfg


# ============================================================
# base_url 规范化
# ============================================================
@pytest.mark.parametrize(
    "raw, expected",
    [
        ("https://api.deepseek.com",         "https://api.deepseek.com/v1"),
        ("https://api.deepseek.com/",        "https://api.deepseek.com/v1"),
        ("https://api.deepseek.com/v1",      "https://api.deepseek.com/v1"),
        ("https://api.deepseek.com/v1/",     "https://api.deepseek.com/v1"),
        ("https://api.deepseek.com/v1/v1",   "https://api.deepseek.com/v1"),
    ],
)
def test_normalize_base_url(raw, expected):
    from src.embeddings import _normalize_base_url
    assert _normalize_base_url(raw) == expected


# ============================================================
# local provider
# ============================================================
def test_local_provider_creates_hf_embeddings(monkeypatch):
    called = {}

    class FakeHF:
        def __init__(self, model_name, encode_kwargs=None):
            called["model_name"] = model_name
            called["encode_kwargs"] = encode_kwargs

    import src.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "HuggingFaceEmbeddings", FakeHF)
    monkeypatch.setattr(emb_mod, "get_config", lambda: _make_cfg())

    result = emb_mod.get_embeddings()
    assert isinstance(result, FakeHF)
    assert called["model_name"] == "shibing624/text2vec-base-chinese"
    assert called["encode_kwargs"] == {"normalize_embeddings": True}


def test_local_model_load_failure_gives_friendly_error(monkeypatch):
    class BoomHF:
        def __init__(self, *a, **kw):
            raise OSError("model not found")

    import src.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "HuggingFaceEmbeddings", BoomHF)
    monkeypatch.setattr(emb_mod, "get_config", lambda: _make_cfg())

    with pytest.raises(RuntimeError, match="本地 embedding 模型加载失败"):
        emb_mod.get_embeddings()


# ============================================================
# deepseek provider
# ============================================================
def test_deepseek_without_key_raises(monkeypatch):
    import src.embeddings as emb_mod
    monkeypatch.setattr(
        emb_mod,
        "get_config",
        lambda: _make_cfg(embedding_provider="deepseek", deepseek_api_key=""),
    )
    with pytest.raises(RuntimeError, match="DEEPSEEK_API_KEY"):
        emb_mod.get_embeddings()


def test_deepseek_creates_openai_embeddings(monkeypatch):
    captured = {}

    class FakeOpenAIEmb:
        def __init__(self, model, api_key, base_url):
            captured["model"] = model
            captured["api_key"] = api_key
            captured["base_url"] = base_url

    import src.embeddings as emb_mod
    monkeypatch.setattr(emb_mod, "OpenAIEmbeddings", FakeOpenAIEmb)
    monkeypatch.setattr(
        emb_mod,
        "get_config",
        lambda: _make_cfg(
            embedding_provider="deepseek",
            deepseek_base_url="https://api.deepseek.com/v1/v1",  # 故意畸形
        ),
    )

    result = emb_mod.get_embeddings()
    assert isinstance(result, FakeOpenAIEmb)
    assert captured["api_key"] == "sk-fake"
    # 关键断言：畸形 URL 被 _normalize_base_url 修正
    assert captured["base_url"] == "https://api.deepseek.com/v1"


def test_illegal_provider_raises(monkeypatch):
    import src.embeddings as emb_mod
    monkeypatch.setattr(
        emb_mod, "get_config",
        lambda: _make_cfg(embedding_provider="ollama"),
    )
    with pytest.raises(ValueError, match="不支持的 embedding_provider"):
        emb_mod.get_embeddings()