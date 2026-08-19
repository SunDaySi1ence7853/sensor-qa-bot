"""
Embedding 封装。

支持两种 provider：
1. local：本地 HuggingFace 模型，离线稳定，默认推荐。
2. deepseek：DeepSeek embedding 接口（需自行验证接口可用）。
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from src.config import get_config, require_api_key


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    cfg = get_config()

    if cfg.embedding_provider == "deepseek":
        return _build_deepseek_embeddings()

    return _build_local_embeddings()


def _build_local_embeddings() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    cfg = get_config()
    return HuggingFaceEmbeddings(
        model_name=cfg.local_embedding_model,
        encode_kwargs={"normalize_embeddings": True},
    )


def _build_deepseek_embeddings() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    require_api_key()
    cfg = get_config()

    return OpenAIEmbeddings(
        model=cfg.deepseek_embedding_model,
        api_key=cfg.deepseek_api_key,
        base_url=cfg.deepseek_base_url + "/v1",
        check_embedding_ctx_length=False,
    )