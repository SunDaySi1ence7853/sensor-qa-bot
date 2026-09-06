"""
Embedding 模型工厂。
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.config import get_config


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    """根据配置返回对应的 Embeddings 实例。"""
    cfg = get_config()

    if cfg.embedding_provider == "local":
        try:
            return HuggingFaceEmbeddings(
                model_name=cfg.local_embedding_model,
                encode_kwargs={"normalize_embeddings": True},
            )
        except Exception as e:
            raise RuntimeError(
                f"本地 embedding 模型加载失败：{cfg.local_embedding_model}。"
                f"请确认模型已下载到缓存，或手动下载到 models/ 目录。\n"
                f"原始错误：{e}"
            ) from e

    if cfg.embedding_provider == "deepseek":
        if not cfg.deepseek_api_key:
            raise RuntimeError(
                "使用 DeepSeek embedding 需要设置 DEEPSEEK_API_KEY 环境变量"
            )
        # base_url 已在 config 层规范化为 /v1 结尾，直接用
        return OpenAIEmbeddings(
            model=cfg.deepseek_embedding_model,
            api_key=cfg.deepseek_api_key,
            base_url=cfg.deepseek_base_url,
        )

    raise ValueError(f"不支持的 embedding_provider: {cfg.embedding_provider}")