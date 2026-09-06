"""
Embedding 模型工厂。
"""

import re
from functools import lru_cache

from langchain_core.embeddings import Embeddings
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.config import get_config


def _normalize_base_url(url: str) -> str:
    """
    规范化 base_url，确保以 /v1 结尾，且不会重复添加。
    """
    if not url:
        return ""
    url = url.strip().rstrip("/")
    # 去掉结尾所有的 /v1 或 /v1/ 防止重复
    url = re.sub(r"(/v1)+$", "", url)
    return url + "/v1"


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
        
        # 防御性二次规范化，避免 config 层没拦住的情况
        base_url = _normalize_base_url(cfg.deepseek_base_url)
        
        return OpenAIEmbeddings(
            model=cfg.deepseek_embedding_model,
            api_key=cfg.deepseek_api_key,
            base_url=base_url,
        )

    raise ValueError(f"不支持的 embedding_provider: {cfg.embedding_provider}")