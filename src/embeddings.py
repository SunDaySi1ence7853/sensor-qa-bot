"""
Embedding 模型工厂。

支持两种 provider：
- local: 使用 HuggingFace 本地模型（默认 shibing624/text2vec-base-chinese）
- deepseek: 使用 DeepSeek 的 embedding API

防御性改动：
- base_url 统一规范化，避免用户配置 https://api.deepseek.com/v1 时
  拼出 /v1/v1 导致 404。
"""

import re
from functools import lru_cache

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_openai import OpenAIEmbeddings

from src.config import get_config


def _normalize_base_url(url: str) -> str:
    """
    统一规范化 base_url，确保末尾恰好是 /v1。

    支持的输入形式（全部归一化为 https://api.deepseek.com/v1）：
        https://api.deepseek.com
        https://api.deepseek.com/
        https://api.deepseek.com/v1
        https://api.deepseek.com/v1/
        https://api.deepseek.com/v1/v1

    如果不做这一步：
    用户在 .env 里配置 DEEPSEEK_BASE_URL=https://api.deepseek.com/v1
    时，OpenAIEmbeddings 内部还会再拼一次 /v1，最终请求
    https://api.deepseek.com/v1/v1/embeddings，直接 404，
    且错误信息不会告诉用户是 base_url 配错了。
    """
    url = url.strip().rstrip("/")
    # 去掉末尾一个或多个 /v1
    url = re.sub(r"(/v1)+$", "", url)
    return url + "/v1"


@lru_cache(maxsize=1)
def get_embeddings():
    """
    根据配置返回 embedding 实例。
    结果被缓存，避免重复加载模型。
    """
    cfg = get_config()

    if cfg.embedding_provider == "local":
        return HuggingFaceEmbeddings(
            model_name=cfg.local_embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )

    if cfg.embedding_provider == "deepseek":
        if not cfg.api_key:
            raise ValueError(
                "使用 deepseek embedding 需要设置 DEEPSEEK_API_KEY"
            )
        return OpenAIEmbeddings(
            model=cfg.deepseek_embedding_model,
            api_key=cfg.api_key,
            base_url=_normalize_base_url(cfg.base_url),
        )

    raise ValueError(
        f"不支持的 embedding_provider: {cfg.embedding_provider}，"
        f"可选值为 'local' 或 'deepseek'"
    )