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

from langchain_core.embeddings import Embeddings

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
def get_embeddings() -> Embeddings:
    """
    根据配置返回对应的 Embeddings 实例。
    使用 lru_cache 确保单例。
    """
    cfg = get_config()

    if cfg.embedding_provider == "local":
        try:
            return HuggingFaceEmbeddings(
                model_name=cfg.local_embedding_model,
                model_kwargs={"device": "cpu"},
            )
        except Exception as e:
            raise RuntimeError(
                f"本地模型 {cfg.local_embedding_model} 加载失败。"
                f"请确认模型已下载到缓存，或手动下载到 models/ 目录。\n"
                f"原始错误：{e}"
            ) from e

    if cfg.embedding_provider == "deepseek":
        if not cfg.deepseek_api_key:  # ← 这里改了
            raise RuntimeError(
                "使用 DeepSeek embedding 需要设置 DEEPSEEK_API_KEY 环境变量"
            )
        return OpenAIEmbeddings(
            model=cfg.deepseek_embedding_model,
            openai_api_key=cfg.deepseek_api_key,
            openai_api_base=_normalize_base_url(cfg.deepseek_base_url),
        )

    raise ValueError(
        f"未知的 embedding_provider: {cfg.embedding_provider}"
    )