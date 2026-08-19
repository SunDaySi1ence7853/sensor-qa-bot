"""
统一配置。

设计要点：
只在本模块调用一次 load_dotenv()，其他模块从这里取配置，
避免 load_dotenv() 被反复调用。
"""

import os
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

load_dotenv()


@dataclass(frozen=True)
class Config:
    # DeepSeek Chat
    deepseek_api_key: str
    deepseek_model: str
    deepseek_base_url: str

    # 费用估算
    input_price_per_1m: float
    output_price_per_1m: float

    # embedding
    embedding_provider: str
    local_embedding_model: str
    deepseek_embedding_model: str

    # RAG
    retrieve_top_k: int
    chunk_size: int
    chunk_overlap: int
    vectorstore_dir: str

    # memory
    memory_turns: int


@lru_cache(maxsize=1)
def get_config() -> Config:
    return Config(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        input_price_per_1m=float(os.getenv("DEEPSEEK_INPUT_PRICE_PER_1M", "1.00")),
        output_price_per_1m=float(os.getenv("DEEPSEEK_OUTPUT_PRICE_PER_1M", "2.00")),
        embedding_provider=os.getenv("EMBEDDING_PROVIDER", "local").lower(),
        local_embedding_model=os.getenv(
            "LOCAL_EMBEDDING_MODEL", "shibing624/text2vec-base-chinese"
        ),
        deepseek_embedding_model=os.getenv(
            "DEEPSEEK_EMBEDDING_MODEL", "deepseek-embedding-v1"
        ),
        retrieve_top_k=int(os.getenv("RETRIEVE_TOP_K", "3")),
        chunk_size=int(os.getenv("CHUNK_SIZE", "500")),
        chunk_overlap=int(os.getenv("CHUNK_OVERLAP", "80")),
        vectorstore_dir=os.getenv("VECTORSTORE_DIR", "vectorstore"),
        memory_turns=int(os.getenv("MEMORY_TURNS", "6")),
    )


def require_api_key() -> str:
    """
    延迟校验：只有真正要调 DeepSeek 时才检查 key。
    """
    cfg = get_config()
    if not cfg.deepseek_api_key:
        raise RuntimeError(
            "没有读取到 DEEPSEEK_API_KEY。\n"
            "请检查项目根目录下的 .env 文件。"
        )
    return cfg.deepseek_api_key