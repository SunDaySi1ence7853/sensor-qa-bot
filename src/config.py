"""
统一配置。

设计要点：
只在本模块调用一次 load_dotenv()，其他模块从这里取配置，
避免 load_dotenv() 被反复调用。

新增：对关键配置进行边界校验，防止用户误配导致运行时异常。
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


def _parse_int(key: str, default: int, min_val: int = 1, max_val: int | None = None) -> int:
    """
    安全解析整数配置，带边界校验。
    """
    try:
        val = int(os.getenv(key, str(default)))
    except ValueError:
        raise ValueError(
            f"配置 {key} 必须是整数，当前值：{os.getenv(key)}"
        )
    
    if val < min_val:
        raise ValueError(
            f"配置 {key} 不能小于 {min_val}，当前值：{val}"
        )
    
    if max_val is not None and val > max_val:
        raise ValueError(
            f"配置 {key} 不能大于 {max_val}，当前值：{val}"
        )
    
    return val


def _parse_float(key: str, default: float, min_val: float = 0.0) -> float:
    """
    安全解析浮点数配置。
    """
    try:
        val = float(os.getenv(key, str(default)))
    except ValueError:
        raise ValueError(
            f"配置 {key} 必须是数字，当前值：{os.getenv(key)}"
        )
    
    if val < min_val:
        raise ValueError(
            f"配置 {key} 不能小于 {min_val}，当前值：{val}"
        )
    
    return val


@lru_cache(maxsize=1)
def get_config() -> Config:
    chunk_size = _parse_int("CHUNK_SIZE", 500, min_val=50, max_val=2000)
    chunk_overlap = _parse_int("CHUNK_OVERLAP", 80, min_val=0, max_val=chunk_size - 1)
    
    if chunk_overlap >= chunk_size:
        raise ValueError(
            f"CHUNK_OVERLAP ({chunk_overlap}) 必须小于 CHUNK_SIZE ({chunk_size})"
        )
    
    embedding_provider = os.getenv("EMBEDDING_PROVIDER", "local").lower()
    if embedding_provider not in ("local", "deepseek"):
        raise ValueError(
            f"EMBEDDING_PROVIDER 只能是 local 或 deepseek，当前值：{embedding_provider}"
        )
    
    return Config(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        deepseek_base_url=os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com"),
        input_price_per_1m=_parse_float("DEEPSEEK_INPUT_PRICE_PER_1M", 1.00),
        output_price_per_1m=_parse_float("DEEPSEEK_OUTPUT_PRICE_PER_1M", 2.00),
        embedding_provider=embedding_provider,
        local_embedding_model=os.getenv(
            "LOCAL_EMBEDDING_MODEL", "shibing624/text2vec-base-chinese"
        ),
        deepseek_embedding_model=os.getenv(
            "DEEPSEEK_EMBEDDING_MODEL", "deepseek-embedding-v1"
        ),
        retrieve_top_k=_parse_int("RETRIEVE_TOP_K", 3, min_val=1, max_val=20),
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        vectorstore_dir=os.getenv("VECTORSTORE_DIR", "vectorstore"),
        memory_turns=_parse_int("MEMORY_TURNS", 6, min_val=2, max_val=100),
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