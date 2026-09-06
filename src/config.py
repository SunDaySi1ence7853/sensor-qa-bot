"""
统一配置。
"""

import os
import re
from dataclasses import dataclass
from functools import lru_cache

from dotenv import load_dotenv

from src.logging_config import get_logger

logger = get_logger(__name__)

load_dotenv()


@dataclass(frozen=True)
class Config:
    deepseek_api_key: str
    deepseek_model: str
    deepseek_base_url: str
    input_price_per_1m: float
    output_price_per_1m: float
    embedding_provider: str
    local_embedding_model: str
    deepseek_embedding_model: str
    retrieve_top_k: int
    chunk_size: int
    chunk_overlap: int
    vectorstore_dir: str
    memory_turns: int


def _normalize_base_url(url: str) -> str:
    """
    末尾规范化为 /v1，防止 LangChain 重复拼接。

    https://api.deepseek.com        -> https://api.deepseek.com/v1
    https://api.deepseek.com/v1     -> https://api.deepseek.com/v1
    https://api.deepseek.com/v1/v1  -> https://api.deepseek.com/v1
    """
    url = url.strip().rstrip("/")
    url = re.sub(r"(/v1)+$", "", url)
    return url + "/v1"


def _parse_int(key: str, default: int, min_val: int = 1, max_val: int | None = None) -> int:
    try:
        val = int(os.getenv(key, str(default)))
    except ValueError:
        logger.error("配置 %s 不是整数：%r", key, os.getenv(key))
        raise ValueError(f"配置 {key} 必须是整数，当前值：{os.getenv(key)}")

    if val < min_val:
        logger.error("配置 %s=%d 小于最小值 %d", key, val, min_val)
        raise ValueError(f"配置 {key} 不能小于 {min_val}，当前值：{val}")

    if max_val is not None and val > max_val:
        logger.error("配置 %s=%d 大于最大值 %d", key, val, max_val)
        raise ValueError(f"配置 {key} 不能大于 {max_val}，当前值：{val}")

    return val


def _parse_float(key: str, default: float, min_val: float = 0.0) -> float:
    try:
        val = float(os.getenv(key, str(default)))
    except ValueError:
        logger.error("配置 %s 不是数字：%r", key, os.getenv(key))
        raise ValueError(f"配置 {key} 必须是数字，当前值：{os.getenv(key)}")

    if val < min_val:
        logger.error("配置 %s=%f 小于最小值 %f", key, val, min_val)
        raise ValueError(f"配置 {key} 不能小于 {min_val}，当前值：{val}")

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
        logger.error("EMBEDDING_PROVIDER 非法：%r", embedding_provider)
        raise ValueError(
            f"EMBEDDING_PROVIDER 只能是 local 或 deepseek，当前值：{embedding_provider}"
        )

    cfg = Config(
        deepseek_api_key=os.getenv("DEEPSEEK_API_KEY", ""),
        deepseek_model=os.getenv("DEEPSEEK_MODEL", "deepseek-chat"),
        # 统一在配置层规范化，llm.py 和 embeddings.py 直接用，无需各自处理
        deepseek_base_url=_normalize_base_url(
            os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com")
        ),
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
        memory_turns=_parse_int("MEMORY_TURNS", 6, min_val=1, max_val=100),
    )

    logger.debug(
        "配置加载完成 | model=%s | provider=%s | top_k=%d | memory_turns=%d",
        cfg.deepseek_model,
        cfg.embedding_provider,
        cfg.retrieve_top_k,
        cfg.memory_turns,
    )

    return cfg


def require_api_key() -> str:
    cfg = get_config()
    if not cfg.deepseek_api_key:
        logger.error("DEEPSEEK_API_KEY 缺失")
        raise RuntimeError(
            "没有读取到 DEEPSEEK_API_KEY。\n"
            "请检查项目根目录下的 .env 文件是否存在且格式正确。"
        )
    return cfg.deepseek_api_key