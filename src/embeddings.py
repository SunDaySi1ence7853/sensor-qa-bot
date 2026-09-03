"""
Embedding 封装。
"""

from functools import lru_cache

from langchain_core.embeddings import Embeddings

from src.config import get_config, require_api_key
from src.logging_config import get_logger

logger = get_logger(__name__)


@lru_cache(maxsize=1)
def get_embeddings() -> Embeddings:
    cfg = get_config()

    if cfg.embedding_provider == "deepseek":
        logger.info("使用 DeepSeek embedding：%s", cfg.deepseek_embedding_model)
        return _build_deepseek_embeddings()

    logger.info("使用本地 embedding：%s", cfg.local_embedding_model)
    return _build_local_embeddings()


def _build_local_embeddings() -> Embeddings:
    from langchain_huggingface import HuggingFaceEmbeddings

    cfg = get_config()

    try:
        logger.debug("正在初始化 HuggingFaceEmbeddings...")
        emb = HuggingFaceEmbeddings(
            model_name=cfg.local_embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
        logger.debug("HuggingFaceEmbeddings 初始化完成")
        return emb
    except Exception as e:
        logger.exception("本地 embedding 模型加载失败：%s", cfg.local_embedding_model)
        raise RuntimeError(
            f"本地 embedding 模型加载失败：{cfg.local_embedding_model}\n"
            f"原因：{e}\n"
            f"请检查模型名是否正确，或首次运行时等待自动下载完成。"
        )


def _build_deepseek_embeddings() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    require_api_key()
    cfg = get_config()

    base = cfg.deepseek_base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"

    logger.debug("DeepSeek embedding base_url=%s", base)

    try:
        return OpenAIEmbeddings(
            model=cfg.deepseek_embedding_model,
            api_key=cfg.deepseek_api_key,
            base_url=base,
            check_embedding_ctx_length=False,
        )
    except Exception as e:
        logger.exception("DeepSeek embedding 初始化失败")
        raise RuntimeError(
            f"DeepSeek embedding 接口调用失败：{e}\n"
            f"建议切换回本地模式：在 .env 里设置 EMBEDDING_PROVIDER=local"
        )