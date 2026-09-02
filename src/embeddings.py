"""
Embedding 封装。

支持两种 provider：
1. local：本地 HuggingFace 模型，离线稳定，默认推荐。
2. deepseek：DeepSeek embedding 接口（需自行验证接口可用）。

新增：捕获常见错误，给出清晰的用户指引。
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
    
    try:
        return HuggingFaceEmbeddings(
            model_name=cfg.local_embedding_model,
            encode_kwargs={"normalize_embeddings": True},
        )
    except Exception as e:
        raise RuntimeError(
            f"本地 embedding 模型加载失败：{cfg.local_embedding_model}\n"
            f"原因：{e}\n"
            f"请检查模型名是否正确，或首次运行时等待自动下载完成。\n"
            f"推荐模型：shibing624/text2vec-base-chinese"
        )


def _build_deepseek_embeddings() -> Embeddings:
    from langchain_openai import OpenAIEmbeddings

    require_api_key()
    cfg = get_config()

    # rstrip("/") 防止用户在 .env 里把 base_url 写成带 /v1 或带结尾斜杠，
    # 拼成 https://.../v1/v1 这种错误地址。
    base = cfg.deepseek_base_url.rstrip("/")
    if not base.endswith("/v1"):
        base = base + "/v1"

    try:
        return OpenAIEmbeddings(
            model=cfg.deepseek_embedding_model,
            api_key=cfg.deepseek_api_key,
            base_url=base,
            check_embedding_ctx_length=False,
        )
    except Exception as e:
        raise RuntimeError(
            f"DeepSeek embedding 接口调用失败：{e}\n"
            f"请确认 DEEPSEEK_API_KEY 和 DEEPSEEK_BASE_URL 配置正确，\n"
            f"且 DeepSeek 官方支持 embedding 接口（部分文档存在矛盾）。\n"
            f"建议切换回本地模式：在 .env 里设置 EMBEDDING_PROVIDER=local"
        )