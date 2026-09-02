"""
DeepSeek Chat 的 LangChain 封装。

延迟初始化：只有第一次调用 get_llm() 时才创建 ChatOpenAI 并校验 API Key。
"""

from functools import lru_cache

from langchain_openai import ChatOpenAI

from src.config import get_config, require_api_key


@lru_cache(maxsize=2)
def get_llm(streaming: bool = False) -> ChatOpenAI:
    """
    获取 LangChain ChatOpenAI 实例（指向 DeepSeek）。

    DeepSeek 兼容 OpenAI 接口，所以用 ChatOpenAI + base_url 即可。
    streaming=True / False 各缓存一个实例。

    stream_usage=True：
        让 ChatOpenAI 在流式模式下也请求 usage（等价于底层
        stream_options={"include_usage": True}），
        这样最后一个 chunk 的 usage_metadata 才会被填充，
        否则流式统计到的 token 会一直是 0。
    """
    require_api_key()
    cfg = get_config()

    return ChatOpenAI(
        model=cfg.deepseek_model,
        api_key=cfg.deepseek_api_key,
        base_url=cfg.deepseek_base_url,
        temperature=0.2,
        streaming=streaming,
        stream_usage=streaming,  # 仅流式时需要
    )