"""
DeepSeek API 调用封装。

本项目使用 DeepSeek 的 OpenAI 兼容接口：
base_url = https://api.deepseek.com
model = deepseek-v4-flash
"""

"""
DeepSeek API 调用封装。

本文件重点：
1. 延迟初始化 API Client，避免 import 阶段因为没配 .env 直接崩溃。
2. 支持普通非流式调用。
3. 支持 stream 流式调用。
4. 返回 token 使用量和费用估算。
"""

import os
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional

from dotenv import load_dotenv
from openai import OpenAI


_client: Optional[OpenAI] = None


@dataclass
class TokenUsage:
    """
    Token 使用量和费用估算。
    """

    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    estimated_cost_cny: float = 0.0


@dataclass
class ChatResult:
    """
    非流式调用结果。
    """

    content: str
    usage: TokenUsage


@dataclass
class StreamEvent:
    """
    流式输出事件。

    delta:
        本次流式返回的一小段文本。

    done:
        是否结束。

    result:
        结束时返回完整结果，包括完整回答和 token 统计。
    """

    delta: str = ""
    done: bool = False
    result: Optional[ChatResult] = None


def get_client() -> OpenAI:
    """
    延迟初始化 DeepSeek Client。

    只有真正调用 API 的时候才读取 .env 和创建 OpenAI Client。
    这样 import src.api_client 时不会因为缺少 DEEPSEEK_API_KEY 直接崩溃。
    """

    global _client

    if _client is not None:
        return _client

    load_dotenv()

    api_key = os.getenv("DEEPSEEK_API_KEY")

    if not api_key:
        raise RuntimeError(
            "没有读取到 DEEPSEEK_API_KEY。\n"
            "请检查项目根目录下的 .env 文件，确认里面有：\n"
            "DEEPSEEK_API_KEY=你的DeepSeek_API_Key"
        )

    _client = OpenAI(
        api_key=api_key,
        base_url="https://api.deepseek.com",
    )

    return _client


def get_model_name() -> str:
    """
    获取模型名称。
    """

    load_dotenv()
    return os.getenv("DEEPSEEK_MODEL", "deepseek-chat")


def get_price_config() -> tuple[float, float]:
    """
    获取 token 估算价格。

    单位：
        元 / 100万 tokens

    默认值只是为了方便本地估算。
    你可以在 .env 中按实际价格修改：
        DEEPSEEK_INPUT_PRICE_PER_1M=1.00
        DEEPSEEK_OUTPUT_PRICE_PER_1M=2.00
    """

    load_dotenv()

    input_price = float(os.getenv("DEEPSEEK_INPUT_PRICE_PER_1M", "1.00"))
    output_price = float(os.getenv("DEEPSEEK_OUTPUT_PRICE_PER_1M", "2.00"))

    return input_price, output_price


def build_usage(usage_obj) -> TokenUsage:
    """
    从 API 返回的 usage 对象中提取 token 数，并计算估算费用。
    """

    if usage_obj is None:
        return TokenUsage()

    prompt_tokens = getattr(usage_obj, "prompt_tokens", 0) or 0
    completion_tokens = getattr(usage_obj, "completion_tokens", 0) or 0
    total_tokens = getattr(usage_obj, "total_tokens", 0) or 0

    input_price, output_price = get_price_config()

    estimated_cost = (
        prompt_tokens / 1_000_000 * input_price
        + completion_tokens / 1_000_000 * output_price
    )

    return TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
        estimated_cost_cny=estimated_cost,
    )


def ask_deepseek(messages: List[Dict[str, str]]) -> ChatResult:
    """
    非流式调用 DeepSeek 聊天接口。
    """

    client = get_client()

    response = client.chat.completions.create(
        model=get_model_name(),
        messages=messages,
        temperature=0.2,
        stream=False,
    )

    content = response.choices[0].message.content or ""

    return ChatResult(
        content=content.strip(),
        usage=build_usage(getattr(response, "usage", None)),
    )


def ask_deepseek_stream(
    messages: List[Dict[str, str]],
) -> Generator[StreamEvent, None, None]:
    """
    流式调用 DeepSeek 聊天接口。

    用法：
        for event in ask_deepseek_stream(messages):
            if not event.done:
                print(event.delta, end="", flush=True)
            else:
                print(event.result.usage)
    """

    client = get_client()

    full_content_parts: list[str] = []
    final_usage = TokenUsage()

    stream = client.chat.completions.create(
        model=get_model_name(),
        messages=messages,
        temperature=0.2,
        stream=True,
        stream_options={"include_usage": True},
    )

    for chunk in stream:
        usage_obj = getattr(chunk, "usage", None)
        if usage_obj is not None:
            final_usage = build_usage(usage_obj)

        choices = getattr(chunk, "choices", None)
        if not choices:
            continue

        delta = choices[0].delta
        text = getattr(delta, "content", None)

        if text:
            full_content_parts.append(text)
            yield StreamEvent(delta=text, done=False)

    full_content = "".join(full_content_parts).strip()

    yield StreamEvent(
        done=True,
        result=ChatResult(
            content=full_content,
            usage=final_usage,
        ),
    )